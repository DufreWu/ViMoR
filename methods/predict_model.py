import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
import os
import time
import numpy as np 
from llama_cpp import Llama
import pickle
import joblib

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.robot_env import RobotEnv

CONFIG_FILE = "./config_eval.yaml"
with open(CONFIG_FILE, "r") as f:
    config = yaml.safe_load(f)

FC_SET = config["jetson_orin_nx_16g"]["cpu_freqs"]
FG_SET = config["jetson_orin_nx_16g"]["gpu_freqs"]
V_SET = np.arange(config["motor"]["v_min"], config["motor"]["v_max"]+config["motor"]["v_step"], config["motor"]["v_step"])

class EGTPHead(nn.Module):
    #     Loss (combined in EGTPTrainer)
    # ------------------------------
    #   ℒ = λ · KL(log_softmax(logits) ‖ soft_labels(y))
    #     + (1−λ) · MSE(pred, y)

    def __init__(self, input_dim, num_bins=20, target_range=(0.0, 2048.0), dropout=0.1):
        super().__init__()
        self.num_bins = num_bins
        self.min_val, self.max_val = target_range
        hidden_dim = input_dim // 2
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_bins),
        )
        width = (self.max_val - self.min_val) / self.num_bins
        centers = self.min_val + (torch.arange(num_bins).float() + 0.5) * width
        self.register_buffer("bin_centers", centers.unsqueeze(0))

    def forward(self, x: torch.Tensor):
        logits = self.fc(x)
        probs = F.softmax(logits, dim=-1)
        pred = torch.sum(probs * self.bin_centers, dim=-1)
        return logits, pred

class LatencyModel:
    def __init__(self, path):
        print(path)
        with open(path, "rb") as f:
            self.model = joblib.load(f)

    def predict(self, fc, fg, n_p, n_d):
        X = np.array([[fc, fg, n_p, n_d]])
        return float(self.model.predict(X)[0])

class EnergyModel:
    def comp_power(self, fc, fg):
        return 0.5 * (fc / 300) ** 2 + 0.3 * (fg / 300) ** 2

    def mech_power(self, v):
        return 1.14005 * v + 0.10893 * v ** 3 + 1.1625


class PredictModelPolicy:
    def __init__(self, model_path, decode_predictor_path, latency_model_path, robot_env):
        self.llm = Llama(model_path=model_path, embedding=True, verbose=False)
        input_dim = self.llm.n_embd()

        self.decode_predictor = EGTPHead(input_dim=input_dim)
        state_dict = torch.load(decode_predictor_path, weights_only=True)
        
        self.decode_predictor.load_state_dict(state_dict)
        self.decode_predictor.eval()

        self.robot = robot_env
        self.latency_model = LatencyModel(latency_model_path)
        self.energy_model = EnergyModel()
    
    def predict_decode(self, prompt):
        emb_data = self.llm.create_embedding(prompt)['data']

        last_vec = torch.tensor(emb_data[-1]['embedding'][:896]).float().unsqueeze(0)
        last_vec = last_vec.to(next(self.decode_predictor.parameters()).device)
        last_vec = last_vec / (last_vec.norm(dim=-1, keepdim=True) + 1e-6)

        with torch.no_grad():
            logits, _ = self.decode_predictor(last_vec)

        probs = torch.softmax(logits, dim=-1)

        bin_centers = self.decode_predictor.bin_centers.to(probs.device)

        n_vals = (probs * bin_centers).sum(dim=-1)
        n_val = n_vals.mean().item()
        n_val = int(n_val)

        prompt_len = len(prompt.split())

        # detect saturation / abnormal prediction
        if n_val > 200 or n_val < 1:
            print("[Warning] Abnormal decode prediction, applying fallback")
            n_val = min(64, max(8, prompt_len * 2))

        # hard safety bound
        n_val = max(1, min(128, n_val))

        return n_val
    
    def act(self, prompt, d_safe):
        # Get predicted decode length
        n_p = len(self.llm.tokenize(prompt.encode('utf-8')))
        n_d = self.predict_decode(prompt)
        print(f"Predicted Tokens: {n_d}")

        # Find optimal hardware knobs
        action = self.search_optimal(n_p, n_d, d_safe)
        print(f"Action: {action}")

        # Apply and generate
        self.apply_hardware_knobs(action)

        output = self.llm(prompt, max_tokens=n_d) 

        return action, output
    
    def search_optimal(self, n_p, n_d, d_safe):
        best_cost = float("inf")
        best_action = None

        for fc in FC_SET:
            for fg in FG_SET:
                for v in V_SET:
                    # 1. latency prediction
                    latency = self.latency_model.predict(fc, fg, n_p, n_d)

                    # 2. latency constraint (critical)
                    t_motion = d_safe / v
                    if latency > t_motion:
                        continue

                    comp_power = self.energy_model.comp_power(fc, fg)
                    mech_power = self.energy_model.mech_power(v)

                    total_energy = comp_power * latency + mech_power * t_motion
                    cost = total_energy / d_safe

                    if cost < best_cost:
                        best_cost = cost
                        best_action = (fc, fg, v)

        # fallback
        if best_action is None:
            return max(FC_SET), max(FG_SET), min(V_SET)

        return best_action
    
    def apply_hardware_knobs(self, action):
        fc, fg, v = action
        self.robot.set_speed(v)
        for cpu_core in range(4):
            self.robot.set_board_cpu_dvfs(cpu_core, fc)
        self.robot.set_board_gpu_dvfs(fg)

        print(f"[Strategy: Proactive Predict Policy] CPU={fc}MHz, GPU={fg}MHz, Speed={v}m/s")
    
    def close(self):
        if self.llm is not None:
            self.llm.close()
            self.llm = None

if __name__ == "__main__":
    LLM_PATH = "./llm_models/qwen2.5-3b-instruct-q4_k_m.gguf"
    PREDICTOR_PATH = "./decode_predictor/outputs/checkpoints/final_head.pt"
    LATENCY_PATH = "./latency_fitting/latency_predictor.pkl"
    ROBOT = RobotEnv(config, is_sim=True)

    policy = PredictModelPolicy(LLM_PATH, PREDICTOR_PATH, LATENCY_PATH, ROBOT)
    
    current_prompt = "Report any leaks and solve this problem."
    distance_to_poi = 5.0 # meters
    
    action, result = policy.act(current_prompt, distance_to_poi)

    print(f"Optimal Action: {action}")
    print(f"LLM Response: {result['choices'][0]['text']}")
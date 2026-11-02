import yaml
import os
import numpy as np
from llama_cpp import Llama

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

CONFIG_FILE = "./config_eval.yaml"
with open(CONFIG_FILE, "r") as f:
    config = yaml.safe_load(f)

FC_SET = config["jetson_orin_nx_16g"]["cpu_freqs"]
FG_SET = config["jetson_orin_nx_16g"]["gpu_freqs"]
V_SET = np.arange(config["motor"]["v_min"], config["motor"]["v_max"]+config["motor"]["v_step"], config["motor"]["v_step"])

class SafeSpeedPolicy:
    """Strategy A: Maximize GPU/CPU frequency and Velocity."""
    def __init__(self, model_path, robot_env):
        self.llm = Llama(model_path=model_path, embedding=False, verbose=False)
        self.robot = robot_env

    def act(self, prompt, d_safe):
        # Select the absolute maximums from your config sets
        fc = FC_SET[len(FC_SET)//2]
        fg = FG_SET[len(FG_SET)//2]
        v = V_SET[0]
        
        self.robot.set_speed(v)
        for cpu_core in range(config["jetson_orin_nx_16g"]["cpu_cores_num"]):
            self.robot.set_board_cpu_dvfs(cpu_core, fc)
        self.robot.set_board_gpu_dvfs(fg)
        
        print(f"[Strategy: MaxPerf] CPU={fc}MHz, GPU={fg}MHz, Speed={v}m/s")
        output = self.llm(prompt, max_tokens=128) 

        return (fc, fg, v), output
    
    def close(self):
        if self.llm is not None:
            self.llm.close()
            self.llm = None
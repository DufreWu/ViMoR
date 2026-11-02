import time
import yaml
import pandas as pd
import torch
import os
import json
from utils.robot_env import RobotEnv
from utils.monitor import RealTimeMonitor
from methods.max_pref import MaxPerfPolicy
from methods.predict_model import PredictModelPolicy, LatencyModel
from methods.stop_wait import StopWaitPolicy
from methods.safe_speed import SafeSpeedPolicy

# Configuration Loading
CONFIG_FILE = "./config_eval.yaml"
with open(CONFIG_FILE, "r") as f:
    cfg = yaml.safe_load(f)

# =========================
# Load test dataset
# =========================
def load_test_samples(path):
    samples = []
    with open(path, "r") as f:
        for line in f:
            item = json.loads(line)
            prompt = item["conversations"][0]["value"]
            n_p = item["prefill_length"]
            n_d = item["decode_length"]
            samples.append((prompt, n_p, n_d))
    return samples

def run_experiment(name, policy, robot, prompt, dist, n_p, n_d, latency_model, is_sim=False):
    print(f"\n>>> Strategy: {name} | n_p={n_p}, n_d={n_d}")
    
    # 1. Setup
    robot.set_speed(0)
    time.sleep(1)

    monitor = RealTimeMonitor(robot)
    monitor.start()

    start_time = time.time()

    # =========================
    # 2. LLM + Policy Execution
    # =========================
    if is_sim == False:
        start_llm = time.time()
        (fc, fg, v), output = policy.act(prompt, dist)
        t_llm_finish = time.time() - start_llm

        print(f"  [Cyber] Inference finished at {t_llm_finish:.2f}s")

        # =========================
        # 3. Physical Execution
        # =========================
        t_motion_arrival = dist / v
        robot_reached_goal = False

        while True:
            elapsed = time.time() - start_time

            if not robot_reached_goal and elapsed >= t_motion_arrival:
                robot.set_speed(0)
                robot_reached_goal = True
                print(f"  [Physical] Goal reached at {elapsed:.2f}s")

            # stop when both done
            if robot_reached_goal and elapsed >= t_llm_finish:
                break

            time.sleep(0.01)

        # =========================
        # 4. Finalize
        # =========================
        total_t = time.time() - start_time

        monitor_data = monitor.stop()
        monitor.join()

        idle_t = max(0, t_llm_finish - t_motion_arrival)
    else: # is_sim == True:
        # 1. Policy decision (no real execution)
        (fc, fg, v), output = policy.act(prompt, dist)
        for cpu in range(4):
            robot.set_board_cpu_dvfs(cpu, fc)
        robot.set_board_gpu_dvfs(fg)
        robot.set_speed(v)

        t_llm_finish = latency_model.predict(fc, fg, n_p, n_d)
        t_motion_arrival = dist / v
        total_t = max(t_llm_finish, t_motion_arrival)
        idle_t = max(0, t_llm_finish - t_motion_arrival)
        total_energy = robot.get_robot_power() * total_t
        

    return {
        "Strategy": name,
        "speed": v,
        "motion_t": t_motion_arrival,
        "latency_t": t_llm_finish,
        "total_t": total_t,
        "idle_t": idle_t,
        "total_energy": total_energy,
        "E_d": total_energy / dist  # 🔥 energy per distance
    }

if __name__ == "__main__":
    # Ensure CUDA is ready if needed
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    LLM_PATH = "./llm_models/qwen2.5-3b-instruct-q4_k_m.gguf"
    LATENCY_PATH = "./latency_fitting/latency_predictor.pkl"
    PREDICTOR_PATH = "./methods/checkpoints/outputs/full_eval/qwen2.5_0.5b/Reasoning/checkpoints/final_head.pt"
    DIST = 10.0 
    TEST_FILE = "./decode_predictor/datasets/test.jsonl"
    CSV_FILE = "experiment_results.csv"

    robot = RobotEnv(cfg, is_sim=True)
    latency_model = LatencyModel(LATENCY_PATH)
    
    test_cases = {
        "MaxPerf": MaxPerfPolicy,
        "StopWait": StopWaitPolicy,
        "SafeSpeed": SafeSpeedPolicy,
        # "Ours": PredictModelPolicy
    }

    # load dataset
    samples = load_test_samples(TEST_FILE)

    try:
        for i, (prompt, n_p, n_d) in enumerate(samples):

            print("\n" + "=" * 60)
            print(f"Sample {i} | Prefill={n_p}, Decode={n_d}")

            for name, PolicyClass in test_cases.items():

                policy = PolicyClass(LLM_PATH, robot)

                try:
                    stats = run_experiment(
                        name,
                        policy,
                        robot,
                        prompt,
                        DIST,
                        n_p,
                        n_d,
                        latency_model,
                        is_sim=True
                    )

                    # =========================
                    # Save immediately
                    # =========================
                    df = pd.DataFrame([stats])

                    df.to_csv(
                        CSV_FILE,
                        mode='a',
                        header=not os.path.exists(CSV_FILE),
                        index=False
                    )

                    print(f"[Saved] {name}")

                finally:
                    if hasattr(policy, "close"):
                        policy.close()

    except KeyboardInterrupt:
        print("\nInterrupted by user.")

    finally:
        print("\n" + "=" * 60)
        print("Simulation COMPLETE. Results saved to experiment_results.csv")
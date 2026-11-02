import os
import time
import yaml
import csv
import subprocess
import re
import sys
import statistics
import numpy as np

# Ensure pathing for local utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.jetson_orin_nx import Jetson_Orin_NX

# ===============================
# Settings & Validation
# ===============================
CONFIG_FILE = "./config_eval.yaml"

def fail(msg):
    print(f"[ERROR] {msg}")
    sys.exit(1)

if not os.path.isfile(CONFIG_FILE):
    fail(f"Config file not found: {CONFIG_FILE}")

with open(CONFIG_FILE, "r") as f:
    config = yaml.safe_load(f)

# Environment Setup
llama_bin = config["llama_dir"]["bench"]
model_path = config["model"]["qwen2.5_3b"] # Ensure this exists in config
repeat_count = config["test_param"]["repeat"]
board = Jetson_Orin_NX(config)

if os.geteuid() != 0:
    fail("This script must be run with sudo to control frequencies.")

# ===============================
# Data Collection Matrix
# ===============================
# We need diversity in lp and ld to solve for beta_1 and beta_2
lp_set = [32, 128, 512]
ld_set = [32, 64, 128, 256]
workload_sweep = [(lp, ld) for lp in lp_set for ld in ld_set]

# ===============================
# Metrics Parser
# ===============================
def parse_llama_metrics(output, p_tokens, n_tokens):
    """
    Parse llama-bench output: pp (prefill) and tg (decode) throughput
    """
    pp_tps = None
    tg_tps = None

    # Match: | llama ... | pp32 | 379.41 ± 14.66 |
    pp_match = re.search(r"pp" + str(p_tokens) + r"\s*\|\s*([0-9]+\.[0-9]+)", output)
    # Match: | llama ... | tg32 | 18.10 ± 0.02 |
    tg_match = re.search(r"tg" + str(n_tokens) + r"\s*\|\s*([0-9]+\.[0-9]+)", output)

    if pp_match:
        pp_tps = float(pp_match.group(1))
    if tg_match:
        tg_tps = float(tg_match.group(1))

    # calculate ttft and tpot
    ttft = p_tokens / pp_tps if pp_tps and pp_tps > 0 else 0
    tpot = 1.0 / tg_tps if tg_tps and tg_tps > 0 else 0
    
    return pp_tps, tg_tps, ttft, tpot

# ===============================
# Benchmark Execution
# ===============================
def run_benchmark(p_tokens, n_tokens):
    cmd = [
        "taskset", "-c", "0-3",
        llama_bin, 
        "-m", model_path,
        "-p", str(p_tokens), 
        "-n", str(n_tokens),
        "-r", "1", 
        "-ngl", "99"
    ]

    # Clean cache
    os.system("sync; echo 3 > /proc/sys/vm/drop_caches")

    start_time = time.time()
    process = subprocess.Popen(
        cmd, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.STDOUT, 
        text=True
    )

    power_trace = []
    time_trace = []

    while process.poll() is None:
        power_trace.append(board.read_power())
        time_trace.append(time.time() - start_time)
        time.sleep(0.02)

    stdout, _ = process.communicate()
    
    pp_tps, tg_tps, ttft, tpot = parse_llama_metrics(stdout, p_tokens, n_tokens)
    print(pp_tps, tg_tps, ttft, tpot)

    energy_pp = 0
    energy_tg = 0

    if ttft and len(power_trace) > 1:
        t_arr = np.array(time_trace)
        p_arr = np.array(power_trace)
        p_board = np.maximum(0, p_arr) 

        idx_split = np.searchsorted(t_arr, ttft)

        # Prefill 能量 (0 到 ttft)
        if idx_split > 0:
            energy_pp = np.trapz(p_board[:idx_split], x=t_arr[:idx_split])
        
        # Decode 能量 (ttft 到结束)
        if idx_split < len(t_arr):
            energy_tg = np.trapz(p_board[idx_split:], x=t_arr[idx_split:])

        e2e_latency = tpot*(n_tokens-1) + ttft
    return {
        "l_p": p_tokens,
        "l_d": n_tokens,
        "ttft": ttft,
        "tpot": tpot,
        "pp_tps": pp_tps,
        "tg_tps": tg_tps,
        "e2e_latency": e2e_latency,
        "energy_total": energy_pp + energy_tg
    }

# ===============================
# Main Loop
# ===============================
def main():
    RESULT_FILE = f"./latency_fitting/latency_fitting_data.csv"
    os.makedirs(os.path.dirname(RESULT_FILE), exist_ok=True)

    with open(RESULT_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        # Header for Regression: Target = ttft (for b1) or tpot (for b2)
        writer.writerow([
            "cpu_freq_hz", "gpu_freq_hz", "l_p", "l_d",
            "ttft_s", "tpot_s", "e2e_latency", "energy_total"
        ])

        board.set_mem_default_dvfs()
        
        # Frequency Sweeps
        for gf in board.gpu_freqs[::2]:
            board.set_gpu_freq(gf)
            for cf in board.cpu_freqs[::2]:
                board.set_cpu_freq(0, cf)

                print(f"=== TESTING: CPU {cf/1e3:.1f}MHz | GPU {gf/1e6:.1f}MHz")
                time.sleep(2) # Thermal cool-down

                for pd_pair in workload_sweep:
                    p_len, n_len = pd_pair
                    
                    print(f"=== n_p: {p_len} | n_d: {n_len}")
                    res = run_benchmark(p_len, n_len)
                    print(res)
                    if res["pp_tps"] is None: continue

                    writer.writerow([
                        board.read_cpu_freq(0), 
                        board.read_gpu_freq(),
                        res["l_p"],
                        res["l_d"],
                        res["ttft"],
                        res["tpot"],
                        res["e2e_latency"],
                        res["energy_total"],
                    ])
                    f.flush()
                    print(f"  lp={p_len}, ld={n_len} -> TTFT: {res['ttft']:.3f}s, TPOT: {res['tpot']:.4f}s")

    print(f"\n[DONE] Data collection complete. File: {RESULT_FILE}")

if __name__ == "__main__":
    main()
import subprocess
import time
import os
import re
import csv
import signal
import yaml
import sys
import numpy as np

# --- Setup ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.jetson_orin_nx import Jetson_Orin_NX

CONFIG_FILE = "./config_eval.yaml"
with open(CONFIG_FILE, "r") as f:
    config = yaml.safe_load(f)

llama_bin = config["llama_dir"]["bench"]
model_path = config["model"]["qwen2.5_3b"]
board = Jetson_Orin_NX(config)

OUTPUT_CSV = "./power_model/power_dataset.csv"

# ===============================
# --------- PARSER --------------
# ===============================
def parse_tegrastats(raw_text):
    p_in = [int(x) for x in re.findall(r"VDD_IN (\d+)mW", raw_text)]
    p_core = [int(x) for x in re.findall(r"VDD_CPU_GPU_CV (\d+)mW", raw_text)]
    g_util = [int(x) for x in re.findall(r"GR3D_FREQ (\d+)%", raw_text)]

    cpu_history = []
    cpu_lines = re.findall(r"CPU \[([\d%@,off]+)\]", raw_text)

    for line in cpu_lines:
        cores = line.split(',')
        snapshot = []
        for c in cores:
            if '%' in c:
                u, f = c.split('%@')
                snapshot.append(int(u))
            else:
                snapshot.append(0)
        cpu_history.append(snapshot)

    def avg(lst):
        return float(np.mean(lst)) if lst else 0.0

    metrics = {
        "avg_p_in_mw": avg(p_in),
        "avg_p_core_mw": avg(p_core),
        "avg_gpu_util": avg(g_util),
    }

    if cpu_history:
        metrics["avg_cpu_util"] = float(np.mean(cpu_history))
    else:
        metrics["avg_cpu_util"] = 0.0

    return metrics


# ===============================
# -------- MONITOR --------------
# ===============================
def start_tegrastats():
    return subprocess.Popen(
        ["tegrastats", "--interval", "100"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        bufsize=1
    )


def stop_tegrastats(proc):
    proc.send_signal(signal.SIGINT)
    out, _ = proc.communicate()
    return out


# ===============================
# -------- WORKLOADS ------------
# ===============================
def run_llm(p_tokens, n_tokens):
    cmd = [
        "taskset", "-c", "0-7",
        llama_bin,
        "-m", model_path,
        "-p", str(p_tokens),
        "-n", str(n_tokens),
        "-ngl", "99",
    ]

    print(f"[LLM] Running: P={p_tokens}, N={n_tokens}")

    start = time.time()

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,   # avoid buffer issues
        stderr=subprocess.PIPE,
        text=True
    )

    # 🔥 BLOCK until finished
    _, stderr = proc.communicate()

    end = time.time()
    duration = end - start

    # 🔴 check failure
    if proc.returncode != 0:
        print("LLM failed!")
        print(stderr)
        raise RuntimeError("LLM process failed")

    return duration

def run_cpu_stress(duration=5):
    proc = subprocess.Popen(["stress-ng", "--cpu", "4", "--timeout", str(duration)])
    proc.wait()


def run_idle(duration=5):
    time.sleep(duration)


# ===============================
# -------- MAIN RUN -------------
# ===============================
def run_experiment(mode, cf, gf, p_tokens=0, n_tokens=0):

    # Set DVFS
    board.set_gpu_freq(gf)
    board.set_cpu_freq(0, cf)

    os.system("sync; sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'")
    time.sleep(1)

    stats_proc = start_tegrastats()

    start = time.time()

    # -------- workload --------
    if mode == "llm":
        duration = run_llm(p_tokens, n_tokens)
    elif mode == "cpu":
        start = time.time()
        run_cpu_stress()
        duration = time.time() - start
    elif mode == "idle":
        start = time.time()
        run_idle()
        duration = time.time() - start
    else:
        raise ValueError("Unknown mode")

    end = time.time()
    duration = end - start

    raw_stats = stop_tegrastats(stats_proc)
    metrics = parse_tegrastats(raw_stats)

    # -------- derived --------
    power_w = metrics["avg_p_in_mw"] / 1000.0
    energy_j = power_w * duration

    row = {
        "mode": mode,
        "p_tokens": p_tokens,
        "n_tokens": n_tokens,
        "cpu_freq": cf,
        "gpu_freq": gf,
        "latency_s": round(duration, 4),
        "energy_j": round(energy_j, 4),
        "tokens_per_sec": (n_tokens / duration) if duration > 0 and n_tokens > 0 else 0
    }

    row.update(metrics)

    save_row(row)


# ===============================
# -------- CSV ------------------
# ===============================
def save_row(row):
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

    file_exists = os.path.isfile(OUTPUT_CSV)

    with open(OUTPUT_CSV, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


# ===============================
# -------- MAIN -----------------
# ===============================
if __name__ == "__main__":

    board.set_mem_default_dvfs()

    workloads = [
        ("idle", 0, 0),
        ("cpu", 0, 0),
        ("llm", 16, 16),
    ]

    try:
        for gf in board.gpu_freqs[::2]:
            for cf in board.cpu_freqs[::2]:

                print(f"\n=== Testing CF={cf}, GF={gf} ===")

                # warm-up
                run_experiment("llm", cf, gf, 32, 32)

                for mode, p, n in workloads:
                    print(f"Running {mode} | P={p}, N={n}")
                    run_experiment(mode, cf, gf, p, n)

    except KeyboardInterrupt:
        print("\nStopped by user.")

    finally:
        print("Done.")
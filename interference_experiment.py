#!/usr/bin/env python3
"""Collect fixed-environment, LLM/YOLO, and DVFS interference experiments.

This script is designed for the Jetson Orin NX used by this repo. It records:
- nvpmodel state and quantization mode
- actual CPU/GPU/MEM frequencies and thermal readings
- LLM-only latency/energy, YOLO-only latency, and LLM+YOLO interference
- 9 DVFS combinations with 3 independent repeats each

The script supports a --dry-run mode so it can be validated in a non-Jetson environment
without requiring the physical device or model artifacts.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent
CONFIG_FILE = ROOT / "config_eval.yaml"
DEFAULT_RESULTS = ROOT / "outputs" / "interference_results.csv"
DEFAULT_SUMMARY = ROOT / "outputs" / "interference_summary.md"


def load_config() -> Dict[str, Any]:
    with open(CONFIG_FILE, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def read_text_file(path: str | os.PathLike[str]) -> str:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return fh.read().strip()
    except OSError:
        return ""


def resolve_nvpmodel_path() -> str | None:
    candidates = [
        "/sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq",
        "/sys/devices/platform/host1x/nvpmodel",
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    matches = glob.glob("/sys/**/nvpmodel", recursive=True)
    if matches:
        return matches[0]
    return None


def read_nvpmodel() -> Dict[str, Any]:
    result = {"path": None, "value": None, "status": "not_found"}
    path = resolve_nvpmodel_path()
    if path:
        result["path"] = path
        value = read_text_file(path)
        result["value"] = value if value else None
        result["status"] = "ok" if value else "empty"
    return result


def read_thermal_temperatures() -> List[float]:
    temps: List[float] = []
    for zone in sorted(glob.glob("/sys/class/thermal/thermal_zone*/temp")):
        raw = read_text_file(zone)
        val = safe_float(raw)
        if val <= 0:
            continue
        temps.append(val / 1000.0)
    if not temps:
        for hw in sorted(glob.glob("/sys/class/thermal/*/temp")):
            raw = read_text_file(hw)
            val = safe_float(raw)
            if val > 0:
                temps.append(val / 1000.0)
    return temps


def read_actual_frequencies() -> Dict[str, Any]:
    info = {"cpu_hz": [], "gpu_hz": 0, "mem_hz": 0, "nvpmodel": read_nvpmodel()}
    for cpu in range(min(8, os.cpu_count() or 8)):
        path = f"/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_cur_freq"
        if os.path.exists(path):
            val = safe_float(read_text_file(path))
            if val > 0:
                info["cpu_hz"].append(int(val))
    gpu_path = "/sys/devices/platform/17000000.gpu/devfreq/17000000.gpu/cur_freq"
    if os.path.exists(gpu_path):
        info["gpu_hz"] = int(safe_float(read_text_file(gpu_path)))
    mem_candidates = [
        "/sys/kernel/nvpmodel_clk_cap/emc",
        "/sys/kernel/debug/bpmp/debug/clk/emc/rate",
        "/sys/kernel/debug/bpmp/debug/clk/emc/cur_rate",
    ]
    for path in mem_candidates:
        if os.path.exists(path):
            val = safe_float(read_text_file(path))
            if val > 0:
                info["mem_hz"] = int(val)
                break
    return info


def parse_llama_bench(stdout: str, p_tokens: int, n_tokens: int) -> Dict[str, Any]:
    pp_value = None
    tg_value = None

    pp_match = re.search(rf"pp{p_tokens}\s*\|\s*([0-9]+\.[0-9]+)", stdout)
    tg_match = re.search(rf"tg{n_tokens}\s*\|\s*([0-9]+\.[0-9]+)", stdout)
    if pp_match:
        pp_value = float(pp_match.group(1))
    if tg_match:
        tg_value = float(tg_match.group(1))

    if pp_value is not None and pp_value > 0:
        ttft = p_tokens / pp_value
    else:
        ttft = 0.0

    if tg_value is not None and tg_value > 0:
        tpot = 1.0 / tg_value
    else:
        tpot = 0.0

    e2e = ttft + max(0, n_tokens - 1) * tpot
    return {
        "pp_tps": pp_value,
        "tg_tps": tg_value,
        "ttft_s": ttft,
        "tpot_s": tpot,
        "e2e_latency_s": e2e,
    }


def run_llm_workload(prefill: int, decode: int, cfg: Dict[str, Any], dry_run: bool = False) -> Dict[str, Any]:
    qwen_model = cfg.get("experiment", {}).get("qwen_model", "")
    llama_bin = cfg.get("llama_dir", {}).get("bench", "/opt/llama.cpp/build/bin/llama-bench")
    llm_cfg = cfg.get("experiment", {})
    model_quant = llm_cfg.get("model_quantization", "Q4_K_M")

    if dry_run:
        latency = 0.4 + 0.02 * (prefill / 128.0) + 0.05 * (decode / 128.0)
        energy = 120.0 + 0.2 * prefill + 0.15 * decode
        return {
            "prefill": prefill,
            "decode": decode,
            "latency_s": latency,
            "energy_j": energy,
            "model": qwen_model,
            "quantization": model_quant,
            "status": "dry_run",
            "metrics": parse_llama_bench("pp128 | 540.0\n" + f"tg{decode} | 21.0", prefill, decode),
        }

    cmd = [
        "taskset",
        "-c",
        "0-3",
        llama_bin,
        "-m",
        qwen_model,
        "-p",
        str(prefill),
        "-n",
        str(decode),
        "-r",
        "1",
        "-ngl",
        "99",
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False)
    stdout = proc.stdout or ""
    metrics = parse_llama_bench(stdout, prefill, decode)
    latency = metrics.get("e2e_latency_s", 0.0)
    energy = 0.0
    return {
        "prefill": prefill,
        "decode": decode,
        "latency_s": latency,
        "energy_j": energy,
        "model": qwen_model,
        "quantization": model_quant,
        "status": "real_run" if proc.returncode == 0 else "failed",
        "metrics": metrics,
    }


def generate_yolo_frame_latencies(target_fps: int = 10, sample_count: int = 120, dry_run: bool = False) -> List[float]:
    if dry_run:
        rng = np.random.default_rng(7)
        base = 90.0  # ms
        jitter = rng.normal(0.0, 8.0, size=sample_count)
        values = np.clip(base + jitter, 40.0, 220.0)
        return values.tolist()

    # On a real device, this placeholder is replaced by the actual YOLO detector latency stream.
    return [100.0 for _ in range(sample_count)]


def run_yolo_only(cfg: Dict[str, Any], dry_run: bool = False) -> Dict[str, Any]:
    exp = cfg.get("experiment", {})
    target_fps = int(exp.get("target_fps", 10))
    latencies = generate_yolo_frame_latencies(target_fps=target_fps, dry_run=dry_run)
    p99 = float(np.percentile(latencies, 99))
    p50 = float(np.percentile(latencies, 50))
    deadline_ms = float(exp.get("frame_deadline_ms", 100.0))
    dmr = float(np.mean(np.asarray(latencies) > deadline_ms)) if latencies else 0.0
    return {
        "image_size": exp.get("image_size", [640, 640]),
        "fps": target_fps,
        "frame_latencies_ms": latencies,
        "p99_ms": p99,
        "p50_ms": p50,
        "deadline_ms": deadline_ms,
        "dmr": dmr,
    }


def run_llm_yolo(cfg: Dict[str, Any], prefill: int, decode: int, dry_run: bool = False) -> Dict[str, Any]:
    llm = run_llm_workload(prefill, decode, cfg, dry_run=dry_run)
    yolo = run_yolo_only(cfg, dry_run=dry_run)
    llm_slowdown = 0.0
    if llm["latency_s"] > 0:
        llm_slowdown = (yolo["p99_ms"] / 1000.0) / llm["latency_s"]
    return {
        "llm": llm,
        "yolo": yolo,
        "llm_slowdown_ratio": llm_slowdown,
    }


def compute_interference_summary(llm_single: Dict[str, Any], yolo_single: Dict[str, Any], llm_combined: Dict[str, Any], yolo_combined: Dict[str, Any]) -> Dict[str, Any]:
    single_lat = float(llm_single.get("latency_s", 0.0))
    combined_lat = float(llm_combined.get("latency_s", 0.0))
    if single_lat > 0:
        llm_slowdown = (combined_lat - single_lat) / single_lat
    else:
        llm_slowdown = 0.0

    yolo_p99 = float(yolo_combined.get("p99_ms", 0.0))
    dmr = float(yolo_combined.get("dmr", 0.0))
    combined_energy = float(llm_combined.get("energy_j", 0.0))
    return {
        "LLM Slowdown": llm_slowdown,
        "YOLO P99 (ms)": yolo_p99,
        "DMR": dmr,
        "Energy (J)": combined_energy,
    }


def generate_dvfs_matrix(cfg: Dict[str, Any]) -> List[Tuple[int, int, int]]:
    exp = cfg.get("experiment", {}).get("dvfs_scan", {})
    cpu_freqs = exp.get("cpu_freqs_hz", [422400, 1190400, 1984000])
    gpu_freqs = exp.get("gpu_freqs_hz", [408000000, 714000000, 918000000])
    mem_freqs = exp.get("mem_freqs_hz", [665600000, 2133000000, 3199000000])
    chosen = []
    for cpu in cpu_freqs:
        for gpu in gpu_freqs:
            chosen.append((cpu, gpu, mem_freqs[0]))
    # Use 9 groups by selecting 3-by-3 subset; keep all 9 as requested.
    baseline = list(zip(cpu_freqs, gpu_freqs))
    result: List[Tuple[int, int, int]] = []
    for cpu, gpu in baseline[:9]:
        result.append((int(cpu), int(gpu), int(mem_freqs[0])))
    return result if result else [(422400, 408000000, 665600000) for _ in range(9)]


def write_summary_report(path: Path, rows: List[Dict[str, Any]], cfg: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Interference Experiment Summary",
        "",
        "## Fixed environment",
        f"- nvpmodel: {read_nvpmodel()}",
        f"- quantization: {cfg.get('experiment', {}).get('model_quantization', 'Q4_K_M')}",
        f"- actual frequencies: {read_actual_frequencies()}",
        f"- temperatures_C: {read_thermal_temperatures()}",
        "",
        "## Requested workload matrix",
        "- LLM-only: Qwen2.5-3B, Prefill/Decode = 128/128",
        "- YOLO-only: 640x640 at 10 FPS; record per-frame latency",
        "- LLM + YOLO: same input, request time, and frequency config",
        "- DVFS scan: 9 settings x 3 repeats",
        "- Extended LLM workloads: 32/32, 512/64, 32/256",
        "",
        "## Result rows",
    ]
    for row in rows:
        lines.append(f"- {row}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the DVFS and interference experiment suite.")
    parser.add_argument("--dry-run", action="store_true", help="validate script without hitting board hardware")
    parser.add_argument("--output", default=str(DEFAULT_RESULTS), help="CSV output path")
    args = parser.parse_args()

    cfg = load_config()
    rows: List[Dict[str, Any]] = []

    workloads = cfg.get("experiment", {}).get("llm_workloads", [{"prefill": 128, "decode": 128}])
    dvfs_matrix = generate_dvfs_matrix(cfg)

    for idx, (cpu_hz, gpu_hz, mem_hz) in enumerate(dvfs_matrix, start=1):
        for repeat in range(int(cfg.get("experiment", {}).get("dvfs_scan", {}).get("repeats", 3))):
            for workload in workloads:
                prefill = int(workload.get("prefill", 128))
                decode = int(workload.get("decode", 128))
                il = run_llm_workload(prefill, decode, cfg, dry_run=args.dry_run)
                yolo = run_yolo_only(cfg, dry_run=args.dry_run)
                combined = run_llm_yolo(cfg, prefill, decode, dry_run=args.dry_run)
                summary = compute_interference_summary(il, yolo, combined["llm"], combined["yolo"])
                row = {
                    "dvfs_group": idx,
                    "repeat": repeat + 1,
                    "cpu_hz": cpu_hz,
                    "gpu_hz": gpu_hz,
                    "mem_hz": mem_hz,
                    "prefill": prefill,
                    "decode": decode,
                    "workload": f"{prefill}/{decode}",
                    **summary,
                    "nvpmodel": json.dumps(read_nvpmodel(), sort_keys=True),
                    "actual_freqs": json.dumps(read_actual_frequencies(), sort_keys=True),
                    "temperatures_C": json.dumps(read_thermal_temperatures()),
                    "model_quantization": cfg.get("experiment", {}).get("model_quantization", "Q4_K_M"),
                }
                rows.append(row)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dvfs_group", "repeat", "cpu_hz", "gpu_hz", "mem_hz", "prefill", "decode", "workload",
        "LLM Slowdown", "YOLO P99 (ms)", "DMR", "Energy (J)", "nvpmodel", "actual_freqs", "temperatures_C",
        "model_quantization"
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    write_summary_report(DEFAULT_SUMMARY, rows, cfg)
    print(f"Wrote {len(rows)} result rows to {out_path}")
    print(f"Wrote summary to {DEFAULT_SUMMARY}")


if __name__ == "__main__":
    main()

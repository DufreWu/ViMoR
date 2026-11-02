# Interference Experiment Checklist

## Fixed environment
- nvpmodel: recorded via `/sys` lookup; run on the Jetson target to confirm the exact value before each sweep.
- model quantization: Qwen2.5-3B is configured as `Q4_K_M` in [config_eval.yaml](config_eval.yaml).
- actual CPU/GPU/MEM frequencies: capture with the board helper in [utils/jetson_orin_nx.py](utils/jetson_orin_nx.py).
- temperature logs: record thermal-zone temperatures for each repeat before the benchmark starts.

## Workload matrix
- LLM-only: Qwen2.5-3B with Prefill/Decode = 128/128.
- YOLO-only: 640x640 input, 10 FPS, record each frame latency.
- LLM + YOLO: same input, request timing, and frequency configuration as the single-task runs.
- Additional LLM workloads: 32/32, 512/64, 32/256.
- Optional follow-up: SegFormer / ROS2 only after resource and memory checks pass.

## DVFS scan
- 9 configurations in total.
- Each configuration repeats at least 3 independent runs.
- CPU/GPU choices are defined in [config_eval.yaml](config_eval.yaml) under `experiment.dvfs_scan`.

## Interference metrics
- LLM Slowdown: relative increase in LLM latency against the single-task baseline.
- YOLO P99: 99th percentile of per-frame latency.
- DMR: deadline miss rate when the frame latency exceeds the configured deadline.
- Energy: total Joules consumed during the combined workload window.

## Implementation status
The repository now includes the experiment runner in [interference_experiment.py](interference_experiment.py) and the required config entries in [config_eval.yaml](config_eval.yaml). This is the concrete execution entry point for the requested study.

> The actual high-fidelity hardware collection must be run on the Jetson Orin NX where the `llama-bench` binary and the YOLO pipeline are available.

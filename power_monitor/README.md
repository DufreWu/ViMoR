# Phase 1 Concurrent Perception + LLM Benchmark

This package provides the first benchmark stage for the ViMoR runtime research project. It measures perception apps (YOLOv8, SegFormer) and LLM inference (Qwen via llama.cpp), standalone and in every combination (W0-W7), on a Jetson Orin NX while CPU and GPU DVFS are swept.

## Directory structure

- `power_monitor/` - benchmark implementation and harness
- `power_monitor/configs/` - experiment and DVFS configuration files
- `power_monitor/controllers/` - DVFS controller and validation helpers
- `power_monitor/monitors/` - power, resource, latency, and multi-app FPS tracking
- `power_monitor/sensors/` - threaded camera source (latest-frame semantics)
- `power_monitor/apps/` - vision apps (YOLO, SegFormer, or any `infer_fn`) and the LLM app
- `power_monitor/experiments/` - workload runner (W0-W7) and DVFS sweep
- `power_monitor/analysis/` - slowdown and summary metrics
- `outputs/power_monitor/` - run results and aggregated summaries

## Python dependencies

Recommended install set:

- Python 3.10+
- PyYAML
- NumPy
- pandas (optional but useful for charts)
- `llama-cpp-python` for real LLM execution when using the existing llama.cpp backend
- `ultralytics` for YOLOv8 execution when available
- `tegrastats` on Jetson Linux for low-overhead resource logging

## Jetson-specific requirements

- Ubuntu/Linux on Jetson Orin NX 16GB
- CPU and GPU sysfs paths must be accessible
- INA power sensor path must be available for input power readout
- Do not modify EMC frequency or the nvpmodel power mode automatically
- Keep a single, stable power mode across the full sweep

## Workloads

The workloads are defined under `workloads:` in `power_monitor/configs/experiments.yaml`:

| ID | Apps                     |
|----|--------------------------|
| W0 | idle baseline: camera streaming, no compute |
| W1 | YOLO                     |
| W2 | SegFormer                |
| W3 | LLM (Qwen)               |
| W4 | YOLO + SegFormer         |
| W5 | YOLO + LLM               |
| W6 | SegFormer + LLM          |
| W7 | YOLO + SegFormer + LLM   |

All apps in a workload run concurrently, each on its own thread, starting from a common
time `t0`. They run for `warmup_seconds + measurement_seconds`, and only the last
`measurement_seconds` are analysed. Models are loaded and initialised before `t0`.

- **Vision apps** (`apps.yolo`, `apps.segformer`): each app releases one frame every
  `1/target_fps` seconds and processes the latest camera frame. They are measured by FPS
  (mean, 1-s-window p5 and min, attainment vs target, jitter, drop rate), plus response
  latency and deadline misses.
- **LLM** (`llm`): requests are released by `request_schedule`. The types are `closed_loop`,
  `periodic`, `poisson` (seeded) and `trace`. Each request has a fixed prompt length and
  at most `decode_tokens` generated tokens (`stop_at_eos: true` stops early at EOS; with
  the random-word prompts it practically always reaches the limit). Every token is timestamped, and the run produces per-request metrics
  (queue, TTFT, TPOT, e2e) and a **100 ms time series** (`llm.sample_interval_ms`).
- **LLM process isolation** (`llm.isolation: process`, the default): the LLM runs in its
  own process, started with `spawn`. This is required for two reasons. First, llama.cpp
  captures decode steps as CUDA graphs, and PyTorch CUDA calls from another thread in the same
  process invalidate the capture (`CUDA error: operation failed due to a previous error during
  capture`). Second, a shared GIL slowed YOLO from 10 to about 6 fps in W5; that slowdown was a
  harness artifact, not real hardware interference. A separate process also matches a real
  deployment, with an LLM server next to the perception nodes. Scripts that call `run_workload`
  directly must therefore put their code under `if __name__ == "__main__":`.
- **Power**: the INA3221 rails (`VDD_IN`, `VDD_CPU_GPU_CV`, `VDD_SOC`) are sampled every
  100 ms. The hwmon directory is detected automatically.

Model paths:

- LLM: `./workloads/llm_models/qwen2.5-3b-instruct-q4_k_m.gguf` (llama.cpp, all layers on GPU)
- YOLO: `./workloads/yolov8n.pt`
- SegFormer: `./workloads/segformer-b0-finetuned-ade-512-512`

To add another LLM (for example Llama3.2-1B), change `llm.model`, `llm.model_path` and
`llm.quantization`. These values are recorded in every run.

## Check the environment

```bash
python -m power_monitor.run --check
```

## Current protocol

The defaults in `experiments.yaml` are:

| setting | value |
|---|---|
| LLM | Qwen2.5-3B Q4_K_M, all layers on GPU, separate process |
| LLM request mode | periodic, every 15 s (first request at t0) |
| prefill / decode | 128 / at most 128 tokens |
| YOLOv8n | 10 FPS, 100 ms deadline |
| SegFormer-B0 | 5 FPS, 200 ms deadline |
| warmup / measurement | 15 s / 180 s (12 LLM requests arrive in the window) |
| repetitions | 3 |
| CPU/GPU | current governor (`dvfs_control: false`), no DVFS writes |
| power monitor | 100 ms |

The full round runs W0-W7 x 3 repetitions = 24 runs, taking about 1.5 h:

```bash
python3 -m power_monitor.run --sweep
```

The CPU/GPU governors and the actual clocks are recorded in each run's `metadata.json`.

## Run one workload

```bash
# at a DVFS point: explicit --cpu-mhz/--gpu-mhz turn DVFS on even with dvfs_control: false.
# Needs root to write sysfs clocks; -E keeps HOME so root's Python sees your ~/.local packages.
# MHz is snapped to the nearest supported frequency.
sudo -E python -m power_monitor.run --workload W5 --cpu-mhz 1190 --gpu-mhz 612
# at the current clocks (no root), short window for a quick test
python3 -m power_monitor.run --workload W7 --no-dvfs --warmup-seconds 3 --measurement-seconds 10
# whole pipeline with a synthetic camera, models and power
python3 -m power_monitor.run --workload W7 --dry-run --measurement-seconds 5
```

## Run the sweep

```bash
# DVFS sweep: needs root (sudo -E, see above).
# --dvfs overrides dvfs_control: false in the config for this invocation.
sudo -E python -m power_monitor.run --sweep --dvfs                    # W0-W7 x 9 DVFS points x 3 reps = 216 runs
sudo -E python -m power_monitor.run --sweep --dvfs --workloads W1     # one workload: 9 x 3 = 27 runs
```

After setting the clocks, each run checks that every CPU core and the GPU actually report
the requested frequency, and fails if they do not. Under sudo, the outputs are handed back
to the invoking user.

Repetition is the outer loop, so slow drift (for example temperature) is spread evenly over
all configurations. The runner waits `cooldown_seconds` between runs.

## Outputs

Each run is written to `outputs/power_monitor/runs/<run_id>/`, where `run_id` is
`W5_cpu1190_gpu612_rep01_20261104-134009`:

| file                    | content |
|-------------------------|---------|
| `metadata.json`         | run_id, workload_config, cpu/gpu_freq_target_mhz (and actual), nvpmodel_mode, llm_model, llm_quantization, yolo_model, segformer_model, input_resolution, target_fps, request_schedule (including the drawn arrival times), measurement_seconds, warmup_seconds, seed, repetition |
| `summary.json`          | power/energy, per-app FPS and latency metrics, LLM metrics |
| `power.csv`             | 100 ms power samples per rail (measurement window) |
| `fps_summary.csv`, `fps_timeseries.csv`, `fps_events.csv` | FPS streams `(camera0, capture)`, `(yolo, output)`, `(segformer, output)`, `(llm, token)` |
| `perception_frames.csv` | one row per released frame per vision app (`in_window` flags the measurement window) |
| `llm_requests.csv`      | one row per LLM request |
| `llm_timeseries.csv`    | one row per 100 ms: phase (idle/prefill/decode), tokens in interval, tokens/s, cumulative tokens, completed and queued requests |

`outputs/power_monitor/runs_index.csv` gets one row per run, containing the metadata
fields followed by all summary metrics. This is the table to analyse.

All timestamps use the same `time.monotonic()` clock, so `power.csv`, `llm_timeseries.csv`
and `fps_timeseries.csv` can be joined on `monotonic`.

## Camera and FPS monitoring

Real (non `--dry-run`) YOLO runs read frames from the camera configured under `camera:` in
`experiments.yaml`. The default is a ZED 2i on `/dev/video0` at 1344x376 @ 30 fps, cropped to the left eye.
If the camera or model cannot be opened, the run falls back to synthetic frames. In that case
`yolo_source` is set to `synthetic_fallback` and a warning is added to `metadata.json`.

Frame rates are tracked by `FpsMonitor` (`power_monitor/monitors/fps_monitor.py`). Each rate is a
stream named by `(app, stage)`, so any number of apps can report to one monitor at the same time
(the monitor is thread-safe):

| stream                 | meaning                                  |
|------------------------|------------------------------------------|
| `(camera0, capture)`   | frames delivered by the camera           |
| `(yolo, output)`       | frames processed by the YOLO app         |
| `(<your_app>, <stage>)`| anything else you register               |

```python
from power_monitor.monitors import FpsMonitor
from power_monitor.sensors import CameraSource
from power_monitor.apps import PeriodicVisionApp, build_yolo_infer_fn

fps = FpsMonitor(window_s=1.0, step_s=0.5)
with CameraSource(0, stereo_crop="left", fps_monitor=fps) as cam:
    yolo = PeriodicVisionApp("yolo", build_yolo_infer_fn("workloads/yolov8n.pt"), cam, 10, 100, fps)
    seg  = PeriodicVisionApp("segformer", my_seg_fn, cam, 5, 200, fps)   # any infer_fn(image)
    yolo.warmup(2); seg.warmup(2)
    fps.start()
    threads = [yolo.start_thread(60), seg.start_thread(60)]
    print(fps.current_all())          # live windowed FPS, e.g. for a DVFS controller
    for t in threads: t.join()
    fps.stop()
fps.write_csv("outputs/power_monitor/my_run")
```

If an app is not driven by the camera (for example LLM tokens/s), register a stream and call
`tick()` once per event: `s = fps.register("llm", "token"); s.tick()`. Call `drop()` for each
scheduled frame that was skipped.

Each run directory also contains:

- `fps_events.csv`: one row per tick (`monotonic, app, stage, frame_id`)
- `fps_timeseries.csv`: sliding-window FPS per stream every `step_s`
- `fps_summary.csv`: per stream `mean_fps`, `window_fps_min/p5/max`, `interval_p50/p95/p99/max_ms`,
  `jitter_ms`, `fps_attainment` (mean / target), `drop_rate`
- `summary.json`: the same metrics flattened as `fps_<app>_<stage>_<metric>`

## Hardware restoration

The DVFS controller saves the original CPU/GPU governor and frequency settings before a run and restores them in a `finally` block. This ensures settings are returned to their previous state even when a run fails midway.

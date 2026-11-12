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
  200 ms. The hwmon directory is detected automatically. Total power is `VDD_IN` (module input).
- **Hardware telemetry** (`monitors/telemetry.py`, 200 ms): tegrastats (CPU/GPU utilisation, RAM,
  EMC utilisation and frequency - EMC only when run as root) plus sysfs (actual CPU clock per
  cluster via `cpuinfo_cur_freq` when readable, GPU devfreq `cur_freq`, thermal zones, and
  throttling indicators: thermal cooling-device states, `*-throttle-alert` trips, soctherm
  over-current event counters, INA3221 alarms).

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
| warmup / measurement | 30 s / 180 s (12 LLM requests arrive in the window) |
| repetitions | 5 |
| CPU/GPU | current governor (`dvfs_control: false`); `--dvfs` sweeps `dvfs.yaml` (3 x 3 static points) |
| power + telemetry sampling | 200 ms |
| nvpmodel / EMC | never changed (recorded at the start and end of every run) |
| thermal gate | wait until Tj <= 55 °C (max 300 s) before each timed region |
| FPS satisfaction window | 1 s |

Without DVFS, the full round runs W0-W7 x 5 repetitions = 40 runs:

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
sudo -E python -m power_monitor.run --sweep --dvfs                    # W0-W7 x 9 DVFS points x 5 reps = 360 runs (~24-26 h)
sudo -E python -m power_monitor.run --sweep --dvfs --workloads W1,W3  # 2 x 9 x 5 = 90 runs
# full grid: every combination of jetson.cpu_available_frequencies_hz x gpu_available_frequencies_hz
# in dvfs.yaml (12 x 7 = 84 points, a superset of the 3x3). Long: 84 x 8 x 5 = 3360 runs (~230 h)
sudo -E python -m power_monitor.run --sweep --dvfs --grid full --workloads W1,W3 --repetitions 3   # 84 x 2 x 3 = 504 runs (~35 h)
# stock dynamic DVFS (CPU schedutil + GPU nvhost_podgov, full frequency range) as a baseline:
sudo -E python -m power_monitor.run --sweep --dvfs --governor default # 9 fixed points + default, shuffled together: 8 x 10 x 5 = 400 runs
sudo -E python -m power_monitor.run --sweep --governor default        # default governor only: 8 x 5 = 40 runs
sudo -E python -m power_monitor.run --workload W5 --governor default  # one run
# interrupted? continue from the plan (add --retry-failed to re-run failed entries)
sudo -E python -m power_monitor.run --resume sweep_20261109-153000
# per-(workload, CPU, GPU) statistics, report and heatmaps
python3 -m power_monitor.run --aggregate [--sweep-id sweep_...] [--include-flagged]
```

Each run follows this protocol: validate the DVFS point (it must be a supported OPP within
1 MHz) -> set the clocks -> verify every CPU cluster and the GPU report them -> thermal gate ->
load and warm up the models -> record the starting temperatures -> start power and telemetry ->
common `t0` -> `warmup_seconds` workload warmup -> `measurement_seconds` window -> stop ->
record the final temperatures and nvpmodel -> completeness and quality checks -> save.

The sweep plan (`outputs/power_monitor/sweeps/<sweep_id>/plan.csv`, with a copy of the config)
uses repetition as the outer loop. Within each (repetition, workload) the DVFS points are
**shuffled** with a seed derived from `(seed, workload, repetition)`. The plan status is
updated after every run. Invalid DVFS points are kept in the plan as `invalid_config`.
Under sudo, the outputs are handed back to the invoking user.

### Default-governor mode (`--governor default`)

The governors come from `dvfs.default_governor` in `dvfs.yaml`. Each run sets them and opens
the CPU and GPU frequency ranges fully. It is recorded as `cpu/gpu_freq_target_mhz = default` and
`governor_mode = default`. Fixed-point runs have `governor_mode = fixed`, and `--no-dvfs` runs
have `current`. No frequency lock is checked, because the clocks are meant to move. Instead, the
run records `cpu/gpu_freq_actual_mean/p5/p95/min/max` and `cpu/gpu_freq_residency` (the share of
time spent at each MHz). The run is flagged `governor_changed` (hard) if the governors are
different at the end of the run. In `--aggregate`, the default row comes after the fixed points
in each table, along with its actual mean clocks, and it is left out of the CPU x GPU heatmaps.

Note: the governors the board started with are restored after every run. On this board that is
`userspace` pinned at 1036.8/612 MHz, which is left over from earlier work, so `--no-dvfs`
currently measures that pinned point and not the stock behaviour. Use `--governor default` for
the stock behaviour.

### Failed and flagged runs

Nothing is silently discarded:

- A run that raises is written anyway (`status: failed`, `failed_phase`, `error`, `error.txt`,
  and any partial power or telemetry data) and appended to `runs_index.csv`.
- A run that completes gets `valid` and `quality_flags`. These **hard** flags set `valid=false`:
  `freq_lock_failed` (< 99 % of telemetry samples at the requested CPU and GPU clocks),
  `thermal_throttled`, `power_throttled`, `incomplete_power`, `incomplete_telemetry`,
  `incomplete_app_data`, `llm_no_completed_requests`. Soft flags (`incomplete_tegrastats`,
  `thermal_gate_timeout`, `llm_request_failures`) are recorded but keep the run valid.
- `--aggregate` excludes `valid=false` runs by default, and reports total, failed and flagged
  counts and flag tallies for every configuration.

### Metric definitions

| metric | definition |
|---|---|
| `energy_j` | trapezoidal integral of VDD_IN over exactly the measurement window, using actual sample timestamps (linear interpolation at the window edges) |
| `mean_power_w` / `peak_power_w` | `energy_j / window`; maximum 200 ms sample |
| `cpu/gpu_freq_requested_mhz`, `cpu/gpu_freq_actual_*` | OPP written to sysfs; read back every 200 ms (`*_lock_rate` = share of samples at the target) |
| `<app>_fps_q` | Q_i^FPS = processed frames / window / target FPS |
| `<app>_fps_satisfaction_rate` | share of consecutive 1 s windows with FPS_i >= target; frames are binned by their scheduled release, so completion jitter cannot shift a frame into the next window |
| `<app>_inference_p50/p95/p99_ms` | model latency (`inference_end - inference_start`); `response_*` adds the wait for a frame |
| `<app>_deadline_miss_rate` | releases that finished after `deadline_ms` or were skipped, divided by releases |
| `llm_ttft_ms_*` | first streamed token - service start (queueing is `llm_queue_ms_*`) |
| `llm_tpot_ms_*` | (last - first token) / (n - 1): excludes the first token |
| `llm_decode_tps_*`, `llm_e2e_s_*` | (n - 1) / (last - first token); last token - arrival |
| `llm_failure_rate` | failed / (arrived - truncated); requests cut off by the end of the run are `truncated`, not failures |

All LLM timings are taken from per-token streaming timestamps (`llm_timing_source`), not
approximated. Energy is always total system energy. `--aggregate` gives J/frame and J/token
only for single-app workloads (W1, W2, W3) and never splits shared GPU energy in concurrent
workloads.

**Open-loop LLM overload:** requests arrive every 15 s whether or not the LLM is busy. If a
request takes longer than that (for example W7 at GPU 612 MHz: TPOT is about 160 ms, so a
128-token request takes about 21 s), the queue grows. `llm_e2e_s_*` and `llm_queue_ms_*` then
show this, and the last requests in the run are `truncated`. Use `request_schedule.type:
closed_loop` to measure service capacity without queueing.

## Outputs

Each run is written to `outputs/power_monitor/runs/<run_id>/`, where `run_id` is
`W5_cpu1190_gpu612_rep01_20261104-134009`:

| file                    | content |
|-------------------------|---------|
| `metadata.json`         | run_id, sweep_id, run_order, status, valid, quality_flags, pre/post-run hardware snapshot, nvpmodel at start and end, workload_config, cpu/gpu_freq_target_mhz (and actual), nvpmodel_mode, llm_model, llm_quantization, yolo_model, segformer_model, input_resolution, target_fps, request_schedule (including the drawn arrival times), measurement_seconds, warmup_seconds, seed, repetition |
| `summary.json`          | power/energy, per-app FPS and latency metrics, LLM metrics |
| `power.csv`             | 200 ms power samples per rail (measurement window) |
| `telemetry.csv`         | 200 ms sysfs samples: actual clocks, GPU load, temperatures, throttle indicators (`in_window` flag) |
| `tegrastats.csv`        | parsed tegrastats lines: per-core CPU util/freq, GR3D util, RAM, EMC, temps (`in_window` flag) |
| `error.txt`             | traceback (failed runs only) |
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

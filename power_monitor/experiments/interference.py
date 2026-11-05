"""Workload-combination benchmark (W0-W7) under CPU/GPU DVFS.

A workload is a set of apps drawn from {yolo, segformer, llm}. All apps of a workload
run concurrently on their own threads from a common start time ``t0`` for
``warmup_seconds + measurement_seconds``; only the measurement window
``[t0 + warmup, t0 + warmup + measurement]`` is analysed. Vision apps are measured by
FPS (plus latency / deadline misses), the LLM by request metrics and a 100 ms
time series, and module power is sampled every 100 ms.
"""

from __future__ import annotations

import csv
import gc
import json
import os
import random
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from power_monitor.controllers.dvfs_controller import DvfsController
from power_monitor.monitors.fps_monitor import FpsMonitor
from power_monitor.monitors.power_monitor import JetsonPowerMonitor, SyntheticPowerMonitor

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "experiments.yaml"
DEFAULT_DVFS_PATH = Path(__file__).resolve().parents[1] / "configs" / "dvfs.yaml"

VISION_APPS = ("yolo", "segformer")
ALL_APPS = VISION_APPS + ("llm",)

# Columns of runs_index.csv that describe the run (results follow after these).
METADATA_FIELDS = [
    "run_id",
    "workload_config",
    "workload_apps",
    "cpu_freq_target_mhz",
    "gpu_freq_target_mhz",
    "cpu_freq_actual_mhz",
    "gpu_freq_actual_mhz",
    "nvpmodel_mode",
    "llm_model",
    "llm_quantization",
    "yolo_model",
    "segformer_model",
    "input_resolution",
    "target_fps",
    "request_schedule",
    "measurement_seconds",
    "warmup_seconds",
    "seed",
    "repetition",
    "start_time",
    "dry_run",
    "status",
]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
def load_experiment_config(path: str | os.PathLike[str] | None = None) -> Dict[str, Any]:
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not config.get("dvfs") and DEFAULT_DVFS_PATH.exists():
        with open(DEFAULT_DVFS_PATH, "r", encoding="utf-8") as dvfs_handle:
            config["dvfs"] = (yaml.safe_load(dvfs_handle) or {}).get("dvfs", {})
    return config


def validate_experiment_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    errors: List[str] = []
    warnings: List[str] = []
    workloads = cfg.get("workloads") or {}
    if not workloads:
        errors.append("No workloads configured")
    for wid, apps in workloads.items():
        for app in apps or []:
            if app not in ALL_APPS:
                errors.append(f"workloads.{wid}: unknown app {app!r} (expected one of {ALL_APPS})")
    used = {app for apps in workloads.values() for app in (apps or [])}
    apps_cfg = cfg.get("apps", {})
    for app in used & set(VISION_APPS):
        app_cfg = apps_cfg.get(app) or {}
        if not app_cfg.get("model_path"):
            errors.append(f"apps.{app}.model_path missing")
        if not isinstance(app_cfg.get("target_fps"), (int, float)) or app_cfg.get("target_fps", 0) <= 0:
            errors.append(f"apps.{app}.target_fps must be a positive number")
    if "llm" in used:
        llm_cfg = cfg.get("llm") or {}
        if not llm_cfg.get("model_path"):
            errors.append("llm.model_path missing")
        for key in ("prompt_tokens", "decode_tokens"):
            if not isinstance(llm_cfg.get(key), int):
                errors.append(f"llm.{key} must be an integer")
        try:
            from power_monitor.apps.llm_app import RequestSchedule

            RequestSchedule.from_config(llm_cfg.get("request_schedule"))
        except (ValueError, TypeError) as exc:
            errors.append(str(exc))
    dvfs_cfg = cfg.get("dvfs", {})
    if not dvfs_cfg.get("cpu_freq_mhz") or not dvfs_cfg.get("gpu_freq_mhz"):
        warnings.append("dvfs.cpu_freq_mhz / gpu_freq_mhz not configured; using default candidate list")
    return {"valid": not errors, "errors": errors, "warnings": warnings}


def build_dvfs_matrix(cfg: Dict[str, Any]) -> List[Tuple[int, int]]:
    """``[(cpu_mhz, gpu_mhz)]`` from the config."""
    dvfs_cfg = cfg.get("dvfs", {})
    cpu_values = [int(v) for v in dvfs_cfg.get("cpu_freq_mhz", [576, 1190, 1984])]
    gpu_values = [int(v) for v in dvfs_cfg.get("gpu_freq_mhz", [306, 612, 918])]
    return [(c, g) for c in cpu_values for g in gpu_values]


def _nearest(target: float, available: List[int]) -> int:
    return min(available, key=lambda v: abs(v - target)) if available else int(target)


def resolve_frequencies(controller: DvfsController, cpu_mhz: float, gpu_mhz: float) -> Tuple[int, int]:
    """MHz -> sysfs units: CPU ``scaling_*`` is kHz, GPU devfreq is Hz. Snaps to the nearest OPP."""
    return _nearest(cpu_mhz * 1000.0, controller.available_cpu_hz), _nearest(gpu_mhz * 1e6, controller.available_gpu_hz)


# Python modules each app needs for a real (non dry-run) run.
APP_REQUIREMENTS = {
    "yolo": ["torch", "ultralytics", "cv2"],
    "segformer": ["torch", "transformers", "cv2"],
    "llm": ["llama_cpp"],
}


def check_dependencies(cfg: Dict[str, Any], workload_ids: Optional[List[str]] = None, dry_run: bool = False) -> List[str]:
    """Errors for missing modules / model files needed by the selected workloads (empty = OK)."""
    import importlib.util

    if dry_run:
        return []
    workloads = cfg.get("workloads", {})
    apps = {a for w in (workload_ids or list(workloads)) for a in (workloads.get(w) or [])}
    needs_camera = bool(apps & set(VISION_APPS)) or cfg.get("camera", {}).get("always_on", True)
    modules = sorted({m for a in apps for m in APP_REQUIREMENTS.get(a, [])} | ({"cv2"} if needs_camera else set()))
    errors = [f"Python module '{m}' not importable (needed by {', '.join(sorted(a for a in apps if m in APP_REQUIREMENTS.get(a, [])) or ['camera'])})"
              for m in modules if importlib.util.find_spec(m) is None]
    if errors and os.geteuid() == 0 and os.environ.get("SUDO_USER"):
        errors.append(f"Running under sudo: root's Python does not see {os.environ['SUDO_USER']}'s user site-packages (~/.local). "
                      "Run without sudo when DVFS is off, or use sudo -E (keeps HOME, so ~/.local is visible): "
                      "sudo -E python -m power_monitor.run ...")
    for app in apps & set(VISION_APPS):
        path = cfg.get("apps", {}).get(app, {}).get("model_path")
        if path and not Path(path).exists():
            errors.append(f"apps.{app}.model_path not found: {path}")
    if "llm" in apps:
        path = cfg.get("llm", {}).get("model_path")
        if path and not Path(path).exists():
            errors.append(f"llm.model_path not found: {path}")
    return errors


def read_nvpmodel_mode() -> Optional[str]:
    try:
        out = subprocess.run(["nvpmodel", "-q"], capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    lines = [line.strip() for line in out.splitlines() if line.strip()]
    name = next((line.split(":", 1)[1].strip() for line in lines if line.startswith("NV Power Mode")), None)
    mode_id = next((line for line in lines if line.isdigit()), None)
    if name and mode_id is not None:
        return f"{name} ({mode_id})"
    return name


def _read_governors() -> Dict[str, Any]:
    def read(path: Path) -> Optional[str]:
        try:
            return path.read_text(encoding="utf-8").strip()
        except OSError:
            return None

    cpu = sorted({g for g in (read(p) for p in Path("/sys/devices/system/cpu").glob("cpu[0-9]*/cpufreq/scaling_governor")) if g})
    return {"cpu_governor": cpu, "gpu_governor": read(Path(DvfsController().gpu_path) / "governor")}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np  # type: ignore

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch  # type: ignore

        torch.manual_seed(seed)
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# App construction
# ---------------------------------------------------------------------------
def _build_vision_app(name: str, cfg: Dict[str, Any], camera: Any, fps_monitor: FpsMonitor, dry_run: bool) -> Any:
    from power_monitor.apps.vision_app import PeriodicVisionApp, SegformerInfer, SyntheticInfer, YoloInfer

    app_cfg = cfg["apps"][name]
    if dry_run:
        size = int(app_cfg.get("image_size", 640))
        infer = SyntheticInfer(app_cfg.get("model", name), float(app_cfg.get("dry_run_service_ms", 20.0)), [size, size])
    elif name == "yolo":
        infer = YoloInfer(app_cfg["model_path"], int(app_cfg.get("image_size", 640)), app_cfg.get("device", 0))
    elif name == "segformer":
        infer = SegformerInfer(app_cfg["model_path"], int(app_cfg.get("image_size", 512)), app_cfg.get("device", "cuda"), bool(app_cfg.get("fp16", False)))
    else:
        raise ValueError(f"Unknown vision app {name}")
    return PeriodicVisionApp(name, infer, camera, float(app_cfg["target_fps"]), float(app_cfg.get("deadline_ms", 1000.0 / app_cfg["target_fps"])), fps_monitor)


def _build_llm_app(cfg: Dict[str, Any], seed: int, fps_monitor: FpsMonitor, dry_run: bool) -> Any:
    from power_monitor.apps.llm_app import LlamaCppBackend, LlmApp, LlmProcessApp, RequestSchedule, SyntheticLlmBackend

    llm_cfg = cfg["llm"]
    backend_kwargs = None if dry_run else {
        "model_path": llm_cfg["model_path"],
        "n_gpu_layers": int(llm_cfg.get("n_gpu_layers", -1)),
        "n_ctx": int(llm_cfg.get("n_ctx", 2048)),
        "n_threads": llm_cfg.get("n_threads"),
        "seed": seed,
        "verbose": bool(llm_cfg.get("verbose", False)),
        "stop_at_eos": bool(llm_cfg.get("stop_at_eos", False)),
    }
    common = {
        "prompt_tokens": int(llm_cfg.get("prompt_tokens", 128)),
        "decode_tokens": int(llm_cfg.get("decode_tokens", 128)),
        "schedule": RequestSchedule.from_config(llm_cfg.get("request_schedule")),
        "seed": seed,
        "fps_monitor": fps_monitor,
    }
    if llm_cfg.get("isolation", "process") == "process":
        return LlmProcessApp(backend_kwargs, **common)
    backend = SyntheticLlmBackend() if backend_kwargs is None else LlamaCppBackend(**backend_kwargs)
    return LlmApp(backend, **common)


# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------
def run_workload(
    workload_id: str,
    cpu_mhz: Optional[float],
    gpu_mhz: Optional[float],
    repetition: int = 1,
    config: Optional[Dict[str, Any]] = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Run one workload at one DVFS point. ``cpu_mhz``/``gpu_mhz`` = None keeps current clocks."""
    from power_monitor.apps.llm_app import guess_quantization
    from power_monitor.sensors.camera import CameraSource, SyntheticCamera

    cfg = config or load_experiment_config()
    workloads = cfg.get("workloads", {})
    if workload_id not in workloads:
        raise KeyError(f"Unknown workload {workload_id!r}; configured: {sorted(workloads)}")
    apps_in_workload: List[str] = list(workloads[workload_id] or [])
    seed = int(cfg.get("seed", 0))
    warmup_s = float(cfg.get("warmup_seconds", 10))
    measure_s = float(cfg.get("measurement_seconds", 60))
    llm_cfg = cfg.get("llm", {})
    apps_cfg = cfg.get("apps", {})
    seed_everything(seed)

    start_time = time.strftime("%Y-%m-%dT%H:%M:%S")
    tag = lambda v: "cur" if v is None else str(int(v))  # noqa: E731
    run_id = f"{workload_id}_cpu{tag(cpu_mhz)}_gpu{tag(gpu_mhz)}_rep{repetition:02d}_{time.strftime('%Y%m%d-%H%M%S')}"
    run_dir = Path(cfg.get("output_root", "outputs/power_monitor")) / "runs" / run_id  # created only on success
    warnings: List[str] = []

    controller = DvfsController()
    controller.save_original_state()
    camera = None
    vision_apps: Dict[str, Any] = {}
    llm_app = None
    power = None
    try:
        # ----- DVFS ----------------------------------------------------------
        if cpu_mhz is not None and gpu_mhz is not None and not dry_run:
            cpu_khz, gpu_hz = resolve_frequencies(controller, cpu_mhz, gpu_mhz)
            controller.validate_frequency_pair(cpu_khz, gpu_hz)
            try:
                controller.set_cpu_frequency(cpu_khz)
                controller.set_gpu_frequency(gpu_hz)
            except PermissionError as exc:
                raise PermissionError("Setting DVFS needs root: run with sudo, or use --no-dvfs") from exc
            time.sleep(0.5)
            pinned = controller.read_actual_frequencies()
            if set(pinned.get("cpu_hz") or []) != {cpu_khz} or pinned.get("gpu_hz") != gpu_hz:
                raise RuntimeError(f"DVFS not applied: requested cpu={cpu_khz} kHz gpu={gpu_hz} Hz, "
                                   f"actual cpu={sorted(set(pinned.get('cpu_hz') or []))} gpu={pinned.get('gpu_hz')}")

        # ----- build apps (model loading is outside the timed region) -------
        fps_cfg = cfg.get("fps_monitor", {})
        fps_monitor = FpsMonitor(window_s=float(fps_cfg.get("window_s", 1.0)), step_s=float(fps_cfg.get("step_s", 0.5)))
        camera_cfg = cfg.get("camera", {})
        needs_camera = any(a in VISION_APPS for a in apps_in_workload) or camera_cfg.get("always_on", True)
        if needs_camera:
            camera_cls = SyntheticCamera if (dry_run or not camera_cfg.get("enabled", True)) else CameraSource
            camera = camera_cls.from_config(camera_cfg, fps_monitor=fps_monitor).open()
        for name in apps_in_workload:
            if name in VISION_APPS:
                vision_apps[name] = _build_vision_app(name, cfg, camera, fps_monitor, dry_run)
            elif name == "llm":
                llm_app = _build_llm_app(cfg, seed, fps_monitor, dry_run)

        # Initialisation warmup (CUDA context, cuDNN autotune, llama.cpp graph) per app.
        if llm_app is not None:
            llm_app.warmup(1)  # process mode: spawns the LLM process and waits until it is ready
        for app in vision_apps.values():
            app.warmup(0.0, min_iters=3)

        # ----- timed region ---------------------------------------------------
        power_cfg = cfg.get("power", {})
        power = SyntheticPowerMonitor() if dry_run else JetsonPowerMonitor(sensor_dir=power_cfg.get("sensor_dir", "auto"))
        power.start(float(power_cfg.get("sample_interval_ms", 100)) / 1000.0)

        total_s = warmup_s + measure_s
        t0 = time.monotonic() + 0.2
        win_start, win_end = t0 + warmup_s, t0 + total_s
        threads = [app.start_thread(total_s, t0) for app in vision_apps.values()]
        if llm_app is not None:
            threads.append(llm_app.start_thread(total_s, t0))
        if not threads:  # W0: idle baseline
            time.sleep(max(0.0, win_end - time.monotonic()))
        for thread in threads:
            thread.join()
        for app in list(vision_apps.values()) + ([llm_app] if llm_app else []):
            if app.error is not None:
                raise app.error
        actual = controller.read_actual_frequencies()
        power.stop()
        fps_monitor.start(win_start)
        fps_monitor.stop(win_end)

        # ----- results --------------------------------------------------------
        power_window = power.window(win_start, win_end)
        summary: Dict[str, Any] = {
            "mean_power_w": (sum(s["power_w"] for s in power_window) / len(power_window)) if power_window else None,
            "peak_power_w": max((s["power_w"] for s in power_window), default=None),
            "energy_j": power.integrate_energy(power_window),
            "power_samples": len(power_window),
        }
        for rail in [name for _, name in power.rails()][1:]:
            values = [s[f"{rail}_w"] for s in power_window if f"{rail}_w" in s]
            summary[f"mean_{rail}_w"] = (sum(values) / len(values)) if values else None
        for name, app in vision_apps.items():
            stream = fps_monitor.stream(name, "output").summary(win_start, win_end, fps_monitor.window_s)
            summary.update(
                {
                    f"{name}_fps_mean": stream["mean_fps"],
                    f"{name}_fps_p5": stream["window_fps_p5"],
                    f"{name}_fps_min": stream["window_fps_min"],
                    f"{name}_fps_attainment": stream["fps_attainment"],
                    f"{name}_frame_interval_p99_ms": stream["interval_p99_ms"],
                    f"{name}_jitter_ms": stream["jitter_ms"],
                    f"{name}_drop_rate": stream["drop_rate"],
                }
            )
            summary.update(app.summary(win_start, win_end))
        if camera is not None:
            summary["camera_fps_mean"] = fps_monitor.stream(camera.name, "capture").summary(win_start, win_end)["mean_fps"]
        if llm_app is not None:
            summary.update(llm_app.summary(win_start, win_end))

        # ----- metadata -------------------------------------------------------
        input_resolution: Dict[str, Any] = {}
        if camera is not None and camera.latest() is not None:
            input_resolution["camera"] = list(camera.latest().image.shape[:2])
        for name, app in vision_apps.items():
            input_resolution[name] = getattr(app.infer_fn, "input_resolution", None)
        if llm_app is not None:
            input_resolution["llm"] = {"prompt_tokens": llm_app.prompt_tokens, "decode_tokens": llm_app.decode_tokens}
        cpu_actual = actual.get("cpu_hz") or []
        has = apps_in_workload.__contains__
        metadata: Dict[str, Any] = {
            "run_id": run_id,
            "workload_config": workload_id,
            "workload_apps": "+".join(apps_in_workload) if apps_in_workload else "idle",
            "cpu_freq_target_mhz": cpu_mhz,
            "gpu_freq_target_mhz": gpu_mhz,
            "cpu_freq_actual_mhz": sorted({round(v / 1000.0, 1) for v in cpu_actual}),
            "gpu_freq_actual_mhz": round(actual["gpu_hz"] / 1e6, 1) if actual.get("gpu_hz") else None,
            "nvpmodel_mode": read_nvpmodel_mode(),
            "llm_model": llm_cfg.get("model") if has("llm") else None,
            "llm_quantization": (llm_cfg.get("quantization") or guess_quantization(llm_cfg.get("model_path", ""))) if has("llm") else None,
            "yolo_model": apps_cfg.get("yolo", {}).get("model") if has("yolo") else None,
            "segformer_model": apps_cfg.get("segformer", {}).get("model") if has("segformer") else None,
            "input_resolution": input_resolution,
            "target_fps": {name: app.target_fps for name, app in vision_apps.items()},
            "request_schedule": llm_app.schedule.describe(total_s, seed) if llm_app else None,
            "measurement_seconds": measure_s,
            "warmup_seconds": warmup_s,
            "seed": seed,
            "repetition": repetition,
            "start_time": start_time,
            "dry_run": dry_run,
            "status": "completed",
            # extras (not in runs_index.csv)
            "camera": camera.properties() if camera is not None else None,
            "llm_config": {k: llm_cfg.get(k, "process" if k == "isolation" else None) for k in ("model_path", "n_gpu_layers", "n_ctx", "prompt_tokens", "decode_tokens", "stop_at_eos", "isolation")} if has("llm") else None,
            "dvfs_control": cpu_mhz is not None and gpu_mhz is not None and not dry_run,
            **_read_governors(),
            "power_sample_interval_ms": float(cfg.get("power", {}).get("sample_interval_ms", 100)),
            "measurement_window_monotonic": [win_start, win_end],
            "warnings": warnings,
        }

        _write_run_outputs(run_dir, metadata, summary, vision_apps, llm_app, fps_monitor, power, win_start, win_end, float(llm_cfg.get("sample_interval_ms", 100)) / 1000.0)
        output_root = Path(cfg.get("output_root", "outputs/power_monitor"))
        _append_runs_index(output_root / "runs_index.csv", metadata, summary)
        _chown_to_sudo_user(output_root, output_root / "runs", run_dir, output_root / "runs_index.csv")
        return {**{k: metadata[k] for k in METADATA_FIELDS}, **summary}
    finally:
        if power is not None:
            power.stop()
        for app in vision_apps.values():
            app.stop()
        if llm_app is not None:
            llm_app.stop()
        if camera is not None:
            camera.close()
        controller.restore_original_state()
        vision_apps.clear()
        llm_app = None
        _free_gpu_memory()


def _free_gpu_memory() -> None:
    gc.collect()
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def _write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: Optional[List[str]] = None) -> None:
    if fieldnames is None:
        fieldnames = list(dict.fromkeys(k for row in rows for k in row))
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_run_outputs(run_dir: Path, metadata: Dict[str, Any], summary: Dict[str, Any], vision_apps: Dict[str, Any], llm_app: Any,
                       fps_monitor: FpsMonitor, power: JetsonPowerMonitor, win_start: float, win_end: float, llm_interval_s: float) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "metadata.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    with open(run_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    power.write_csv(run_dir / "power.csv", power.window(win_start, win_end))
    fps_monitor.write_csv(run_dir)

    frames = []
    for app in vision_apps.values():
        for frame in app.frames:
            frames.append({**frame, "in_window": win_start <= frame["scheduled_release_ts"] < win_end})
    if frames:
        _write_csv(run_dir / "perception_frames.csv", frames, [
            "app", "frame_id", "in_window", "skipped", "camera_frame_id", "stale_frame", "capture_ts",
            "scheduled_release_ts", "actual_submit_ts", "inference_start_ts", "inference_end_ts",
            "response_ms", "e2e_ms", "deadline_ms", "deadline_met",
        ])

    if llm_app is not None:
        requests = [{**{k: v for k, v in r.items() if k != "token_ts"}, "in_window": win_start <= r["arrival_ts"] < win_end} for r in llm_app.requests]
        if requests:
            _write_csv(run_dir / "llm_requests.csv", requests)
        _write_csv(run_dir / "llm_timeseries.csv", llm_app.timeseries(win_start, win_end, llm_interval_s), [
            "monotonic", "t_rel_s", "phase", "active_request", "tokens_in_interval", "tokens_per_s",
            "cumulative_tokens", "completed_requests", "queued_requests",
        ])


def _flatten(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value, separators=(",", ":"))
    return value


def _chown_to_sudo_user(*paths: Path) -> None:
    """Under sudo, hand outputs back to the invoking user so later non-root runs can append."""
    uid, gid = os.environ.get("SUDO_UID"), os.environ.get("SUDO_GID")
    if os.geteuid() != 0 or not uid or not gid:
        return
    for path in paths:
        if not path.exists():
            continue
        targets = [path] + (list(path.rglob("*")) if path.is_dir() and path.name != "runs" else [])
        for target in targets:
            try:
                os.chown(target, int(uid), int(gid))
            except OSError:
                pass


def _append_runs_index(path: Path, metadata: Dict[str, Any], summary: Dict[str, Any]) -> None:
    """One row per run. Columns are the union over runs (different workloads have different metrics)."""
    row = {k: _flatten(metadata.get(k)) for k in METADATA_FIELDS}
    row.update({k: _flatten(v) for k, v in summary.items()})
    rows: List[Dict[str, Any]] = []
    if path.exists():
        with open(path, newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    rows.append(row)
    fieldnames = list(dict.fromkeys(METADATA_FIELDS + [k for r in rows for k in r]))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".csv.tmp")
    _write_csv(tmp, rows, fieldnames)
    tmp.replace(path)


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------
def run_sweep(cfg: Optional[Dict[str, Any]] = None, workloads: Optional[List[str]] = None, dry_run: bool = False,
              no_dvfs: bool = False) -> List[Dict[str, Any]]:
    cfg = cfg or load_experiment_config()
    workload_ids = workloads or list(cfg.get("workloads", {}))
    dvfs_points: List[Tuple[Optional[int], Optional[int]]] = [(None, None)] if no_dvfs else list(build_dvfs_matrix(cfg))
    repetitions = int(cfg.get("repetitions", 3))
    cooldown_s = float(cfg.get("cooldown_seconds", 0))
    total = repetitions * len(dvfs_points) * len(workload_ids)
    results: List[Dict[str, Any]] = []
    index = 0
    # Repetition is the outer loop so slow drift (thermal, background) spreads over all configs.
    for repetition in range(1, repetitions + 1):
        for cpu_mhz, gpu_mhz in dvfs_points:
            for workload_id in workload_ids:
                index += 1
                print(f"[{index}/{total}] {workload_id} cpu={cpu_mhz} gpu={gpu_mhz} rep={repetition}", flush=True)
                try:
                    results.append(run_workload(workload_id, cpu_mhz, gpu_mhz, repetition, cfg, dry_run))
                except Exception as exc:  # keep sweeping; record the failure
                    print(f"  FAILED: {exc!r}", flush=True)
                    results.append({"workload_config": workload_id, "cpu_freq_target_mhz": cpu_mhz, "gpu_freq_target_mhz": gpu_mhz,
                                    "repetition": repetition, "status": "failed", "error": repr(exc)})
                if cooldown_s and index < total:
                    time.sleep(cooldown_s)
    return results

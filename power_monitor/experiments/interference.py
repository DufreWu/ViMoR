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
# DVFS point value meaning "stock dynamic governors" (dvfs.default_governor) instead of a fixed clock.
DEFAULT_GOVERNOR = "default"
ALL_APPS = VISION_APPS + ("llm",)

# Columns of runs_index.csv that describe the run (results follow after these).
METADATA_FIELDS = [
    "run_id",
    "sweep_id",
    "run_order",
    "workload_config",
    "workload_apps",
    "status",
    "valid",
    "quality_flags",
    "error",
    "cpu_freq_target_mhz",
    "gpu_freq_target_mhz",
    "cpu_freq_requested_mhz",
    "gpu_freq_requested_mhz",
    "cpu_freq_actual_mhz",
    "gpu_freq_actual_mhz",
    "governor_mode",
    "nvpmodel_mode",
    "emc_clk_cap_mhz",
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
]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
def load_experiment_config(path: str | os.PathLike[str] | None = None) -> Dict[str, Any]:
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if (not config.get("dvfs") or not config.get("jetson")) and DEFAULT_DVFS_PATH.exists():
        with open(DEFAULT_DVFS_PATH, "r", encoding="utf-8") as dvfs_handle:
            dvfs_file = yaml.safe_load(dvfs_handle) or {}
        if not config.get("dvfs"):
            config["dvfs"] = dvfs_file.get("dvfs", {})
        if not config.get("jetson"):
            config["jetson"] = dvfs_file.get("jetson", {})
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


DVFS_GRIDS = ("coarse", "full")


def build_dvfs_matrix(cfg: Dict[str, Any], grid: str = "coarse") -> List[Tuple[int, int]]:
    """``[(cpu_mhz, gpu_mhz)]`` for a DVFS grid.

    coarse - ``dvfs.cpu_freq_mhz`` x ``dvfs.gpu_freq_mhz`` (3 x 3 by default)
    full   - every combination of ``jetson.cpu_available_frequencies_hz`` (kHz) x
             ``jetson.gpu_available_frequencies_hz`` (Hz) from dvfs.yaml (12 x 7 = 84 by default)
    """
    if grid == "coarse":
        dvfs_cfg = cfg.get("dvfs", {})
        cpu_values = [int(v) for v in dvfs_cfg.get("cpu_freq_mhz", [576, 1190, 1984])]
        gpu_values = [int(v) for v in dvfs_cfg.get("gpu_freq_mhz", [306, 612, 918])]
    elif grid == "full":
        jetson = cfg.get("jetson", {})
        cpu_khz, gpu_hz = jetson.get("cpu_available_frequencies_hz"), jetson.get("gpu_available_frequencies_hz")
        if not cpu_khz or not gpu_hz:
            raise ValueError("grid 'full' needs jetson.cpu_available_frequencies_hz and jetson.gpu_available_frequencies_hz in dvfs.yaml")
        # MHz labels as used everywhere else (1190400 kHz -> 1190); validate_dvfs_point maps them back to the OPP.
        cpu_values = sorted({int(v) // 1000 for v in cpu_khz})
        gpu_values = sorted({int(v) // 1_000_000 for v in gpu_hz})
    else:
        raise ValueError(f"Unknown DVFS grid {grid!r}; expected one of {DVFS_GRIDS}")
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


def read_emc_clk_cap_mhz() -> Optional[float]:
    """nvpmodel EMC clock cap (memory bandwidth ceiling); not changed by this benchmark, only recorded."""
    try:
        return int(Path("/sys/kernel/nvpmodel_clk_cap/emc").read_text(encoding="utf-8").strip()) / 1e6
    except (OSError, ValueError):
        return None


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
# DVFS validation and hardware state
# ---------------------------------------------------------------------------
def validate_dvfs_point(controller: DvfsController, cpu_mhz: float, gpu_mhz: float, tolerance_mhz: float = 1.0) -> Tuple[int, int, List[str]]:
    """Map MHz to the nearest OPP (CPU kHz, GPU Hz) and report errors if it is not a supported OPP."""
    cpu_khz, gpu_hz = resolve_frequencies(controller, cpu_mhz, gpu_mhz)
    errors = []
    if controller.available_cpu_hz and abs(cpu_khz / 1000.0 - cpu_mhz) > tolerance_mhz:
        errors.append(f"CPU {cpu_mhz} MHz is not a supported OPP (nearest {cpu_khz / 1000.0} MHz)")
    if controller.available_gpu_hz and abs(gpu_hz / 1e6 - gpu_mhz) > tolerance_mhz:
        errors.append(f"GPU {gpu_mhz} MHz is not a supported OPP (nearest {gpu_hz / 1e6} MHz)")
    return cpu_khz, gpu_hz, errors


def _temps(snapshot: Dict[str, Any], suffix: str) -> Dict[str, Any]:
    return {f"temp_{zone}_{suffix}_c": snapshot.get(f"temp_{zone}_c") for zone in ("cpu", "gpu", "tj")}


def wait_for_thermal_baseline(telemetry: Any, max_temp_c: Optional[float], timeout_s: float, zone: str = "tj") -> Dict[str, Any]:
    """Block until ``temp_<zone>`` <= ``max_temp_c`` (or timeout). Reduces carry-over heat between runs."""
    t_start = time.monotonic()
    temp = telemetry.snapshot().get(f"temp_{zone}_c")
    if max_temp_c is None or temp is None:
        return {"thermal_wait_s": 0.0, "thermal_gate_met": None}
    while temp > max_temp_c and time.monotonic() - t_start < timeout_s:
        time.sleep(2.0)
        temp = telemetry.snapshot().get(f"temp_{zone}_c")
    return {"thermal_wait_s": round(time.monotonic() - t_start, 1), "thermal_gate_met": temp <= max_temp_c}


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
    sweep_id: Optional[str] = None,
    run_order: Optional[int] = None,
) -> Dict[str, Any]:
    """Run one workload at one static DVFS point. ``cpu_mhz``/``gpu_mhz`` = None keeps current clocks.

    Protocol: validate -> apply DVFS -> verify clocks/hardware state -> (thermal gate) ->
    load + init-warmup apps -> start telemetry -> common t0 -> workload warmup ->
    measurement window -> stop telemetry -> completeness/quality checks -> save.
    A run that raises is still written (``status: failed``, partial data) and indexed;
    the exception is then re-raised.
    """
    from power_monitor.apps.llm_app import guess_quantization
    from power_monitor.monitors.telemetry import HardwareTelemetry
    from power_monitor.sensors.camera import CameraSource, SyntheticCamera

    cfg = config or load_experiment_config()
    workloads = cfg.get("workloads", {})
    if workload_id not in workloads:
        raise KeyError(f"Unknown workload {workload_id!r}; configured: {sorted(workloads)}")
    apps_in_workload: List[str] = list(workloads[workload_id] or [])
    seed = int(cfg.get("seed", 0))
    warmup_s = float(cfg.get("warmup_seconds", 30))
    measure_s = float(cfg.get("measurement_seconds", 180))
    llm_cfg = cfg.get("llm", {})
    apps_cfg = cfg.get("apps", {})
    power_cfg = cfg.get("power", {})
    tele_cfg = cfg.get("telemetry", {})
    thermal_cfg = cfg.get("thermal", {})
    quality_cfg = cfg.get("quality", {})
    power_interval_s = float(power_cfg.get("sample_interval_ms", 200)) / 1000.0
    tele_interval_s = float(tele_cfg.get("sample_interval_ms", 200)) / 1000.0
    sat_window_s = float(cfg.get("fps_satisfaction_window_s", 1.0))
    seed_everything(seed)

    start_time = time.strftime("%Y-%m-%dT%H:%M:%S")
    tag = lambda v: "cur" if v is None else (v if isinstance(v, str) else str(int(v)))  # noqa: E731
    run_id = f"{workload_id}_cpu{tag(cpu_mhz)}_gpu{tag(gpu_mhz)}_rep{repetition:02d}_{time.strftime('%Y%m%d-%H%M%S')}"
    output_root = Path(cfg.get("output_root", "outputs/power_monitor"))
    run_dir = output_root / "runs" / run_id
    governor_default = DEFAULT_GOVERNOR in (cpu_mhz, gpu_mhz)
    if governor_default and cpu_mhz != gpu_mhz:
        raise ValueError(f"Default-governor mode needs cpu and gpu both {DEFAULT_GOVERNOR!r}, got {cpu_mhz!r}/{gpu_mhz!r}")
    dvfs_on = cpu_mhz is not None and gpu_mhz is not None and not dry_run and not governor_default
    governor_mode = "default" if governor_default else ("fixed" if cpu_mhz is not None else "current")
    has = apps_in_workload.__contains__

    metadata: Dict[str, Any] = {
        "run_id": run_id,
        "sweep_id": sweep_id,
        "run_order": run_order,
        "workload_config": workload_id,
        "workload_apps": "+".join(apps_in_workload) if apps_in_workload else "idle",
        "cpu_freq_target_mhz": cpu_mhz,
        "gpu_freq_target_mhz": gpu_mhz,
        "cpu_freq_requested_mhz": None,
        "gpu_freq_requested_mhz": None,
        "cpu_freq_actual_mhz": None,
        "gpu_freq_actual_mhz": None,
        "nvpmodel_mode": None,
        "llm_model": llm_cfg.get("model") if has("llm") else None,
        "llm_quantization": (llm_cfg.get("quantization") or guess_quantization(llm_cfg.get("model_path", ""))) if has("llm") else None,
        "yolo_model": apps_cfg.get("yolo", {}).get("model") if has("yolo") else None,
        "segformer_model": apps_cfg.get("segformer", {}).get("model") if has("segformer") else None,
        "input_resolution": None,
        "target_fps": {a: apps_cfg[a]["target_fps"] for a in apps_in_workload if a in VISION_APPS},
        "request_schedule": None,
        "measurement_seconds": measure_s,
        "warmup_seconds": warmup_s,
        "seed": seed,
        "repetition": repetition,
        "start_time": start_time,
        "dry_run": dry_run,
        "status": "running",
        "valid": None,
        "quality_flags": [],
        "error": None,
        "failed_phase": None,
        "dvfs_control": dvfs_on,
        "governor_mode": governor_mode,
        "power_sample_interval_ms": power_interval_s * 1000.0,
        "telemetry_sample_interval_ms": tele_interval_s * 1000.0,
        "fps_satisfaction_window_s": sat_window_s,
        "warnings": [],
    }
    summary: Dict[str, Any] = {}
    phase = "init"

    controller = DvfsController()
    controller.save_original_state()
    telemetry = HardwareTelemetry(tele_interval_s, use_tegrastats=bool(tele_cfg.get("tegrastats", True)) and not dry_run)
    camera = None
    vision_apps: Dict[str, Any] = {}
    llm_app = None
    power = None
    fps_monitor = None
    win_start = win_end = None
    try:
        # 1. validate the DVFS configuration ---------------------------------
        phase = "validate"
        metadata["nvpmodel_mode"] = read_nvpmodel_mode()
        cpu_khz = gpu_hz = None
        if dvfs_on:
            cpu_khz, gpu_hz, errors = validate_dvfs_point(controller, cpu_mhz, gpu_mhz, float(quality_cfg.get("freq_tolerance_mhz", 1.0)))
            if errors:
                raise ValueError("; ".join(errors))
            metadata["cpu_freq_requested_mhz"] = cpu_khz / 1000.0
            metadata["gpu_freq_requested_mhz"] = gpu_hz / 1e6

        # 2. apply + 3. verify ------------------------------------------------
        phase = "apply_dvfs"
        if dvfs_on:
            try:
                controller.set_cpu_frequency(cpu_khz)
                controller.set_gpu_frequency(gpu_hz)
            except PermissionError as exc:
                raise PermissionError("Setting DVFS needs root: run with sudo -E, or use --no-dvfs") from exc
            time.sleep(0.5)
            pinned = telemetry.snapshot()
            tol = float(quality_cfg.get("freq_tolerance_mhz", 1.0))
            cpu_ok = all(abs((pinned.get(k) or -1) - cpu_khz / 1000.0) <= tol for k in ("cpu_freq_actual_min_mhz", "cpu_freq_actual_max_mhz"))
            gpu_ok = abs((pinned.get("gpu_freq_actual_mhz") or -1) - gpu_hz / 1e6) <= tol
            if not (cpu_ok and gpu_ok):
                raise RuntimeError(f"DVFS lock failed: requested cpu={cpu_khz / 1000.0} gpu={gpu_hz / 1e6} MHz, "
                                   f"actual cpu=[{pinned.get('cpu_freq_actual_min_mhz')}, {pinned.get('cpu_freq_actual_max_mhz')}] "
                                   f"gpu={pinned.get('gpu_freq_actual_mhz')}")
        if governor_default and not dry_run:
            gov_cfg = cfg.get("dvfs", {}).get("default_governor", {})
            try:
                controller.set_default_governors(gov_cfg.get("cpu", "schedutil"), gov_cfg.get("gpu", "nvhost_podgov"))
            except PermissionError as exc:
                raise PermissionError("Setting governors needs root: run with sudo -E") from exc
        metadata.update(_read_governors())
        metadata["emc_clk_cap_mhz"] = read_emc_clk_cap_mhz()

        phase = "thermal_gate"
        if not dry_run:
            summary.update(wait_for_thermal_baseline(telemetry, thermal_cfg.get("max_start_temp_c"), float(thermal_cfg.get("max_wait_s", 300))))

        # 4. launch + init-warmup (model loading is outside the timed region) --
        phase = "launch"
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
        if llm_app is not None:
            llm_app.warmup(1)  # process mode: spawns the LLM process and waits until it is ready
        for app in vision_apps.values():
            app.warmup(0.0, min_iters=3)

        # Pre-run hardware state (after model load, right before the timed region).
        phase = "pre_state"
        pre = telemetry.snapshot()
        summary.update(_temps(pre, "start"))

        # 5-8. synchronised start, telemetry, workload warmup + measurement ----
        phase = "measure"
        power = SyntheticPowerMonitor() if dry_run else JetsonPowerMonitor(sensor_dir=power_cfg.get("sensor_dir", "auto"))
        power.start(power_interval_s)
        telemetry.start()
        total_s = warmup_s + measure_s
        t0 = time.monotonic() + 0.2
        win_start, win_end = t0 + warmup_s, t0 + total_s
        metadata["measurement_window_monotonic"] = [win_start, win_end]
        threads = [app.start_thread(total_s, t0) for app in vision_apps.values()]
        if llm_app is not None:
            threads.append(llm_app.start_thread(total_s, t0))
        time.sleep(max(0.0, win_end - time.monotonic()))
        for thread in threads:
            thread.join()
        time.sleep(power_interval_s * 1.5)  # one more power sample past win_end for edge interpolation

        # 9. stop telemetry -----------------------------------------------------
        phase = "collect"
        power.stop()
        telemetry.stop()
        for app in list(vision_apps.values()) + ([llm_app] if llm_app else []):
            if app.error is not None:
                raise app.error
        post = telemetry.snapshot()
        summary.update(_temps(post, "end"))
        nvp_end = read_nvpmodel_mode()
        governors_end = _read_governors()
        if not dry_run and governors_end != {k: metadata.get(k) for k in governors_end}:
            metadata["warnings"].append(f"governors changed during run: {governors_end}")
            metadata["governor_changed"] = True
        if nvp_end != metadata["nvpmodel_mode"]:
            metadata["warnings"].append(f"nvpmodel changed during run: {metadata['nvpmodel_mode']} -> {nvp_end}")
        fps_monitor.start(win_start)
        fps_monitor.stop(win_end)

        # ----- hardware metrics ----------------------------------------------
        cpu_req = metadata["cpu_freq_requested_mhz"]
        gpu_req = metadata["gpu_freq_requested_mhz"]
        power_window = power.window(win_start, win_end)
        energy = power.integrate_window(win_start, win_end)
        summary.update({
            "mean_power_w": energy["mean_w"],
            "peak_power_w": max((s["power_w"] for s in power_window), default=None),
            "energy_j": energy["energy_j"],
            "power_samples": len(power_window),
            "power_covered_s": energy["covered_s"],
            "power_max_gap_s": energy["max_gap_s"],
        })
        for rail in [name for _, name in power.rails()][1:]:
            rail_e = power.integrate_window(win_start, win_end, key=f"{rail}_w")
            summary[f"mean_{rail}_w"] = rail_e["mean_w"]
            summary[f"energy_{rail}_j"] = rail_e["energy_j"]
        hw = telemetry.summary(win_start, win_end, cpu_req, gpu_req, float(quality_cfg.get("freq_tolerance_mhz", 1.0)))
        summary.update(hw)
        summary["cpu_freq_requested_mhz"] = cpu_req
        summary["gpu_freq_requested_mhz"] = gpu_req
        metadata["cpu_freq_actual_mhz"] = hw.get("cpu_freq_actual_mean")
        metadata["gpu_freq_actual_mhz"] = hw.get("gpu_freq_actual_mean")

        # ----- application metrics ---------------------------------------------
        for name, app in vision_apps.items():
            stream = fps_monitor.stream(name, "output").summary(win_start, win_end, fps_monitor.window_s)
            summary.update({
                f"{name}_fps_mean": stream["mean_fps"],
                f"{name}_fps_p5": stream["window_fps_p5"],
                f"{name}_fps_attainment": stream["fps_attainment"],
                f"{name}_frame_interval_p99_ms": stream["interval_p99_ms"],
                f"{name}_jitter_ms": stream["jitter_ms"],
                f"{name}_drop_rate": stream["drop_rate"],
            })
            summary.update(app.summary(win_start, win_end, sat_window_s))
        if camera is not None:
            summary["camera_fps_mean"] = fps_monitor.stream(camera.name, "capture").summary(win_start, win_end)["mean_fps"]
        if llm_app is not None:
            summary.update(llm_app.summary(win_start, win_end))
        if len(vision_apps) > 0:
            qs = [summary[f"{n}_fps_q"] for n in vision_apps if summary.get(f"{n}_fps_q") is not None]
            summary["all_apps_fps_target_met"] = bool(qs) and all(q >= 1.0 - 1e-9 for q in qs)

        # 10. completeness + quality -------------------------------------------
        flags = _quality_flags(summary, metadata, vision_apps, llm_app, measure_s, power_interval_s, tele_interval_s, quality_cfg, dvfs_on, dry_run, total_s, seed)
        metadata["quality_flags"] = flags
        metadata["valid"] = not any(f in HARD_QUALITY_FLAGS for f in flags)
        metadata["status"] = "completed"

        # ----- metadata extras -------------------------------------------------
        input_resolution: Dict[str, Any] = {}
        if camera is not None and camera.latest() is not None:
            input_resolution["camera"] = list(camera.latest().image.shape[:2])
        for name, app in vision_apps.items():
            input_resolution[name] = getattr(app.infer_fn, "input_resolution", None)
        if llm_app is not None:
            input_resolution["llm"] = {"prompt_tokens": llm_app.prompt_tokens, "decode_tokens": llm_app.decode_tokens}
        metadata.update({
            "input_resolution": input_resolution,
            "request_schedule": llm_app.schedule.describe(total_s, seed) if llm_app else None,
            "camera": camera.properties() if camera is not None else None,
            "llm_config": {k: llm_cfg.get(k, "process" if k == "isolation" else None) for k in ("model_path", "n_gpu_layers", "n_ctx", "prompt_tokens", "decode_tokens", "stop_at_eos", "isolation")} if has("llm") else None,
            "pre_run_hardware": pre,
            "post_run_hardware": post,
            "nvpmodel_mode_end": nvp_end,
        })

        # 11. save -------------------------------------------------------------
        phase = "save"
        _write_run_outputs(run_dir, metadata, summary, vision_apps, llm_app, fps_monitor, power, win_start, win_end, float(llm_cfg.get("sample_interval_ms", 100)) / 1000.0)
        telemetry.write_csv(run_dir, win_start, win_end)
        _append_runs_index(output_root / "runs_index.csv", metadata, summary)
        _chown_to_sudo_user(output_root, output_root / "runs", run_dir, output_root / "runs_index.csv")
        return {**{k: metadata.get(k) for k in METADATA_FIELDS}, **summary}
    except BaseException as exc:
        # Do not silently discard: persist whatever was collected and index the failure.
        import traceback

        metadata.update(status="failed", valid=False, error=repr(exc), failed_phase=phase)
        try:
            power and power.stop()
            telemetry.stop()
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
            with open(run_dir / "metadata.json", "w", encoding="utf-8") as handle:
                json.dump(metadata, handle, indent=2, default=str)
            with open(run_dir / "summary.json", "w", encoding="utf-8") as handle:
                json.dump(summary, handle, indent=2, default=str)
            if power is not None and power.samples:
                power.write_csv(run_dir / "power.csv")
            telemetry.write_csv(run_dir, win_start, win_end)
            _append_runs_index(output_root / "runs_index.csv", metadata, summary)
            _chown_to_sudo_user(output_root, output_root / "runs", run_dir, output_root / "runs_index.csv")
        except Exception as save_exc:  # never mask the original error
            print(f"  (could not persist failed run {run_id}: {save_exc!r})", flush=True)
        raise
    finally:
        if power is not None:
            power.stop()
        telemetry.stop()
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


# Flags that make a run invalid for analysis (kept in the index, excluded by default when aggregating).
HARD_QUALITY_FLAGS = {"freq_lock_failed", "governor_changed", "thermal_throttled", "power_throttled", "incomplete_power", "incomplete_telemetry",
                      "incomplete_app_data", "llm_no_completed_requests"}


def _quality_flags(summary: Dict[str, Any], metadata: Dict[str, Any], vision_apps: Dict[str, Any], llm_app: Any, measure_s: float,
                   power_interval_s: float, tele_interval_s: float, quality_cfg: Dict[str, Any], dvfs_on: bool, dry_run: bool,
                   total_s: float, seed: int) -> List[str]:
    flags: List[str] = []
    min_cov = float(quality_cfg.get("min_sample_coverage", 0.9))
    max_gap = float(quality_cfg.get("max_gap_factor", 5.0))
    if summary.get("energy_j") is None or (summary.get("power_covered_s") or 0) < 0.99 * measure_s \
            or summary.get("power_samples", 0) < min_cov * measure_s / power_interval_s \
            or (summary.get("power_max_gap_s") or 0) > max_gap * power_interval_s:
        flags.append("incomplete_power")
    if summary.get("telemetry_samples", 0) < min_cov * measure_s / tele_interval_s:
        flags.append("incomplete_telemetry")
    if not dry_run and summary.get("tegrastats_samples", 0) < min_cov * measure_s / tele_interval_s:
        flags.append("incomplete_tegrastats")  # soft: utilisation/EMC partly missing
    if metadata.get("governor_changed"):
        flags.append("governor_changed")
    if dvfs_on:
        lock_min = float(quality_cfg.get("min_freq_lock_rate", 0.99))
        if (summary.get("cpu_freq_lock_rate") or 0) < lock_min or (summary.get("gpu_freq_lock_rate") or 0) < lock_min:
            flags.append("freq_lock_failed")
    if summary.get("thermal_throttled"):
        flags.append("thermal_throttled")
    if summary.get("power_throttled"):
        flags.append("power_throttled")
    if summary.get("thermal_gate_met") is False:
        flags.append("thermal_gate_timeout")  # soft: started warmer than configured
    for name, app in vision_apps.items():
        expected = int(round(measure_s * app.target_fps))
        if abs((summary.get(f"{name}_releases") or 0) - expected) > 1:
            flags.append("incomplete_app_data")
            break
    if llm_app is not None:
        offsets = llm_app.schedule.arrival_offsets(total_s, seed)
        expect_arrivals = offsets is None or any(metadata["warmup_seconds"] <= t < total_s for t in offsets)
        if expect_arrivals and not summary.get("llm_requests_completed"):
            flags.append("llm_no_completed_requests")
        if summary.get("llm_requests_failed"):
            flags.append("llm_request_failures")
    return flags


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
        json.dump(metadata, handle, indent=2, default=str)
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
PLAN_FIELDS = ["order", "repetition", "workload", "cpu_mhz", "gpu_mhz", "status", "run_id", "valid", "quality_flags", "error"]


def build_sweep_plan(cfg: Dict[str, Any], workload_ids: List[str], dvfs_points: List[Tuple[Optional[int], Optional[int]]],
                     repetitions: int, randomize: bool = True) -> List[Dict[str, Any]]:
    """Repetition is the outer loop (slow drift spreads over all configs); within each
    (repetition, workload) the DVFS points are shuffled with a seed derived from
    (seed, workload, repetition), so the order is random but reproducible."""
    seed = int(cfg.get("seed", 0))
    plan: List[Dict[str, Any]] = []
    for repetition in range(1, repetitions + 1):
        for workload_id in workload_ids:
            points = list(dvfs_points)
            if randomize:
                random.Random(f"{seed}:{workload_id}:{repetition}").shuffle(points)
            for cpu_mhz, gpu_mhz in points:
                plan.append({"order": len(plan) + 1, "repetition": repetition, "workload": workload_id, "cpu_mhz": cpu_mhz,
                             "gpu_mhz": gpu_mhz, "status": "pending", "run_id": "", "valid": "", "quality_flags": "", "error": ""})
    return plan


def _write_plan(path: Path, plan: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".csv.tmp")
    _write_csv(tmp, [{k: _flatten(v) for k, v in row.items()} for row in plan], PLAN_FIELDS)
    tmp.replace(path)


def _read_plan(path: Path) -> List[Dict[str, Any]]:
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["order"], row["repetition"] = int(row["order"]), int(row["repetition"])
        for key in ("cpu_mhz", "gpu_mhz"):
            value = row[key]
            row[key] = None if value in ("", "None") else (value if value == DEFAULT_GOVERNOR else int(float(value)))
    return rows


def run_sweep(cfg: Optional[Dict[str, Any]] = None, workloads: Optional[List[str]] = None, dry_run: bool = False,
              no_dvfs: bool = False, resume: Optional[str] = None, retry_failed: bool = False,
              governor_default: bool = False, grid: str = "coarse",
              points: Optional[List[Tuple[Any, Any]]] = None) -> List[Dict[str, Any]]:
    """Run workloads x valid DVFS points x repetitions following ``build_sweep_plan``.

    The plan and per-entry status are kept in ``<output_root>/sweeps/<sweep_id>/plan.csv``
    and updated after every run, so an interrupted sweep continues with ``resume=<sweep_id>``
    (pending entries; failed ones too with ``retry_failed``). Invalid DVFS points are
    recorded in the plan as ``invalid_config`` instead of being dropped.
    """
    cfg = cfg or load_experiment_config()
    output_root = Path(cfg.get("output_root", "outputs/power_monitor"))
    cooldown_s = float(cfg.get("cooldown_seconds", 0))

    if resume:
        sweep_id = resume
        plan_path = output_root / "sweeps" / sweep_id / "plan.csv"
        if not plan_path.exists():
            raise FileNotFoundError(f"No sweep plan at {plan_path}")
        plan = _read_plan(plan_path)
    else:
        sweep_id = time.strftime("sweep_%Y%m%d-%H%M%S")
        plan_path = output_root / "sweeps" / sweep_id / "plan.csv"
        workload_ids = workloads or list(cfg.get("workloads", {}))
        dvfs_points: List[Tuple[Any, Any]] = list(points) if points is not None else ([] if no_dvfs else list(build_dvfs_matrix(cfg, grid)))
        if governor_default and points is None:
            dvfs_points.append((DEFAULT_GOVERNOR, DEFAULT_GOVERNOR))  # shuffled together with the fixed points
        if not dvfs_points:
            dvfs_points = [(None, None)]  # current clocks, untouched
        repetitions = int(cfg.get("repetitions", 5))
        plan = build_sweep_plan(cfg, workload_ids, dvfs_points, repetitions, bool(cfg.get("randomize_dvfs_order", True)))
        if not dry_run:
            controller = DvfsController()
            tol = float(cfg.get("quality", {}).get("freq_tolerance_mhz", 1.0))
            for row in plan:
                if row["cpu_mhz"] in (None, DEFAULT_GOVERNOR) or row["gpu_mhz"] in (None, DEFAULT_GOVERNOR):
                    continue
                errors = validate_dvfs_point(controller, row["cpu_mhz"], row["gpu_mhz"], tol)[2]
                if errors:
                    row.update(status="invalid_config", error="; ".join(errors))
        plan_path.parent.mkdir(parents=True, exist_ok=True)
        with open(plan_path.parent / "sweep_config.yaml", "w", encoding="utf-8") as handle:
            yaml.safe_dump({**cfg, "sweep_id": sweep_id, "dry_run": dry_run, "no_dvfs": no_dvfs, "grid": None if no_dvfs else grid,
                            "governor_default": governor_default}, handle, sort_keys=False)
        _write_plan(plan_path, plan)
    print(f"Sweep {sweep_id}: plan at {plan_path}", flush=True)

    todo_status = {"pending", "running"} | ({"failed"} if retry_failed else set())
    todo = [row for row in plan if row["status"] in todo_status]
    # Rough wall-clock estimate: window + cooldown + ~30 s model load / thermal gate per run.
    per_run_s = float(cfg.get("warmup_seconds", 30)) + float(cfg.get("measurement_seconds", 180)) + cooldown_s + 30.0
    print(f"{len(todo)} runs to do, estimated {len(todo) * per_run_s / 3600:.1f} h", flush=True)
    skipped = [row for row in plan if row["status"] == "invalid_config"]
    for row in skipped:
        print(f"  SKIP invalid config #{row['order']}: {row['workload']} cpu={row['cpu_mhz']} gpu={row['gpu_mhz']}: {row['error']}", flush=True)
    results: List[Dict[str, Any]] = []
    for i, row in enumerate(todo, 1):
        print(f"[{i}/{len(todo)}] #{row['order']} {row['workload']} cpu={row['cpu_mhz']} gpu={row['gpu_mhz']} rep={row['repetition']}", flush=True)
        row.update(status="running", error="")
        _write_plan(plan_path, plan)
        try:
            result = run_workload(row["workload"], row["cpu_mhz"], row["gpu_mhz"], row["repetition"], cfg, dry_run, sweep_id=sweep_id, run_order=row["order"])
            row.update(status="completed", run_id=result["run_id"], valid=result.get("valid"), quality_flags=result.get("quality_flags"))
            if result.get("quality_flags"):
                print(f"  flags: {result['quality_flags']}", flush=True)
            results.append(result)
        except KeyboardInterrupt:
            row.update(status="pending", error="interrupted")
            _write_plan(plan_path, plan)
            print(f"Interrupted. Resume with: --sweep --resume {sweep_id}", flush=True)
            raise
        except Exception as exc:  # keep sweeping; the failed run is persisted by run_workload
            print(f"  FAILED: {exc!r}", flush=True)
            row.update(status="failed", valid=False, error=repr(exc))
            results.append({"workload_config": row["workload"], "cpu_freq_target_mhz": row["cpu_mhz"], "gpu_freq_target_mhz": row["gpu_mhz"],
                            "repetition": row["repetition"], "status": "failed", "error": repr(exc)})
        _write_plan(plan_path, plan)
        _chown_to_sudo_user(output_root / "sweeps", plan_path.parent)
        if cooldown_s and i < len(todo):
            time.sleep(cooldown_s)
    return results

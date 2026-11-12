"""Tests for the DVFS characterization additions: telemetry parsing, energy integration,
FPS-target satisfaction, LLM metrics, sweep planning, and failed-run persistence."""

import csv
import json

import pytest

from power_monitor.monitors.power_monitor import JetsonPowerMonitor
from power_monitor.monitors.telemetry import parse_tegrastats_line

ROOT_LINE = ("11-09-2026 15:17:31 RAM 5463/15655MB (lfb 136x4MB) SWAP 289/7827MB (cached 2MB) "
             "CPU [10%@1190,off,20%@1190,0%@1190,0%@1190,0%@1190,4%@1190,0%@1190] EMC_FREQ 3%@3199 "
             "GR3D_FREQ 45%@[612] cpu@48.406C gpu@45.343C tj@48.531C VDD_IN 5262mW/5262mW VDD_SOC 1509mW/1509mW")
USER_LINE = ("11-09-2026 15:17:31 RAM 5463/15655MB (lfb 136x4MB) SWAP 289/7827MB (cached 2MB) "
             "CPU [0%@1036,0%@1036] GR3D_FREQ 7% cpu@48.406C VDD_IN 5262mW/5262mW")


def test_tegrastats_parser_root_and_user_formats():
    root = parse_tegrastats_line(ROOT_LINE)
    assert root["ram_used_mb"] == 5463 and root["emc_freq_mhz"] == 3199 and root["emc_util_pct"] == 3
    assert root["gpu_util_pct"] == 45 and root["gpu_freq_mhz_tegrastats"] == 612
    assert root["cpu_online_cores"] == 7 and root["cpu1_util_pct"] is None
    assert root["cpu_util_pct"] == pytest.approx(34 / 7)
    assert root["temp_tj_c"] == pytest.approx(48.531) and root["vdd_in_mw_tegrastats"] == 5262
    user = parse_tegrastats_line(USER_LINE)
    assert user["gpu_util_pct"] == 7 and "emc_freq_mhz" not in user and "gpu_freq_mhz_tegrastats" not in user


def test_window_energy_uses_actual_timestamps_and_interpolates_edges():
    pm = JetsonPowerMonitor(sensor_dir="/nonexistent")
    # Irregular sampling; power ramps linearly 0 -> 10 W over t = 0..10 s.
    pm.samples = [{"monotonic": t, "power_w": t} for t in (0.0, 0.15, 1.0, 1.3, 4.0, 7.7, 10.0)]
    res = pm.integrate_window(2.0, 6.0)
    assert res["energy_j"] == pytest.approx((6 ** 2 - 2 ** 2) / 2)  # integral of t dt
    assert res["mean_w"] == pytest.approx(4.0) and res["covered_s"] == pytest.approx(4.0)
    assert res["max_gap_s"] == pytest.approx(3.7)


def _vision_app(target_fps=10.0):
    from power_monitor.apps.vision_app import PeriodicVisionApp

    return PeriodicVisionApp("yolo", lambda img: None, camera=None, target_fps=target_fps, deadline_ms=100)


def _frame(release, infer_ms=20.0, skipped=False):
    if skipped:
        return {"scheduled_release_ts": release, "deadline_met": False, "skipped": True}
    return {"scheduled_release_ts": release, "inference_start_ts": release + 0.001, "inference_end_ts": release + 0.001 + infer_ms / 1000,
            "response_ms": infer_ms + 1, "e2e_ms": infer_ms + 30, "deadline_met": True, "skipped": False}


def test_fps_satisfaction_is_robust_to_completion_jitter_and_counts_misses():
    app = _vision_app(10.0)
    # 5 s at 10 fps; completions jitter by up to ~95 ms (would cross window edges if bucketed by completion).
    app.frames = [_frame(100.0 + i / 10, infer_ms=95.0 if i % 2 else 5.0) for i in range(50)]
    s = app.summary(100.0, 105.0, 1.0)
    assert s["yolo_fps_satisfaction_rate"] == 1.0 and s["yolo_fps_q"] == pytest.approx(1.0)
    assert s["yolo_fps_satisfaction_windows"] == 5 and s["yolo_processed_frames"] == 50
    # Skip one frame in window 2 -> 4/5 windows satisfied, Q = 49/50.
    app.frames[23] = _frame(100.0 + 2.3, skipped=True)
    s = app.summary(100.0, 105.0, 1.0)
    assert s["yolo_fps_satisfaction_rate"] == pytest.approx(0.8)
    assert s["yolo_fps_q"] == pytest.approx(0.98)
    assert s["yolo_deadline_miss_rate"] == pytest.approx(1 / 50)
    assert s["yolo_inference_p50_ms"] == pytest.approx(5.0, abs=50)


def test_llm_summary_separates_failed_and_truncated_requests():
    from power_monitor.apps.llm_app import LlmApp, RequestSchedule

    app = LlmApp(backend=None, prompt_tokens=128, decode_tokens=4, schedule=RequestSchedule())

    def req(arrival, status, n=4, ttft=0.2, tpot=0.05):
        first = arrival + ttft
        ts = [first + k * tpot for k in range(n)] if status != "failed" else []
        return {"arrival_ts": arrival, "start_ts": arrival, "first_token_ts": ts[0] if ts else None, "end_ts": ts[-1] if ts else arrival,
                "prompt_tokens": 128, "generated_tokens": len(ts), "status": status, "token_ts": ts,
                "ttft_ms": ttft * 1000 if ts else None, "tpot_ms": tpot * 1000 if len(ts) > 1 else None,
                "e2e_ms": ((ts[-1] if ts else arrival) - arrival) * 1000, "queue_ms": 0.0,
                "decode_tps": 1 / tpot if len(ts) > 1 else None}

    app.requests = [req(0.0, "completed"), req(10.0, "completed"), req(20.0, "failed"), req(29.0, "truncated", n=2), req(-5.0, "completed")]
    s = app.summary(0.0, 30.0)
    assert s["llm_requests_arrived"] == 4 and s["llm_requests_completed"] == 2
    assert s["llm_requests_failed"] == 1 and s["llm_requests_truncated"] == 1
    assert s["llm_failure_rate"] == pytest.approx(1 / 3)
    assert s["llm_ttft_ms_p50"] == pytest.approx(200.0) and s["llm_tpot_ms_p95"] == pytest.approx(50.0)
    assert s["llm_decode_tps_mean"] == pytest.approx(20.0) and s["llm_e2e_s_p50"] == pytest.approx(0.35)
    assert s["llm_input_tokens_mean"] == 128 and s["llm_output_tokens_mean"] == 4
    assert s["llm_timing_source"] == "streaming_token_timestamps"


def test_sweep_plan_is_randomized_per_workload_and_reproducible():
    from power_monitor.experiments.interference import build_sweep_plan

    cfg = {"seed": 42}
    points = [(c, g) for c in (576, 1190, 1984) for g in (306, 612, 918)]
    plan = build_sweep_plan(cfg, ["W1", "W3"], points, repetitions=2)
    assert plan == build_sweep_plan(cfg, ["W1", "W3"], points, repetitions=2)
    assert len(plan) == 2 * 2 * 9 and [r["order"] for r in plan] == list(range(1, 37))
    blocks = [[(r["cpu_mhz"], r["gpu_mhz"]) for r in plan[i:i + 9]] for i in range(0, 36, 9)]
    assert all(sorted(b) == sorted(points) for b in blocks)  # every block covers every point once
    assert len({tuple(b) for b in blocks}) > 1  # orders differ between blocks
    assert [r["repetition"] for r in plan[:18]] == [1] * 18  # repetition is the outer loop
    fixed = build_sweep_plan(cfg, ["W1"], points, repetitions=1, randomize=False)
    assert [(r["cpu_mhz"], r["gpu_mhz"]) for r in fixed] == points


def test_invalid_dvfs_point_is_reported():
    from power_monitor.controllers.dvfs_controller import DvfsController
    from power_monitor.experiments.interference import validate_dvfs_point

    dc = DvfsController()
    dc.available_cpu_hz = [576000, 1190400, 1984000]
    dc.available_gpu_hz = [306000000, 612000000]
    assert validate_dvfs_point(dc, 1190, 612)[2] == []
    _, _, errors = validate_dvfs_point(dc, 1000, 700)
    assert len(errors) == 2


def test_failed_run_is_persisted_and_indexed(tmp_path, monkeypatch):
    from power_monitor.experiments import interference

    cfg = interference.load_experiment_config()
    cfg.update(output_root=str(tmp_path), warmup_seconds=0.2, measurement_seconds=0.6)

    def boom(*args, **kwargs):
        raise RuntimeError("camera exploded")

    monkeypatch.setattr(interference, "_build_vision_app", boom)
    with pytest.raises(RuntimeError, match="camera exploded"):
        interference.run_workload("W1", None, None, config=cfg, dry_run=True)
    run_dir = next((tmp_path / "runs").iterdir())
    meta = json.loads((run_dir / "metadata.json").read_text())
    assert meta["status"] == "failed" and meta["failed_phase"] == "launch" and "camera exploded" in meta["error"]
    assert (run_dir / "error.txt").exists()
    rows = list(csv.DictReader(open(tmp_path / "runs_index.csv")))
    assert rows[-1]["status"] == "failed" and rows[-1]["valid"] == "False"


def test_dry_run_records_quality_and_hardware_fields(tmp_path):
    from power_monitor.experiments.interference import load_experiment_config, run_workload

    cfg = load_experiment_config()
    cfg.update(output_root=str(tmp_path), warmup_seconds=0.2, measurement_seconds=2.0)
    r = run_workload("W1", None, None, config=cfg, dry_run=True)
    assert r["status"] == "completed" and r["valid"] is True and r["quality_flags"] == []
    assert r["energy_j"] == pytest.approx(20.0, rel=0.01)  # synthetic 10 W x 2 s
    for key in ("temp_tj_start_c", "temp_tj_end_c", "thermal_throttled", "power_throttled", "yolo_fps_satisfaction_rate",
                "yolo_inference_p99_ms", "yolo_fps_target"):
        assert key in r, key
    run_dir = next((tmp_path / "runs").iterdir())
    assert (run_dir / "telemetry.csv").exists()


class _FakeDevfreq:
    """Mimics the Jetson behaviour behind the sweep failures: a min_freq write above the
    current max (or max below min) is accepted but ignored, so the clock stays put."""

    def __init__(self, root, cur):
        self.root = root
        (root / "userspace").mkdir()
        self.state = {"min_freq": cur, "max_freq": cur, "cur_freq": cur, "userspace/set_freq": None}
        self._sync()

    def _sync(self):
        for k, v in self.state.items():
            (self.root / k).write_text("undefined" if v is None else str(v))

    def write(self, name, value):
        value = int(value) if name != "governor" else value
        if name == "min_freq" and value > self.state["max_freq"]:
            return
        if name == "max_freq" and value < self.state["min_freq"]:
            return
        if name in self.state:
            self.state[name] = value
        lo, hi = self.state["min_freq"], self.state["max_freq"]
        want = self.state["userspace/set_freq"] or self.state["cur_freq"]
        self.state["cur_freq"] = min(max(want, lo), hi)
        self._sync()


@pytest.mark.parametrize("start,target", [(306000000, 612000000), (612000000, 918000000), (918000000, 306000000), (306000000, 918000000)])
def test_gpu_frequency_lock_works_up_and_down(tmp_path, start, target):
    from power_monitor.controllers.dvfs_controller import DvfsController

    fake = _FakeDevfreq(tmp_path, start)
    dc = DvfsController()
    dc.gpu_path = str(tmp_path)
    dc.available_gpu_hz = [306000000, 408000000, 510000000, 612000000, 714000000, 816000000, 918000000]
    dc._write_text = lambda path, value: fake.write(str(path.relative_to(tmp_path)), value)
    result = dc.set_gpu_frequency(target, settle_timeout_s=0.3)
    assert dc.read_gpu_frequency() == target and result.get("locked", True)
    assert fake.state["min_freq"] == fake.state["max_freq"] == target
    # Restoring the starting state also works in either direction.
    dc.original_state = {"cpu": {}, "gpu": {"governor": "userspace", "min": str(start), "max": str(start), "current": str(start)}}
    dc.restore_original_state()
    assert dc.read_gpu_frequency() == start


def test_set_default_governors_opens_full_range(tmp_path):
    from power_monitor.controllers.dvfs_controller import DvfsController

    cpu_root = tmp_path / "cpu"
    for core in range(2):
        d = cpu_root / f"cpu{core}" / "cpufreq"
        d.mkdir(parents=True)
        for name, value in {"scaling_governor": "userspace", "scaling_available_governors": "ondemand userspace performance schedutil",
                            "cpuinfo_min_freq": "115200", "cpuinfo_max_freq": "1984000", "scaling_min_freq": "1036800",
                            "scaling_max_freq": "1036800"}.items():
            (d / name).write_text(value)
    gpu = tmp_path / "gpu"
    gpu.mkdir()
    for name, value in {"governor": "userspace", "available_governors": "userspace nvhost_podgov performance", "min_freq": "612000000",
                        "max_freq": "612000000"}.items():
        (gpu / name).write_text(value)
    dc = DvfsController()
    dc.cpu_path, dc.gpu_path = str(cpu_root), str(gpu)
    dc.available_gpu_hz = [306000000, 612000000, 918000000]
    state = dc.set_default_governors("schedutil", "nvhost_podgov")
    assert state == {"cpu_governor": ["schedutil"], "gpu_governor": "nvhost_podgov"}
    assert (cpu_root / "cpu1/cpufreq/scaling_min_freq").read_text() == "115200"
    assert (cpu_root / "cpu1/cpufreq/scaling_max_freq").read_text() == "1984000"
    assert (gpu / "min_freq").read_text() == "306000000" and (gpu / "max_freq").read_text() == "918000000"
    with pytest.raises(ValueError, match="not available"):
        dc.set_default_governors("conservative", "nvhost_podgov")


def test_sweep_with_default_governor_point_and_aggregate(tmp_path):
    from power_monitor.analysis.aggregate import aggregate_runs
    from power_monitor.experiments.interference import load_experiment_config, run_sweep

    cfg = load_experiment_config()
    cfg.update(output_root=str(tmp_path), warmup_seconds=0.2, measurement_seconds=1.0, repetitions=1, cooldown_seconds=0)
    cfg["dvfs"]["cpu_freq_mhz"], cfg["dvfs"]["gpu_freq_mhz"] = [576, 1984], [306]
    results = run_sweep(cfg, workloads=["W1"], dry_run=True, governor_default=True)
    assert sorted(str(r["cpu_freq_target_mhz"]) for r in results) == ["1984", "576", "default"]
    default_run = next(r for r in results if r["cpu_freq_target_mhz"] == "default")
    assert default_run["governor_mode"] == "default" and default_run["valid"] is True
    assert "W1_cpudefault_gpudefault_rep01" in default_run["run_id"]
    plan = next((tmp_path / "sweeps").glob("*/plan.csv")).read_text()
    assert "default,default,completed" in plan
    # Only the default point when the fixed matrix is off.
    only = run_sweep(cfg, workloads=["W1"], dry_run=True, no_dvfs=True, governor_default=True)
    assert [r["cpu_freq_target_mhz"] for r in only] == ["default"]
    out = aggregate_runs(tmp_path)
    agg = (out / "aggregate.csv").read_text().splitlines()
    rows = [line.split(",")[2:4] for line in agg[1:]]
    assert rows == [["576", "306"], ["1984", "306"], ["default", "default"]]  # numeric first, default last
    assert "| default | default |" in (out / "report.md").read_text()


def test_full_grid_is_all_list_combinations_and_contains_coarse():
    from power_monitor.experiments.interference import build_dvfs_matrix, load_experiment_config

    cfg = load_experiment_config()
    coarse = build_dvfs_matrix(cfg, "coarse")
    full = build_dvfs_matrix(cfg, "full")
    cpus = cfg["jetson"]["cpu_available_frequencies_hz"]
    gpus = cfg["jetson"]["gpu_available_frequencies_hz"]
    assert len(coarse) == 9 and len(full) == len(cpus) * len(gpus) == 84
    assert set(coarse) <= set(full)
    assert (422, 306) in full and (1984, 918) in full
    with pytest.raises(ValueError):
        build_dvfs_matrix(cfg, "medium")


def test_full_grid_labels_map_back_to_exact_opps():
    from power_monitor.controllers.dvfs_controller import DvfsController
    from power_monitor.experiments.interference import build_dvfs_matrix, load_experiment_config, validate_dvfs_point

    cfg = load_experiment_config()
    dc = DvfsController()
    dc.available_cpu_hz = [int(v) for v in cfg["jetson"]["cpu_available_frequencies_hz"]]
    dc.available_gpu_hz = [int(v) for v in cfg["jetson"]["gpu_available_frequencies_hz"]]
    for cpu, gpu in build_dvfs_matrix(cfg, "full"):
        cpu_khz, gpu_hz, errors = validate_dvfs_point(dc, cpu, gpu)
        assert errors == [] and cpu_khz in dc.available_cpu_hz and gpu_hz in dc.available_gpu_hz


def test_single_configuration_repeated_as_one_point_sweep(tmp_path):
    from power_monitor.experiments.interference import load_experiment_config, run_sweep

    cfg = load_experiment_config()
    cfg.update(output_root=str(tmp_path), warmup_seconds=0.1, measurement_seconds=0.6, repetitions=3, cooldown_seconds=0)
    results = run_sweep(cfg, workloads=["W1"], dry_run=True, points=[(1190, 612)])
    assert [r["repetition"] for r in results] == [1, 2, 3]
    assert {(r["cpu_freq_target_mhz"], r["gpu_freq_target_mhz"]) for r in results} == {(1190, 612)}
    assert all(r["sweep_id"] == results[0]["sweep_id"] for r in results)

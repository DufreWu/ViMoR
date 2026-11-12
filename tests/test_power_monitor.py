import pytest

from power_monitor.analysis.metrics import (
    deadline_miss_rate,
    llm_slowdown,
    perception_slowdown,
)
from power_monitor.controllers.dvfs_controller import DvfsController
from power_monitor.experiments.interference import validate_experiment_config
from power_monitor.monitors.power_monitor import JetsonPowerMonitor


def test_llm_and_perception_slowdown_calculations():
    assert llm_slowdown(1.5, 1.0) == pytest.approx(50.0)
    assert perception_slowdown(250.0, 200.0) == pytest.approx(25.0)
    assert deadline_miss_rate(5, 20) == pytest.approx(0.25)


def test_config_validation_allows_supported_flags():
    cfg = {
        "workloads": {"W1": ["yolo"], "W3": ["llm"], "W5": ["yolo", "llm"]},
        "apps": {"yolo": {"model_path": "x.pt", "target_fps": 10}},
        "llm": {"model_path": "x.gguf", "prompt_tokens": 128, "decode_tokens": 128, "request_schedule": {"type": "periodic"}},
        "dvfs": {"cpu_freq_mhz": [576, 1190, 1984], "gpu_freq_mhz": [306, 612, 918]},
    }
    assert validate_experiment_config(cfg)["valid"] is True
    cfg["workloads"]["W9"] = ["bert"]
    assert validate_experiment_config(cfg)["valid"] is False


def test_dvfs_controller_validates_frequency_lists():
    dc = DvfsController()
    dc.available_cpu_hz = [576000, 1190000, 1984000]
    dc.available_gpu_hz = [306000000, 612000000, 918000000]
    assert dc.is_supported_cpu_freq(1190000) is True
    assert dc.is_supported_gpu_freq(999000000) is False


def test_power_monitor_missing_sensor_raises_clear_error(tmp_path):
    missing = tmp_path / "missing"
    pm = JetsonPowerMonitor(sensor_dir=str(missing))
    with pytest.raises(FileNotFoundError):
        pm.sample()


def test_fps_monitor_tracks_multiple_apps_independently():
    from power_monitor.monitors.fps_monitor import FpsMonitor

    fps = FpsMonitor(window_s=1.0, step_s=0.5)
    cam = fps.register("camera0", "capture", target_fps=30)
    yolo = fps.register("yolo", "output", target_fps=10)
    for i in range(301):  # 10 s @ 30 fps
        cam.tick(ts=i / 30.0)
    for i in range(81):  # 10 s @ 8 fps, below target
        yolo.tick(ts=i / 8.0, frame_id=i)
    for k in range(20):
        yolo.drop(ts=5.0)
    fps.start(0.0)
    fps.stop(10.0)

    rows = {(r["app"], r["stage"]): r for r in fps.summary()}
    assert rows[("camera0", "capture")]["mean_fps"] == pytest.approx(30.0)
    assert rows[("yolo", "output")]["mean_fps"] == pytest.approx(8.0)
    assert rows[("yolo", "output")]["fps_attainment"] == pytest.approx(0.8)
    assert rows[("yolo", "output")]["drop_rate"] == pytest.approx(20 / 101)
    assert rows[("yolo", "output")]["window_fps_min"] == pytest.approx(8.0)
    assert fps.register("yolo", "output") is yolo
    flat = fps.flat_summary()
    assert flat["fps_yolo_output_mean_fps"] == pytest.approx(8.0)


def test_fps_monitor_current_fps_and_csv(tmp_path):
    from power_monitor.monitors.fps_monitor import FpsMonitor

    fps = FpsMonitor()
    for i in range(20):
        fps.tick("app_a", ts=100.0 + i * 0.1)
    assert fps.stream("app_a", "output").current_fps(window_s=1.0, now=101.9) == pytest.approx(10.0)
    fps.write_csv(tmp_path)
    for name in ("fps_events.csv", "fps_timeseries.csv", "fps_summary.csv"):
        assert (tmp_path / name).exists()


def test_default_config_defines_w0_to_w7():
    from power_monitor.experiments.interference import load_experiment_config

    cfg = load_experiment_config()
    assert validate_experiment_config(cfg)["valid"] is True
    combos = {frozenset(v) for v in cfg["workloads"].values()}
    assert len(combos) == 8  # empty set + 7 non-empty combinations of 3 apps


def test_request_schedule_arrivals():
    from power_monitor.apps.llm_app import RequestSchedule

    assert RequestSchedule(type="closed_loop").arrival_offsets(10, 0) is None
    assert RequestSchedule(type="periodic", interval_s=2.5).arrival_offsets(10, 0) == [0.0, 2.5, 5.0, 7.5]
    a = RequestSchedule(type="poisson", rate_hz=1.0).arrival_offsets(30, seed=7)
    assert a == RequestSchedule(type="poisson", rate_hz=1.0).arrival_offsets(30, seed=7)
    assert all(0 < t < 30 for t in a)


def test_resolve_frequencies_snaps_mhz_to_sysfs_units():
    from power_monitor.experiments.interference import resolve_frequencies

    dc = DvfsController()
    dc.available_cpu_hz = [576000, 1190400, 1984000]  # kHz
    dc.available_gpu_hz = [306000000, 612000000]  # Hz
    assert resolve_frequencies(dc, 1190, 612) == (1190400, 612000000)


def test_dry_run_w7_writes_all_outputs(tmp_path):
    import json

    from power_monitor.experiments.interference import load_experiment_config, run_workload

    cfg = load_experiment_config()
    cfg.update(output_root=str(tmp_path), warmup_seconds=0.5, measurement_seconds=2.0)
    result = run_workload("W7", None, None, repetition=1, config=cfg, dry_run=True)
    assert result["workload_apps"] == "yolo+segformer+llm"
    assert result["yolo_fps_mean"] == pytest.approx(10.0, rel=0.1)
    assert result["segformer_fps_mean"] == pytest.approx(5.0, rel=0.15)
    assert result["llm_tokens_per_s"] > 0
    run_dir = next((tmp_path / "runs").iterdir())
    for name in ("metadata.json", "summary.json", "power.csv", "fps_summary.csv", "perception_frames.csv", "llm_requests.csv", "llm_timeseries.csv"):
        assert (run_dir / name).exists(), name
    meta = json.loads((run_dir / "metadata.json").read_text())
    for key in ("run_id", "workload_config", "cpu_freq_target_mhz", "nvpmodel_mode", "llm_quantization", "input_resolution", "target_fps", "request_schedule", "seed", "repetition"):
        assert key in meta
    ts = (run_dir / "llm_timeseries.csv").read_text().splitlines()
    assert len(ts) - 1 == 20  # 2 s window / 100 ms
    assert (tmp_path / "runs_index.csv").exists()


def test_dry_run_w0_idle(tmp_path):
    from power_monitor.experiments.interference import load_experiment_config, run_workload

    cfg = load_experiment_config()
    cfg.update(output_root=str(tmp_path), warmup_seconds=0.2, measurement_seconds=1.0)
    result = run_workload("W0", None, None, config=cfg, dry_run=True)
    assert result["workload_apps"] == "idle" and result["power_samples"] >= 3

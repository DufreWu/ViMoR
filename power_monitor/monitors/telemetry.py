"""Hardware telemetry: tegrastats + sysfs, sampled on a fixed interval (default 200 ms).

Two sources run side by side, each stamped with ``time.monotonic()`` on arrival so
they join with ``power.csv`` and the app traces:

* **tegrastats** (``tegrastats.csv``): per-core CPU utilisation, GPU (GR3D) utilisation,
  RAM used, EMC utilisation/frequency, temperatures. EMC_FREQ and the GR3D clock are
  only printed when tegrastats runs as root (as it does during DVFS sweeps).
* **sysfs** (``telemetry.csv``): the *actual* clocks (``cpuinfo_cur_freq`` per CPU
  policy - hardware readback, root only - falling back to ``scaling_cur_freq``; GPU
  devfreq ``cur_freq``), GPU load, thermal zones, and throttling indicators:

  - ``thermal_cap_*``: cur_state of the ``cpufreq-cpu*`` / ``devfreq-*gpu`` thermal
    cooling devices (> 0 = the thermal framework is capping that clock);
  - ``throttle_alert_*``: cur_state of the ``*-throttle-alert`` cooling devices
    (> 0 = a soctherm hardware throttle trip is active);
  - ``oc*_event_cnt``: soctherm over-current (power) throttle event counters - a
    non-zero delta over the window means hardware power throttling happened;
  - ``ina_alarm_*``: INA3221 critical/warning current alarms.
"""

from __future__ import annotations

import csv
import glob
import os
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from power_monitor.analysis.metrics import percentile

GPU_DEVFREQ = "/sys/devices/platform/17000000.gpu/devfreq/17000000.gpu"
GPU_LOAD_CANDIDATES = ("/sys/devices/platform/17000000.gpu/load", "/sys/devices/platform/bus@0/17000000.gpu/load")
CPU_ROOT = "/sys/devices/system/cpu"


# ---------------------------------------------------------------------------
# tegrastats parsing
# ---------------------------------------------------------------------------
_RE_RAM = re.compile(r"\bRAM (\d+)/(\d+)MB")
_RE_SWAP = re.compile(r"\bSWAP (\d+)/(\d+)MB")
_RE_CPU = re.compile(r"\bCPU \[([^\]]*)\]")
_RE_EMC = re.compile(r"\bEMC_FREQ (\d+)%(?:@(\d+))?")
_RE_GR3D = re.compile(r"\bGR3D_FREQ (\d+)%(?:@\[?(\d+)(?:,\d+)*\]?)?")
_RE_TEMP = re.compile(r"\b([A-Za-z][\w]*)@(-?\d+(?:\.\d+)?)C\b")
_RE_RAIL = re.compile(r"\b(VDD_\w+|VIN_\w+) (\d+)mW/(\d+)mW")


def parse_tegrastats_line(line: str) -> Dict[str, Any]:
    """Parse one tegrastats line into flat numeric fields (missing fields are omitted)."""
    out: Dict[str, Any] = {}
    if m := _RE_RAM.search(line):
        out["ram_used_mb"], out["ram_total_mb"] = int(m.group(1)), int(m.group(2))
    if m := _RE_SWAP.search(line):
        out["swap_used_mb"] = int(m.group(1))
    if m := _RE_CPU.search(line):
        utils, freqs = [], []
        for i, entry in enumerate(m.group(1).split(",")):
            entry = entry.strip()
            cm = re.match(r"(\d+)%@(\d+)", entry)
            if cm:
                utils.append(int(cm.group(1)))
                freqs.append(int(cm.group(2)))
                out[f"cpu{i}_util_pct"] = int(cm.group(1))
                out[f"cpu{i}_freq_mhz"] = int(cm.group(2))
            else:  # "off"
                out[f"cpu{i}_util_pct"] = None
        if utils:
            out["cpu_util_pct"] = sum(utils) / len(utils)  # mean over online cores
            out["cpu_util_max_core_pct"] = max(utils)
            out["cpu_online_cores"] = len(utils)
    if m := _RE_EMC.search(line):
        out["emc_util_pct"] = int(m.group(1))
        if m.group(2):
            out["emc_freq_mhz"] = int(m.group(2))
    if m := _RE_GR3D.search(line):
        out["gpu_util_pct"] = int(m.group(1))
        if m.group(2):
            out["gpu_freq_mhz_tegrastats"] = int(m.group(2))
    for name, value in _RE_TEMP.findall(line):
        out[f"temp_{name.lower()}_c"] = float(value)
    for name, inst, _avg in _RE_RAIL.findall(line):
        out[f"{name.lower()}_mw_tegrastats"] = int(inst)
    return out


class TegrastatsReader:
    """Runs ``tegrastats --interval N`` and timestamps each line on arrival."""

    def __init__(self, interval_s: float = 0.2):
        self.interval_s = float(interval_s)
        self.available = shutil.which("tegrastats") is not None
        self.samples: List[Dict[str, Any]] = []
        self._proc: Optional[subprocess.Popen[str]] = None
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    def start(self) -> None:
        if not self.available:
            return
        self._proc = subprocess.Popen(
            ["tegrastats", "--interval", str(max(1, int(round(self.interval_s * 1000))))],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1,
        )

        def _loop() -> None:
            assert self._proc is not None and self._proc.stdout is not None
            for line in self._proc.stdout:
                stamp = time.monotonic()
                parsed = parse_tegrastats_line(line)
                if parsed:
                    with self._lock:
                        self.samples.append({"monotonic": stamp, **parsed})

        self._thread = threading.Thread(target=_loop, name="tegrastats-reader", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._proc is not None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self._proc.kill()
            self._proc = None
        if self._thread is not None:
            self._thread.join(timeout=3)
            self._thread = None

    def window(self, start: float, end: float) -> List[Dict[str, Any]]:
        with self._lock:
            return [s for s in self.samples if start <= s["monotonic"] <= end]


# ---------------------------------------------------------------------------
# sysfs sampling
# ---------------------------------------------------------------------------
def _read(path: str) -> Optional[str]:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return handle.read().strip()
    except OSError:
        return None


def _read_int(path: str) -> Optional[int]:
    text = _read(path)
    try:
        return int(text) if text is not None else None
    except ValueError:
        return None


class SysfsSnapshot:
    """Discovers the relevant sysfs files once, then reads them cheaply on every sample."""

    def __init__(self, gpu_devfreq: str = GPU_DEVFREQ):
        self.gpu_devfreq = gpu_devfreq
        # One cpufreq policy per cluster (policy0 = cpu0-3, policy4 = cpu4-7 on Orin NX).
        self.cpu_policies: Dict[str, str] = {}
        for policy in sorted(glob.glob(f"{CPU_ROOT}/cpufreq/policy*")):
            hw = os.path.join(policy, "cpuinfo_cur_freq")
            sw = os.path.join(policy, "scaling_cur_freq")
            self.cpu_policies[os.path.basename(policy)] = hw if _read_int(hw) is not None else sw
        self.cpu_freq_source = "cpuinfo_cur_freq" if all(p.endswith("cpuinfo_cur_freq") for p in self.cpu_policies.values()) and self.cpu_policies else "scaling_cur_freq"
        self.gpu_load_path = next((p for p in GPU_LOAD_CANDIDATES if os.path.exists(p)), None)
        self.thermal_zones: Dict[str, str] = {}
        for zone in sorted(glob.glob("/sys/class/thermal/thermal_zone*")):
            ztype = _read(os.path.join(zone, "type"))
            if ztype:
                self.thermal_zones[ztype.replace("-thermal", "").lower()] = os.path.join(zone, "temp")
        self.cooling: Dict[str, str] = {}
        for dev in sorted(glob.glob("/sys/class/thermal/cooling_device*")):
            ctype = _read(os.path.join(dev, "type")) or ""
            if ctype.startswith("cpufreq-") or (ctype.startswith("devfreq-") and "gpu" in ctype):
                self.cooling["thermal_cap_" + ctype.split("-", 1)[1].replace(".", "_")] = os.path.join(dev, "cur_state")
            elif ctype.endswith("-throttle-alert"):
                self.cooling["throttle_alert_" + ctype[: -len("-throttle-alert")].replace("-", "_")] = os.path.join(dev, "cur_state")
        self.oc_counters: Dict[str, str] = {}
        for hw in glob.glob("/sys/devices/platform/soctherm-oc-event/hwmon/hwmon*"):
            for counter in sorted(glob.glob(os.path.join(hw, "oc*_event_cnt"))):
                self.oc_counters[os.path.basename(counter)] = counter
        self.ina_alarms: Dict[str, str] = {}
        for hw in glob.glob("/sys/bus/i2c/drivers/ina3221/*/hwmon/hwmon*"):
            for alarm in sorted(glob.glob(os.path.join(hw, "curr*_crit_alarm")) + glob.glob(os.path.join(hw, "curr*_max_alarm"))):
                self.ina_alarms["ina_alarm_" + os.path.basename(alarm).replace("_alarm", "")] = alarm

    def read(self) -> Dict[str, Any]:
        sample: Dict[str, Any] = {"monotonic": time.monotonic()}
        cpu_mhz = []
        for name, path in self.cpu_policies.items():
            khz = _read_int(path)
            sample[f"cpu_{name}_freq_mhz"] = khz / 1000.0 if khz is not None else None
            if khz is not None:
                cpu_mhz.append(khz / 1000.0)
        sample["cpu_freq_actual_min_mhz"] = min(cpu_mhz) if cpu_mhz else None
        sample["cpu_freq_actual_max_mhz"] = max(cpu_mhz) if cpu_mhz else None
        sample["cpu_freq_actual_avg_mhz"] = sum(cpu_mhz) / len(cpu_mhz) if cpu_mhz else None
        gpu_hz = _read_int(os.path.join(self.gpu_devfreq, "cur_freq"))
        sample["gpu_freq_actual_mhz"] = gpu_hz / 1e6 if gpu_hz is not None else None
        if self.gpu_load_path:
            load = _read_int(self.gpu_load_path)  # 0..1000
            sample["gpu_load_pct_sysfs"] = load / 10.0 if load is not None else None
        for name, path in self.thermal_zones.items():
            milli = _read_int(path)
            sample[f"temp_{name}_c"] = milli / 1000.0 if milli is not None else None
        for name, path in {**self.cooling, **self.oc_counters, **self.ina_alarms}.items():
            sample[name] = _read_int(path)
        return sample


class HardwareTelemetry:
    """tegrastats + sysfs sampler with a single start/stop and a windowed summary."""

    def __init__(self, interval_s: float = 0.2, use_tegrastats: bool = True):
        self.interval_s = float(interval_s)
        self.sysfs = SysfsSnapshot()
        self.tegrastats = TegrastatsReader(interval_s) if use_tegrastats else None
        self.samples: List[Dict[str, Any]] = []
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._lock = threading.Lock()

    def snapshot(self) -> Dict[str, Any]:
        """One immediate sysfs reading (used for the pre/post-run hardware state)."""
        return self.sysfs.read()

    def start(self) -> None:
        if self.tegrastats is not None:
            self.tegrastats.start()
        self._stop.clear()

        def _loop() -> None:
            next_t = time.monotonic()
            while not self._stop.is_set():
                sample = self.sysfs.read()
                with self._lock:
                    self.samples.append(sample)
                next_t += self.interval_s
                delay = next_t - time.monotonic()
                if delay > 0 and self._stop.wait(delay):
                    break

        self._thread = threading.Thread(target=_loop, name="sysfs-telemetry", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self.tegrastats is not None:
            self.tegrastats.stop()

    def window(self, start: float, end: float) -> List[Dict[str, Any]]:
        with self._lock:
            return [s for s in self.samples if start <= s["monotonic"] <= end]

    def tegrastats_window(self, start: float, end: float) -> List[Dict[str, Any]]:
        return self.tegrastats.window(start, end) if self.tegrastats is not None else []

    # ----- summary ---------------------------------------------------------
    def summary(self, start: float, end: float, cpu_target_mhz: Optional[float] = None, gpu_target_mhz: Optional[float] = None,
                freq_tolerance_mhz: float = 1.0) -> Dict[str, Any]:
        sys_rows = self.window(start, end)
        teg_rows = self.tegrastats_window(start, end)
        out: Dict[str, Any] = {
            "telemetry_samples": len(sys_rows),
            "tegrastats_samples": len(teg_rows),
            "cpu_freq_source": self.sysfs.cpu_freq_source,
        }
        out.update(_stats(teg_rows, "cpu_util_pct", "cpu_util_pct"))
        out.update(_stats(teg_rows, "gpu_util_pct", "gpu_util_pct"))
        out.update(_stats(sys_rows, "gpu_load_pct_sysfs", "gpu_load_pct_sysfs"))
        out.update(_stats(teg_rows, "ram_used_mb", "ram_used_mb", ("mean", "max")))
        out.update(_stats(teg_rows, "emc_util_pct", "emc_util_pct"))
        out.update(_stats(teg_rows, "emc_freq_mhz", "emc_freq_mhz", ("mean", "min", "max")))
        out.update(_stats(sys_rows, "cpu_freq_actual_avg_mhz", "cpu_freq_actual", ("mean", "p5", "p95")))
        out.update(_stats(sys_rows, "cpu_freq_actual_min_mhz", "cpu_freq_actual", ("min",)))
        out.update(_stats(sys_rows, "cpu_freq_actual_max_mhz", "cpu_freq_actual", ("max",)))
        out.update(_stats(sys_rows, "gpu_freq_actual_mhz", "gpu_freq_actual", ("mean", "p5", "p95", "min", "max")))
        # Time share per frequency (samples are evenly spaced): {MHz: fraction}. Mostly useful
        # with dynamic governors; with a pinned clock it is {target: 1.0}.
        policy_keys = [f"cpu_{name}_freq_mhz" for name in self.sysfs.cpu_policies]
        out["cpu_freq_residency"] = _residency([r.get(k) for r in sys_rows for k in policy_keys])
        out["gpu_freq_residency"] = _residency([r.get("gpu_freq_actual_mhz") for r in sys_rows])
        for zone in ("cpu", "gpu", "tj", "soc0"):
            out.update(_stats(sys_rows, f"temp_{zone}_c", f"temp_{zone}_c", ("mean", "max")))

        # Frequency lock: share of samples where every CPU policy / the GPU reports the target.
        if cpu_target_mhz is not None:
            ok = [r for r in sys_rows if r.get("cpu_freq_actual_min_mhz") is not None
                  and abs(r["cpu_freq_actual_min_mhz"] - cpu_target_mhz) <= freq_tolerance_mhz
                  and abs(r["cpu_freq_actual_max_mhz"] - cpu_target_mhz) <= freq_tolerance_mhz]
            out["cpu_freq_lock_rate"] = len(ok) / len(sys_rows) if sys_rows else None
        if gpu_target_mhz is not None:
            ok = [r for r in sys_rows if r.get("gpu_freq_actual_mhz") is not None and abs(r["gpu_freq_actual_mhz"] - gpu_target_mhz) <= freq_tolerance_mhz]
            out["gpu_freq_lock_rate"] = len(ok) / len(sys_rows) if sys_rows else None

        # Throttling indicators.
        cap_keys = [k for k in self.sysfs.cooling if k.startswith("thermal_cap_")]
        alert_keys = [k for k in self.sysfs.cooling if k.startswith("throttle_alert_")]
        out["thermal_cap_max_state"] = max((r.get(k) or 0 for r in sys_rows for k in cap_keys), default=None)
        out["throttle_alert_max_state"] = max((r.get(k) or 0 for r in sys_rows for k in alert_keys), default=None)
        out["throttle_alert_active"] = sorted({k for r in sys_rows for k in alert_keys if (r.get(k) or 0) > 0})
        oc_delta = 0
        for key in self.sysfs.oc_counters:
            values = [r[key] for r in sys_rows if r.get(key) is not None]
            if len(values) >= 2:
                oc_delta += max(0, values[-1] - values[0])
        out["oc_throttle_events"] = oc_delta if self.sysfs.oc_counters else None
        out["ina_alarm_samples"] = sum(1 for r in sys_rows for k in self.sysfs.ina_alarms if (r.get(k) or 0) > 0) if self.sysfs.ina_alarms else None
        out["thermal_throttled"] = bool((out["thermal_cap_max_state"] or 0) > 0 or (out["throttle_alert_max_state"] or 0) > 0)
        out["power_throttled"] = bool((out["oc_throttle_events"] or 0) > 0 or (out["ina_alarm_samples"] or 0) > 0)
        return out

    # ----- output ------------------------------------------------------------
    def write_csv(self, run_dir: str | os.PathLike[str], start: Optional[float] = None, end: Optional[float] = None) -> None:
        out = Path(run_dir)
        out.mkdir(parents=True, exist_ok=True)
        flag = (lambda t: start <= t <= end) if start is not None and end is not None else (lambda t: True)
        with self._lock:
            sys_rows = [{**r, "in_window": flag(r["monotonic"])} for r in self.samples]
        _write_rows(out / "telemetry.csv", sys_rows)
        if self.tegrastats is not None:
            with self.tegrastats._lock:
                teg_rows = [{**r, "in_window": flag(r["monotonic"])} for r in self.tegrastats.samples]
            _write_rows(out / "tegrastats.csv", teg_rows)


def _stats(rows: Iterable[Dict[str, Any]], key: str, prefix: str, which: Iterable[str] = ("mean", "p95", "max")) -> Dict[str, Any]:
    values = [float(r[key]) for r in rows if r.get(key) is not None]
    result: Dict[str, Any] = {}
    for stat in which:
        name = f"{prefix}_{stat}"
        if not values:
            result[name] = None
        elif stat == "mean":
            result[name] = sum(values) / len(values)
        elif stat == "max":
            result[name] = max(values)
        elif stat == "min":
            result[name] = min(values)
        elif stat.startswith("p"):
            result[name] = percentile(values, float(stat[1:]))
    return result


def _residency(values: Iterable[Optional[float]]) -> Dict[str, float]:
    counts: Dict[str, int] = {}
    for v in values:
        if v is not None:
            key = f"{v:.1f}".rstrip("0").rstrip(".")
            counts[key] = counts.get(key, 0) + 1
    total = sum(counts.values())
    return {k: round(c / total, 4) for k, c in sorted(counts.items(), key=lambda kv: float(kv[0]))} if total else {}


def _write_rows(path: Path, rows: List[Dict[str, Any]]) -> None:
    fieldnames = list(dict.fromkeys(k for row in rows for k in row)) or ["monotonic"]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

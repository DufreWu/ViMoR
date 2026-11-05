from __future__ import annotations

import csv
import glob
import os
import threading
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional

INA3221_GLOB = "/sys/bus/i2c/drivers/ina3221/*/hwmon/hwmon*"


def find_ina3221_dir() -> Optional[str]:
    """The hwmon index (hwmon2/hwmon3/...) is not stable across boots, so search for it."""
    for candidate in sorted(glob.glob(INA3221_GLOB)):
        if os.path.exists(os.path.join(candidate, "in1_input")):
            return candidate
    return None


class JetsonPowerMonitor:
    """Read the INA3221 rails of a Jetson module and integrate input power over time.

    ``sample()`` reads every labelled rail (``VDD_IN``, ``VDD_CPU_GPU_CV``, ``VDD_SOC`` on
    Orin NX). ``power_w`` is the module input power (rail 1, ``VDD_IN``).
    ``start(interval_s)`` / ``stop()`` sample periodically on a background thread.
    """

    def __init__(self, sensor_dir: Optional[str] = None):
        if sensor_dir in (None, "", "auto"):
            sensor_dir = find_ina3221_dir() or "/sys/bus/i2c/drivers/ina3221/1-0040/hwmon/hwmon2"
        self.sensor_dir = sensor_dir
        self.samples: List[Dict[str, float]] = []
        self._rails: Optional[List[tuple]] = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._lock = threading.Lock()

    def _read_value(self, filename: str) -> float:
        full_path = Path(self.sensor_dir) / filename
        if not full_path.exists():
            raise FileNotFoundError(f"Power sensor file not found: {full_path}")
        with open(full_path, "r", encoding="utf-8") as handle:
            raw = handle.read().strip()
        try:
            return float(raw) / 1000.0
        except ValueError as exc:
            raise ValueError(f"Unable to parse sensor value from {full_path}: {raw!r}") from exc

    def rails(self) -> List[tuple]:
        """``[(index, column_name)]`` for each labelled rail, e.g. ``(1, "vdd_in")``."""
        if self._rails is None:
            rails = []
            for index in range(1, 4):
                label_path = Path(self.sensor_dir) / f"in{index}_label"
                if not (Path(self.sensor_dir) / f"curr{index}_input").exists():
                    continue
                label = label_path.read_text(encoding="utf-8").strip() if label_path.exists() else f"rail{index}"
                rails.append((index, label.lower()))
            self._rails = rails or [(1, "vdd_in")]
        return self._rails

    def sample(self) -> Dict[str, float]:
        if not os.path.isdir(self.sensor_dir):
            raise FileNotFoundError(f"Power sensor directory not found: {self.sensor_dir}")
        stamp = time.monotonic()
        sample: Dict[str, float] = {"timestamp": time.time(), "monotonic": stamp}
        for index, name in self.rails():
            sample[f"{name}_w"] = self._read_value(f"in{index}_input") * self._read_value(f"curr{index}_input")
        sample["power_w"] = sample[f"{self.rails()[0][1]}_w"]
        with self._lock:
            self.samples.append(sample)
        return sample

    # ----- periodic sampling ------------------------------------------------
    def start(self, interval_s: float = 0.1) -> None:
        self.sample()  # fail fast if the sensor is missing
        self._stop.clear()

        def _loop() -> None:
            next_t = time.monotonic() + interval_s
            while not self._stop.is_set():
                delay = next_t - time.monotonic()
                if delay > 0 and self._stop.wait(delay):
                    break
                try:
                    self.sample()
                except (OSError, ValueError):
                    pass
                next_t += interval_s

        self._thread = threading.Thread(target=_loop, name="power-sampler", daemon=True)
        self._thread.start()

    def stop(self) -> List[Dict[str, float]]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        return self.samples

    def window(self, start: float, end: float) -> List[Dict[str, float]]:
        with self._lock:
            return [s for s in self.samples if start <= s["monotonic"] <= end]

    def integrate_energy(self, samples: Optional[Iterable[Dict[str, float]]] = None, key: str = "power_w") -> float:
        seq = list(samples) if samples is not None else list(self.samples)
        if len(seq) < 2:
            return 0.0
        energy = 0.0
        for previous, current in zip(seq, seq[1:]):
            dt = max(0.0, float(current["monotonic"]) - float(previous["monotonic"]))
            avg_power = 0.5 * (float(previous.get(key, 0.0)) + float(current.get(key, 0.0)))
            energy += avg_power * dt
        return energy

    def reset(self) -> None:
        with self._lock:
            self.samples = []

    def write_csv(self, path: str | os.PathLike[str], samples: Optional[List[Dict[str, float]]] = None) -> None:
        rows = list(samples) if samples is not None else list(self.samples)
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = ["timestamp", "monotonic", "power_w"] + [f"{name}_w" for _, name in self.rails()] if rows else ["timestamp", "monotonic", "power_w"]
        with open(output_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    def check_environment(self) -> Dict[str, object]:
        if not os.path.isdir(self.sensor_dir):
            return {"valid": False, "issues": [f"Power sensor not found: {self.sensor_dir}"]}
        return {"valid": True, "issues": []}


class SyntheticPowerMonitor(JetsonPowerMonitor):
    """Dry-run stand-in with the same interface (constant 10 W)."""

    def __init__(self) -> None:
        super().__init__(sensor_dir="/nonexistent")
        self._rails = [(1, "vdd_in")]

    def sample(self) -> Dict[str, float]:
        sample = {"timestamp": time.time(), "monotonic": time.monotonic(), "vdd_in_w": 10.0, "power_w": 10.0}
        with self._lock:
            self.samples.append(sample)
        return sample

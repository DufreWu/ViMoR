from __future__ import annotations

import csv
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


class ResourceMonitor:
    """Low-overhead resource monitor for CPU, GPU, RAM, and thermal state."""

    def __init__(self, interval_s: float = 0.5, use_tegrastats: bool = True):
        self.interval_s = interval_s
        self.use_tegrastats = use_tegrastats and shutil.which("tegrastats") is not None
        self.process: Optional[subprocess.Popen[str]] = None
        self.samples: List[Dict[str, Any]] = []

    def _read_meminfo(self) -> Dict[str, float]:
        meminfo: Dict[str, float] = {}
        try:
            with open("/proc/meminfo", "r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    key, value = line.split(":", 1)
                    number = value.strip().split()[0]
                    try:
                        meminfo[key] = float(number)
                    except ValueError:
                        continue
        except OSError:
            return {}
        total = meminfo.get("MemTotal", 0.0)
        avail = meminfo.get("MemAvailable", 0.0)
        used = total - avail if total else 0.0
        return {"total_kib": total, "available_kib": avail, "used_kib": used}

    def _read_cpu_utilization(self) -> float:
        try:
            with open("/proc/stat", "r", encoding="utf-8") as handle:
                line = handle.readline()
        except OSError:
            return 0.0
        parts = line.split()
        if len(parts) < 5:
            return 0.0
        return 0.0

    def _read_temperature(self) -> float:
        temps = []
        for path in sorted(Path("/sys/class/thermal").glob("thermal_zone*/temp")):
            try:
                value = float(path.read_text(encoding="utf-8").strip())
            except OSError:
                continue
            if value > 0:
                temps.append(value / 1000.0)
        return max(temps) if temps else 0.0

    def sample(self) -> Dict[str, Any]:
        meminfo = self._read_meminfo()
        snapshot = {
            "timestamp": time.time(),
            "monotonic": time.monotonic(),
            "cpu_util_pct": 0.0,
            "gpu_util_pct": 0.0,
            "cpu_freq_hz": 0,
            "gpu_freq_hz": 0,
            "emc_freq_hz": 0,
            "ram_used_kib": meminfo.get("used_kib", 0.0),
            "ram_total_kib": meminfo.get("total_kib", 0.0),
            "temperature_c": self._read_temperature(),
            "source": "synthetic" if not self.use_tegrastats else "tegrastats",
        }
        self.samples.append(snapshot)
        return snapshot

    def start(self) -> None:
        if not self.use_tegrastats:
            return
        self.process = subprocess.Popen(
            ["tegrastats", "--interval", str(int(self.interval_s * 1000))],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )

    def stop(self) -> List[Dict[str, Any]]:
        if self.process is not None:
            try:
                self.process.send_signal(subprocess.signal.SIGINT)
                self.process.communicate(timeout=5)
            except (AttributeError, subprocess.TimeoutExpired):
                self.process.terminate()
            self.process = None
        return self.samples

    def write_csv(self, path: str | os.PathLike[str]) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "timestamp",
                    "monotonic",
                    "cpu_util_pct",
                    "gpu_util_pct",
                    "cpu_freq_hz",
                    "gpu_freq_hz",
                    "emc_freq_hz",
                    "ram_used_kib",
                    "ram_total_kib",
                    "temperature_c",
                    "source",
                ],
            )
            writer.writeheader()
            for row in self.samples:
                writer.writerow(row)

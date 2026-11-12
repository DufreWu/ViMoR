from __future__ import annotations

import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


class DvfsController:
    """Small Jetson DVFS wrapper that supports validation and restoration."""

    def __init__(self, config_path: Optional[str | os.PathLike[str]] = None):
        self.config_path = Path(config_path) if config_path else None
        self.cpu_path = "/sys/devices/system/cpu"
        self.gpu_path = "/sys/devices/platform/17000000.gpu/devfreq/17000000.gpu"
        self.available_cpu_hz: List[int] = []
        self.available_gpu_hz: List[int] = []
        self.original_state: Dict[str, Any] = {}
        self._refresh_supported_frequencies()

    def _read_text(self, path: str | os.PathLike[str]) -> str:
        try:
            with open(path, "r", encoding="utf-8") as handle:
                return handle.read().strip()
        except OSError:
            return ""

    def _write_text(self, path: str | os.PathLike[str], value: str) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(value)

    def _refresh_supported_frequencies(self) -> None:
        self.available_cpu_hz = self.query_available_cpu_freqs()
        self.available_gpu_hz = self.query_available_gpu_freqs()

    def query_available_cpu_freqs(self) -> List[int]:
        base = Path(self.cpu_path)
        if not base.exists():
            return []
        candidate = base / "cpu0" / "cpufreq" / "scaling_available_frequencies"
        if candidate.exists():
            text = self._read_text(candidate)
            values = [int(v) for v in text.split() if v.strip()]
            return sorted(set(values))
        return []

    def query_available_gpu_freqs(self) -> List[int]:
        path = Path(self.gpu_path)
        if not path.exists():
            return []
        for name in ["available_frequencies", "max_freq", "min_freq"]:
            candidate = path / name
            if candidate.exists():
                text = self._read_text(candidate)
                if not text:
                    continue
                values = []
                for part in text.replace("\n", " ").split():
                    try:
                        values.append(int(part))
                    except ValueError:
                        continue
                if values:
                    return sorted(set(values))
        return []

    def is_supported_cpu_freq(self, freq_hz: int) -> bool:
        return freq_hz in self.available_cpu_hz if self.available_cpu_hz else True

    def is_supported_gpu_freq(self, freq_hz: int) -> bool:
        return freq_hz in self.available_gpu_hz if self.available_gpu_hz else True

    def validate_frequency_pair(self, cpu_hz: int, gpu_hz: int) -> None:
        if not self.is_supported_cpu_freq(cpu_hz):
            raise ValueError(f"Unsupported CPU frequency requested: {cpu_hz} Hz")
        if not self.is_supported_gpu_freq(gpu_hz):
            raise ValueError(f"Unsupported GPU frequency requested: {gpu_hz} Hz")

    def save_original_state(self) -> Dict[str, Any]:
        state: Dict[str, Any] = {"cpu": {}, "gpu": {}}
        cpu_root = Path(self.cpu_path)
        cpu_count = os.cpu_count() or 1
        for cpu_index in range(cpu_count):
            cpu_dir = cpu_root / f"cpu{cpu_index}" / "cpufreq"
            if not cpu_dir.exists():
                continue
            governor = cpu_dir / "scaling_governor"
            state["cpu"][cpu_index] = {
                "governor": self._read_text(governor),
                "min": self._read_text(cpu_dir / "scaling_min_freq"),
                "max": self._read_text(cpu_dir / "scaling_max_freq"),
                "current": self._read_text(cpu_dir / "scaling_cur_freq"),
            }
        gpu_dir = Path(self.gpu_path)
        if gpu_dir.exists():
            state["gpu"] = {
                "governor": self._read_text(gpu_dir / "governor"),
                "min": self._read_text(gpu_dir / "min_freq"),
                "max": self._read_text(gpu_dir / "max_freq"),
                "current": self._read_text(gpu_dir / "cur_freq"),
            }
        self.original_state = state
        return state

    def restore_original_state(self) -> Dict[str, Any]:
        if not self.original_state:
            self.save_original_state()
        for cpu_index, values in self.original_state.get("cpu", {}).items():
            cpu_dir = Path(self.cpu_path) / f"cpu{cpu_index}" / "cpufreq"
            if not cpu_dir.exists():
                continue
            try:
                if values.get("governor"):
                    self._write_text(cpu_dir / "scaling_governor", values["governor"])
                if values.get("min") and values.get("max"):
                    self._set_cpu_range(cpu_dir, int(values["min"]), int(values["max"]))
                if values.get("governor") == "userspace" and values.get("current"):
                    self._write_text(cpu_dir / "scaling_setspeed", str(values["current"]))
            except (OSError, ValueError):
                pass
        gpu_values = self.original_state.get("gpu", {})
        gpu_dir = Path(self.gpu_path)
        if gpu_dir.exists():
            try:
                if gpu_values.get("governor"):
                    self._write_text(gpu_dir / "governor", gpu_values["governor"])
                if gpu_values.get("min") and gpu_values.get("max"):
                    userspace = gpu_values.get("governor") == "userspace"
                    target = int(gpu_values.get("current") or gpu_values["max"]) if userspace else None
                    self._set_gpu_range(int(gpu_values["min"]), int(gpu_values["max"]), target)
            except (OSError, ValueError):
                pass
        return self.original_state

    def _set_cpu_range(self, core_dir: Path, min_khz: int, max_khz: int) -> None:
        """Order-independent cpufreq limits: widen to the hardware range, then narrow."""
        hw_min = self._read_text(core_dir / "cpuinfo_min_freq") or str(min_khz)
        hw_max = self._read_text(core_dir / "cpuinfo_max_freq") or str(max_khz)
        self._write_text(core_dir / "scaling_max_freq", hw_max)
        self._write_text(core_dir / "scaling_min_freq", hw_min)
        self._write_text(core_dir / "scaling_max_freq", str(int(max_khz)))
        self._write_text(core_dir / "scaling_min_freq", str(int(min_khz)))

    # ----- default (dynamic) governors ------------------------------------
    def read_governors(self) -> Dict[str, Any]:
        cpus = sorted({g for g in (self._read_text(p) for p in Path(self.cpu_path).glob("cpu[0-9]*/cpufreq/scaling_governor")) if g})
        return {"cpu_governor": cpus, "gpu_governor": self._read_text(Path(self.gpu_path) / "governor") or None}

    def set_default_governors(self, cpu_governor: str = "schedutil", gpu_governor: str = "nvhost_podgov") -> Dict[str, Any]:
        """Stock dynamic DVFS: the given governors with the full hardware frequency range."""
        cpu_root = Path(self.cpu_path)
        for core in range(os.cpu_count() or 1):
            core_dir = cpu_root / f"cpu{core}" / "cpufreq"
            if not core_dir.exists():
                continue
            available = (self._read_text(core_dir / "scaling_available_governors") or "").split()
            if available and cpu_governor not in available:
                raise ValueError(f"CPU governor {cpu_governor!r} not available (have {available})")
            self._write_text(core_dir / "scaling_governor", cpu_governor)
            hw_min, hw_max = self._read_text(core_dir / "cpuinfo_min_freq"), self._read_text(core_dir / "cpuinfo_max_freq")
            if hw_min and hw_max:
                self._set_cpu_range(core_dir, int(hw_min), int(hw_max))
        gpu_dir = Path(self.gpu_path)
        if gpu_dir.exists():
            available = (self._read_text(gpu_dir / "available_governors") or "").split()
            if available and gpu_governor not in available:
                raise ValueError(f"GPU governor {gpu_governor!r} not available (have {available})")
            self._write_text(gpu_dir / "governor", gpu_governor)
            if self.available_gpu_hz:
                self._set_gpu_range(min(self.available_gpu_hz), max(self.available_gpu_hz))
        state = self.read_governors()
        if state["cpu_governor"] and state["cpu_governor"] != [cpu_governor]:
            raise RuntimeError(f"CPU governor not applied: {state['cpu_governor']}")
        if state["gpu_governor"] and state["gpu_governor"] != gpu_governor:
            raise RuntimeError(f"GPU governor not applied: {state['gpu_governor']}")
        return state

    def set_cpu_frequency(self, freq_hz: int, cpu_index: Optional[int] = None) -> Dict[str, Any]:
        cpu_root = Path(self.cpu_path)
        if not cpu_root.exists():
            return {"cpu_hz": int(freq_hz), "cores": [], "dry_run": True}
        if not self.is_supported_cpu_freq(freq_hz):
            raise ValueError(f"Unsupported CPU frequency requested: {freq_hz} kHz")
        target_cores = [cpu_index] if cpu_index is not None else range(os.cpu_count() or 1)
        for core in target_cores:
            core_dir = cpu_root / f"cpu{core}" / "cpufreq"
            if not core_dir.exists():
                continue
            self._write_text(core_dir / "scaling_governor", "userspace")
            self._set_cpu_range(core_dir, int(freq_hz), int(freq_hz))
            self._write_text(core_dir / "scaling_setspeed", str(int(freq_hz)))
        return {"cpu_hz": int(freq_hz), "cores": list(target_cores)}

    def _set_gpu_range(self, min_hz: int, max_hz: int, set_freq_hz: Optional[int] = None) -> None:
        """Order-independent devfreq update.

        Writing ``min_freq`` above the current ``max_freq`` (or ``max_freq`` below the
        current ``min_freq``) leaves the GPU at its old clock on Jetson, so a naive
        ``min=f; max=f`` only works when lowering the clock. Instead: widen to the full
        OPP range, set the userspace governor's target, then narrow to the requested range.
        """
        gpu_dir = Path(self.gpu_path)
        lo = min(self.available_gpu_hz) if self.available_gpu_hz else min_hz
        hi = max(self.available_gpu_hz) if self.available_gpu_hz else max_hz
        self._write_text(gpu_dir / "max_freq", str(int(hi)))
        self._write_text(gpu_dir / "min_freq", str(int(lo)))
        set_freq = gpu_dir / "userspace" / "set_freq"
        if set_freq_hz is not None and set_freq.exists():
            self._write_text(set_freq, str(int(set_freq_hz)))
        self._write_text(gpu_dir / "max_freq", str(int(max_hz)))
        self._write_text(gpu_dir / "min_freq", str(int(min_hz)))

    def set_gpu_frequency(self, freq_hz: int, settle_timeout_s: float = 3.0, retries: int = 3) -> Dict[str, Any]:
        """Pin the GPU to ``freq_hz`` and wait until ``cur_freq`` reads it back (re-applying if needed)."""
        gpu_dir = Path(self.gpu_path)
        if not gpu_dir.exists():
            return {"gpu_hz": int(freq_hz), "dry_run": True}
        if not self.is_supported_gpu_freq(freq_hz):
            raise ValueError(f"Unsupported GPU frequency requested: {freq_hz} Hz")
        self._write_text(gpu_dir / "governor", "userspace")
        attempts = 0
        for attempts in range(1, retries + 1):
            self._set_gpu_range(int(freq_hz), int(freq_hz), int(freq_hz))
            deadline = time.monotonic() + settle_timeout_s / retries
            while time.monotonic() < deadline:
                if self.read_gpu_frequency() == int(freq_hz):
                    return {"gpu_hz": int(freq_hz), "attempts": attempts}
                time.sleep(0.05)
        return {"gpu_hz": int(freq_hz), "attempts": attempts, "locked": False}

    def read_cpu_frequency(self, cpu_index: int = 0) -> Optional[int]:
        path = Path(self.cpu_path) / f"cpu{cpu_index}" / "cpufreq" / "scaling_cur_freq"
        if not path.exists():
            return None
        value = self._read_text(path)
        try:
            return int(value)
        except ValueError:
            return None

    def read_gpu_frequency(self) -> Optional[int]:
        gpu_dir = Path(self.gpu_path)
        if not gpu_dir.exists():
            return None
        for name in ["cur_freq", "freq"]:
            candidate = gpu_dir / name
            if candidate.exists():
                value = self._read_text(candidate)
                try:
                    return int(value)
                except ValueError:
                    continue
        return None

    def read_actual_frequencies(self) -> Dict[str, Any]:
        cpu_values = []
        for core in range(max(1, os.cpu_count() or 1)):
            value = self.read_cpu_frequency(core)
            if value is not None:
                cpu_values.append(value)
        return {
            "cpu_hz": cpu_values,
            "gpu_hz": self.read_gpu_frequency(),
            "timestamp": time.time(),
        }

    def check_environment(self) -> Dict[str, Any]:
        issues: List[str] = []
        warnings: List[str] = []
        if not os.path.exists(self.cpu_path):
            issues.append("CPU frequency sysfs path not found")
        if not os.path.exists(self.gpu_path):
            issues.append("GPU frequency sysfs path not found")
        if not shutil.which("tegrastats"):
            warnings.append("tegrastats not installed; resource monitoring will be synthetic.")
        if (not self.available_cpu_hz) or (not self.available_gpu_hz):
            warnings.append("No CPU/GPU frequency lists discovered; validation is conservative.")
        return {"valid": not issues, "issues": issues, "warnings": warnings}

"""Multi-application FPS (event-rate) monitor.

Design
------
Every rate we care about is a *stream* identified by ``(app, stage)``:

    app   - which application produced the event, e.g. ``"yolo"``, ``"segformer"``,
            ``"camera0"``, ``"llm"``.
    stage - which point in that app's pipeline, e.g. ``"capture"``, ``"output"``,
            ``"token"``.

Producers obtain a lightweight handle once and call ``tick()`` per completed frame::

    fps = FpsMonitor()
    cam = fps.register("camera0", "capture", target_fps=30)
    det = fps.register("yolo", "output", target_fps=10)
    ...
    det.tick()            # one processed frame, timestamped with time.monotonic()
    det.drop()            # a scheduled frame that was skipped

All methods are thread-safe, so several apps (threads) can report into one monitor.
Timestamps may also be passed explicitly (``tick(ts=...)``), which lets synthetic /
offline traces reuse the same analysis. Everything downstream (live FPS, time series,
summaries, CSVs) is computed from the recorded timestamps, so no sampler thread is
needed and results are deterministic.
"""

from __future__ import annotations

import bisect
import csv
import os
import statistics
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from power_monitor.analysis.metrics import percentile

StreamKey = Tuple[str, str]


@dataclass
class FpsStream:
    """Handle for one ``(app, stage)`` stream. Obtain via ``FpsMonitor.register``."""

    app: str
    stage: str
    target_fps: Optional[float] = None
    timestamps: List[float] = field(default_factory=list)
    frame_ids: List[Optional[int]] = field(default_factory=list)
    drop_ts: List[float] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def key(self) -> StreamKey:
        return (self.app, self.stage)

    def tick(self, ts: Optional[float] = None, frame_id: Optional[int] = None) -> float:
        """Record one completed frame/event. ``ts`` defaults to ``time.monotonic()``."""
        stamp = time.monotonic() if ts is None else float(ts)
        with self._lock:
            # Keep the list sorted even if explicit timestamps arrive slightly out of order.
            if self.timestamps and stamp < self.timestamps[-1]:
                idx = bisect.bisect_right(self.timestamps, stamp)
                self.timestamps.insert(idx, stamp)
                self.frame_ids.insert(idx, frame_id)
            else:
                self.timestamps.append(stamp)
                self.frame_ids.append(frame_id)
        return stamp

    def drop(self, count: int = 1, ts: Optional[float] = None) -> None:
        """Record frames that were scheduled/available but not processed."""
        stamp = time.monotonic() if ts is None else float(ts)
        with self._lock:
            self.drop_ts.extend([stamp] * int(count))

    @property
    def dropped(self) -> int:
        return len(self.drop_ts)

    def snapshot(self) -> Tuple[List[float], List[Optional[int]], List[float]]:
        with self._lock:
            return list(self.timestamps), list(self.frame_ids), sorted(self.drop_ts)

    def current_fps(self, window_s: float = 1.0, now: Optional[float] = None) -> float:
        """Rate over the trailing ``window_s`` seconds (live query, e.g. for a controller)."""
        now = time.monotonic() if now is None else now
        with self._lock:
            lo = bisect.bisect_right(self.timestamps, now - window_s)
            hi = bisect.bisect_right(self.timestamps, now)
        return (hi - lo) / window_s if window_s > 0 else 0.0

    def summary(self, start: Optional[float] = None, end: Optional[float] = None, window_s: float = 1.0) -> Dict[str, Any]:
        timestamps, _, drops = self.snapshot()
        if start is not None:
            timestamps = [t for t in timestamps if t >= start]
            drops = [t for t in drops if t >= start]
        if end is not None:
            timestamps = [t for t in timestamps if t <= end]
            drops = [t for t in drops if t <= end]
        return summarize_timestamps(
            timestamps,
            target_fps=self.target_fps,
            window_s=window_s,
            dropped=len(drops),
            span=(start, end),
            extra={"app": self.app, "stage": self.stage},
        )


def summarize_timestamps(
    timestamps: List[float],
    target_fps: Optional[float] = None,
    window_s: float = 1.0,
    dropped: int = 0,
    span: Tuple[Optional[float], Optional[float]] = (None, None),
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """FPS statistics for a sorted list of event timestamps (seconds)."""
    result: Dict[str, Any] = dict(extra or {})
    n = len(timestamps)
    result.update({"frames": n, "dropped": int(dropped), "target_fps": target_fps})

    intervals_ms = [(b - a) * 1000.0 for a, b in zip(timestamps, timestamps[1:])]
    first = span[0] if span[0] is not None else (timestamps[0] if timestamps else None)
    last = span[1] if span[1] is not None else (timestamps[-1] if timestamps else None)
    duration = (last - first) if (first is not None and last is not None) else 0.0

    if n >= 2 and timestamps[-1] > timestamps[0]:
        # (n-1)/(last-first) is the unbiased rate between first and last observed frame.
        mean_fps = (n - 1) / (timestamps[-1] - timestamps[0])
    else:
        mean_fps = 0.0

    windows = [row["fps"] for row in sliding_window_fps(timestamps, window_s, window_s, first, last)]
    result.update(
        {
            "duration_s": duration,
            "mean_fps": mean_fps,
            "window_fps_min": min(windows) if windows else 0.0,
            "window_fps_p5": percentile(windows, 5) if windows else 0.0,
            "window_fps_max": max(windows) if windows else 0.0,
            "interval_p50_ms": percentile(intervals_ms, 50),
            "interval_p95_ms": percentile(intervals_ms, 95),
            "interval_p99_ms": percentile(intervals_ms, 99),
            "interval_max_ms": max(intervals_ms) if intervals_ms else 0.0,
            "jitter_ms": statistics.pstdev(intervals_ms) if len(intervals_ms) >= 2 else 0.0,
            "fps_attainment": (mean_fps / target_fps) if target_fps else None,
            "drop_rate": (dropped / (n + dropped)) if (n + dropped) else 0.0,
        }
    )
    return result


def sliding_window_fps(
    timestamps: List[float],
    window_s: float = 1.0,
    step_s: float = 0.5,
    start: Optional[float] = None,
    end: Optional[float] = None,
) -> List[Dict[str, float]]:
    """FPS sampled every ``step_s``; each sample counts events in ``(t - window_s, t]``."""
    if not timestamps or window_s <= 0 or step_s <= 0:
        return []
    start = timestamps[0] if start is None else start
    end = timestamps[-1] if end is None else end
    rows: List[Dict[str, float]] = []
    t = start + window_s
    while t <= end + 1e-9:
        lo = bisect.bisect_right(timestamps, t - window_s)
        hi = bisect.bisect_right(timestamps, t)
        rows.append({"t": t, "fps": (hi - lo) / window_s})
        t += step_s
    return rows


class FpsMonitor:
    """Registry of FPS streams for any number of concurrently running apps."""

    def __init__(self, window_s: float = 1.0, step_s: float = 0.5):
        self.window_s = float(window_s)
        self.step_s = float(step_s)
        self._streams: Dict[StreamKey, FpsStream] = {}
        self._lock = threading.Lock()
        self.start_ts: Optional[float] = None
        self.end_ts: Optional[float] = None

    # ----- registration -------------------------------------------------
    def register(self, app: str, stage: str = "output", target_fps: Optional[float] = None) -> FpsStream:
        """Create (or return the existing) stream for ``(app, stage)``."""
        key = (str(app), str(stage))
        with self._lock:
            stream = self._streams.get(key)
            if stream is None:
                stream = FpsStream(app=key[0], stage=key[1], target_fps=target_fps)
                self._streams[key] = stream
            elif target_fps is not None:
                stream.target_fps = target_fps
            return stream

    def stream(self, app: str, stage: str = "output") -> FpsStream:
        with self._lock:
            return self._streams[(app, stage)]

    def streams(self) -> List[FpsStream]:
        with self._lock:
            return list(self._streams.values())

    def tick(self, app: str, stage: str = "output", ts: Optional[float] = None, frame_id: Optional[int] = None) -> float:
        """Convenience: record an event without holding a handle (auto-registers)."""
        return self.register(app, stage).tick(ts=ts, frame_id=frame_id)

    # ----- measurement window -------------------------------------------
    def start(self, ts: Optional[float] = None) -> None:
        """Mark the beginning of the measurement window (events before it are ignored)."""
        self.start_ts = time.monotonic() if ts is None else ts

    def stop(self, ts: Optional[float] = None) -> None:
        self.end_ts = time.monotonic() if ts is None else ts

    # ----- queries --------------------------------------------------------
    def current_fps(self, app: str, stage: str = "output", window_s: Optional[float] = None) -> float:
        return self.stream(app, stage).current_fps(window_s or self.window_s)

    def current_all(self, window_s: Optional[float] = None) -> Dict[str, float]:
        return {f"{s.app}/{s.stage}": s.current_fps(window_s or self.window_s) for s in self.streams()}

    def summary(self) -> List[Dict[str, Any]]:
        return [s.summary(self.start_ts, self.end_ts, self.window_s) for s in self.streams()]

    def flat_summary(self) -> Dict[str, Any]:
        """Flattened ``fps_<app>_<stage>_<metric>`` keys for run-level summary.json / CSV."""
        out: Dict[str, Any] = {}
        for row in self.summary():
            prefix = f"fps_{row['app']}_{row['stage']}"
            for metric in ("mean_fps", "window_fps_p5", "window_fps_min", "interval_p99_ms", "jitter_ms", "fps_attainment", "drop_rate", "frames"):
                out[f"{prefix}_{metric}"] = row.get(metric)
        return out

    def timeseries(self) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for s in self.streams():
            timestamps, _, _ = s.snapshot()
            start = self.start_ts if self.start_ts is not None else (timestamps[0] if timestamps else None)
            end = self.end_ts if self.end_ts is not None else (timestamps[-1] if timestamps else None)
            for row in sliding_window_fps(timestamps, self.window_s, self.step_s, start, end):
                rows.append({"monotonic": row["t"], "app": s.app, "stage": s.stage, "fps": row["fps"], "target_fps": s.target_fps})
        rows.sort(key=lambda r: (r["monotonic"], r["app"], r["stage"]))
        return rows

    # ----- output ---------------------------------------------------------
    def write_csv(self, run_dir: str | os.PathLike[str]) -> None:
        """Write ``fps_events.csv``, ``fps_timeseries.csv`` and ``fps_summary.csv``."""
        out = Path(run_dir)
        out.mkdir(parents=True, exist_ok=True)

        events: List[Dict[str, Any]] = []
        for s in self.streams():
            timestamps, frame_ids, _ = s.snapshot()
            events.extend({"monotonic": t, "app": s.app, "stage": s.stage, "frame_id": fid} for t, fid in zip(timestamps, frame_ids))
        events.sort(key=lambda r: r["monotonic"])
        _write_rows(out / "fps_events.csv", ["monotonic", "app", "stage", "frame_id"], events)
        _write_rows(out / "fps_timeseries.csv", ["monotonic", "app", "stage", "fps", "target_fps"], self.timeseries())
        summary = self.summary()
        fieldnames = list(summary[0].keys()) if summary else ["app", "stage"]
        _write_rows(out / "fps_summary.csv", fieldnames, summary)


def _write_rows(path: Path, fieldnames: List[str], rows: Iterable[Dict[str, Any]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

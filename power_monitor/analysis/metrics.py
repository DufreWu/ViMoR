from __future__ import annotations

import math
from typing import Iterable, Sequence


def llm_slowdown(concurrent_latency: float, solo_latency: float) -> float:
    """Return percentage increase in LLM request latency under contention."""
    if solo_latency in (None, 0):
        return 0.0
    return ((float(concurrent_latency) / float(solo_latency)) - 1.0) * 100.0


def perception_slowdown(concurrent_latency: float, solo_latency: float) -> float:
    """Return percentage increase in per-frame perception latency under contention."""
    if solo_latency in (None, 0):
        return 0.0
    return ((float(concurrent_latency) / float(solo_latency)) - 1.0) * 100.0


def deadline_miss_rate(missed_deadlines: int, scheduled_frames: int) -> float:
    """Return the ratio of missed deadlines to scheduled frames."""
    if scheduled_frames <= 0:
        return 0.0
    return float(missed_deadlines) / float(scheduled_frames)


def percentile(values: Sequence[float], pct: float) -> float:
    """Compute a percentile for a numeric sequence."""
    if not values:
        return 0.0
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 1:
        return ordered[0]
    rank = max(0.0, min(1.0, pct / 100.0)) * (len(ordered) - 1)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    fraction = rank - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def safe_mean(values: Iterable[float]) -> float:
    values = [float(v) for v in values if v is not None]
    return sum(values) / len(values) if values else 0.0


def safe_std(values: Iterable[float]) -> float:
    values = [float(v) for v in values if v is not None]
    if len(values) < 2:
        return 0.0
    mean_value = safe_mean(values)
    variance = sum((x - mean_value) ** 2 for x in values) / (len(values) - 1)
    return math.sqrt(variance)

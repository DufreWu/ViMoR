from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable, List

from .metrics import safe_mean, safe_std


def summarize_rows(rows: Iterable[dict]) -> dict:
    rows = list(rows)
    if not rows:
        return {"count": 0, "successful": 0, "failed": 0}

    metrics = defaultdict(list)
    for row in rows:
        for key, value in row.items():
            if key in {"scenario", "run_id", "status"}:
                continue
            try:
                metrics[key].append(float(value))
            except (TypeError, ValueError):
                continue

    summary = {
        "count": len(rows),
        "successful": sum(1 for row in rows if row.get("status") != "failed"),
        "failed": sum(1 for row in rows if row.get("status") == "failed"),
    }
    for key, values in metrics.items():
        summary[f"{key}_mean"] = safe_mean(values)
        summary[f"{key}_std"] = safe_std(values)
    return summary


def summarize_csv(input_csv: str | Path, output_csv: str | Path) -> List[dict]:
    rows = []
    with open(input_csv, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append(row)

    summary = summarize_rows(rows)
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)
    return rows


def write_summary_json(summary_path: str | Path, runs: Iterable[dict]) -> dict:
    summary = summarize_rows(runs)
    path = Path(summary_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)
    return summary

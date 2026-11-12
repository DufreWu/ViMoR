"""Aggregate ``runs_index.csv`` over repetitions: one row per (workload, CPU MHz, GPU MHz).

For every numeric metric: mean, std, n and a 95 % Student-t confidence half-width.
Run accounting is kept per group (total / failed / flagged runs and which flags), so
excluded runs stay visible. Outputs, under ``<output_root>/analysis[/<sweep_id>]/``:

* ``aggregate.csv``   - all metrics, wide format
* ``report.md``       - per-workload tables of the key hardware / QoS / LLM metrics
* ``heatmaps/*.png``  - CPU x GPU heatmaps per workload for key metrics

Energy is always *total system* (VDD_IN) energy. Per-unit energy (J/frame, J/token) is
only derived for single-app workloads, where the whole system energy belongs to that
one app; it is not computed for concurrent workloads (no shared-GPU attribution).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

# CPU/GPU targets are grouped as labels: "576", "1984", ..., "default" (stock governors), "current" (untouched).
GROUP_KEYS = ["workload_config", "workload_apps", "cpu_freq_target_mhz", "gpu_freq_target_mhz"]


def _label(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)) or str(value) in ("", "None", "nan"):
        return "current"
    try:
        return f"{float(value):.0f}"
    except ValueError:
        return str(value)


def _sort_key(label: str) -> tuple:
    try:
        return (0, float(label))
    except ValueError:
        return (1, 0.0) if label == "default" else (2, 0.0)

# (column, label, lower_is_better) for report tables / heatmaps.
KEY_METRICS = [
    ("energy_j", "Energy (J)", True),
    ("mean_power_w", "Avg power (W)", True),
    ("peak_power_w", "Peak power (W)", True),
    ("cpu_util_pct_mean", "CPU util (%)", None),
    ("gpu_util_pct_mean", "GPU util (%)", None),
    ("emc_freq_mhz_mean", "EMC (MHz)", None),
    ("ram_used_mb_mean", "RAM (MB)", None),
    ("temp_tj_c_max", "Tj max (°C)", True),
    ("yolo_fps_mean", "YOLO FPS", False),
    ("yolo_fps_q", "YOLO Q_FPS", False),
    ("yolo_fps_satisfaction_rate", "YOLO FPS sat.", False),
    ("yolo_inference_p95_ms", "YOLO P95 (ms)", True),
    ("yolo_deadline_miss_rate", "YOLO DMR", True),
    ("segformer_fps_mean", "SegF FPS", False),
    ("segformer_fps_q", "SegF Q_FPS", False),
    ("segformer_fps_satisfaction_rate", "SegF FPS sat.", False),
    ("segformer_inference_p95_ms", "SegF P95 (ms)", True),
    ("segformer_deadline_miss_rate", "SegF DMR", True),
    ("llm_ttft_ms_p50", "TTFT P50 (ms)", True),
    ("llm_ttft_ms_p95", "TTFT P95 (ms)", True),
    ("llm_tpot_ms_p50", "TPOT P50 (ms)", True),
    ("llm_tpot_ms_p95", "TPOT P95 (ms)", True),
    ("llm_decode_tps_mean", "Decode tok/s", False),
    ("llm_e2e_s_p50", "E2E P50 (s)", True),
    ("llm_failure_rate", "LLM fail rate", True),
    ("energy_per_frame_j", "J/frame (solo)", True),
    ("energy_per_output_token_j", "J/token (solo)", True),
]
HEATMAP_METRICS = ["energy_j", "mean_power_w", "yolo_fps_q", "segformer_fps_q", "yolo_inference_p95_ms",
                   "segformer_inference_p95_ms", "llm_ttft_ms_p50", "llm_tpot_ms_p50", "temp_tj_c_max"]
SEQ_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def _t95(n: int) -> float:
    if n < 2:
        return float("nan")
    try:
        from scipy import stats  # type: ignore

        return float(stats.t.ppf(0.975, n - 1))
    except ImportError:
        return 1.96


def _load(output_root: Path, sweep_id: Optional[str]) -> Any:
    import pandas as pd  # type: ignore

    index = output_root / "runs_index.csv"
    if not index.exists():
        raise FileNotFoundError(f"No runs index at {index}")
    df = pd.read_csv(index, dtype={"cpu_freq_target_mhz": str, "gpu_freq_target_mhz": str})
    if sweep_id is not None:
        df = df[df.get("sweep_id") == sweep_id]
    for col in ("cpu_freq_target_mhz", "gpu_freq_target_mhz"):
        df[col] = df[col].map(_label)
    if "valid" not in df:
        df["valid"] = True
    df["valid"] = df["valid"].astype(str).str.lower().isin(["true", "1"])
    # Per-unit energy only for single-app workloads (no attribution in concurrent ones).
    single_vision = df["workload_apps"].isin(["yolo", "segformer"])
    for app in ("yolo", "segformer"):
        col = f"{app}_processed_frames"
        if col in df:
            mask = single_vision & (df["workload_apps"] == app) & (df[col] > 0)
            df.loc[mask, "energy_per_frame_j"] = df.loc[mask, "energy_j"] / df.loc[mask, col]
    if "llm_tokens_in_window" in df:
        mask = (df["workload_apps"] == "llm") & (df["llm_tokens_in_window"] > 0)
        df.loc[mask, "energy_per_output_token_j"] = df.loc[mask, "energy_j"] / df.loc[mask, "llm_tokens_in_window"]
    return df


def aggregate_runs(output_root: Path, sweep_id: Optional[str] = None, include_flagged: bool = False, plots: bool = True) -> Path:
    import pandas as pd  # type: ignore

    df = _load(Path(output_root), sweep_id)
    out_dir = Path(output_root) / "analysis" / (sweep_id or "all")
    out_dir.mkdir(parents=True, exist_ok=True)

    accounting = []
    for key, g in df.groupby(GROUP_KEYS, dropna=False):
        flags: Dict[str, int] = {}
        for raw in g.get("quality_flags", pd.Series(dtype=object)).dropna():
            try:
                for f in json.loads(raw):
                    flags[f] = flags.get(f, 0) + 1
            except (TypeError, ValueError):
                continue
        accounting.append({**dict(zip(GROUP_KEYS, key)), "runs_total": len(g),
                           "runs_failed": int((g["status"] == "failed").sum()),
                           "runs_flagged": int(((g["status"] == "completed") & ~g["valid"]).sum()),
                           "flag_counts": json.dumps(flags, sort_keys=True)})
    acc = pd.DataFrame(accounting)

    use = df[df["status"] == "completed"]
    if not include_flagged:
        use = use[use["valid"]]
    numeric = [c for c in use.columns if c not in GROUP_KEYS and pd.api.types.is_numeric_dtype(use[c])
               and c not in ("repetition", "run_order", "seed", "measurement_seconds", "warmup_seconds")]
    rows = []
    for key, g in use.groupby(GROUP_KEYS, dropna=False):
        row: Dict[str, Any] = dict(zip(GROUP_KEYS, key))
        row["runs_used"] = len(g)
        for col in numeric:
            vals = g[col].dropna().astype(float)
            if vals.empty:
                continue
            n = len(vals)
            sd = float(vals.std(ddof=1)) if n > 1 else float("nan")
            row[f"{col}__mean"] = float(vals.mean())
            row[f"{col}__std"] = sd
            row[f"{col}__ci95"] = _t95(n) * sd / math.sqrt(n) if n > 1 else float("nan")
            row[f"{col}__n"] = n
        rows.append(row)
    agg = pd.DataFrame(rows)
    agg = acc.merge(agg, on=GROUP_KEYS, how="left") if not agg.empty else acc.assign(runs_used=0)
    agg["_order"] = [(w, _sort_key(c), _sort_key(g)) for w, c, g in zip(agg["workload_config"], agg["cpu_freq_target_mhz"], agg["gpu_freq_target_mhz"])]
    agg = agg.sort_values("_order").drop(columns="_order")
    agg.to_csv(out_dir / "aggregate.csv", index=False)
    _write_report(agg, out_dir / "report.md", sweep_id, include_flagged)
    if plots:
        _heatmaps(agg, out_dir / "heatmaps")
    return out_dir


def _fmt(mean: Any, ci: Any) -> str:
    if mean is None or (isinstance(mean, float) and math.isnan(mean)):
        return "–"
    digits = 0 if abs(mean) >= 100 else (1 if abs(mean) >= 10 else 3)
    text = f"{mean:.{digits}f}"
    if ci is not None and not (isinstance(ci, float) and math.isnan(ci)):
        text += f" ± {ci:.{digits}f}"
    return text


def _write_report(agg: Any, path: Path, sweep_id: Optional[str], include_flagged: bool) -> None:
    lines = [f"# DVFS characterization{f' — {sweep_id}' if sweep_id else ''}", "",
             "Values are mean ± 95 % CI over repetitions "
             f"({'all completed runs' if include_flagged else 'valid runs only; flagged runs are counted but excluded'}). "
             "Energy and power are total module input (VDD_IN) over the measurement window; "
             "no per-app attribution is made for concurrent workloads. CPU/GPU = `default` means the stock dynamic "
             "governors (actual mean / P5–P95 clocks are listed); `current` means clocks were not touched.", ""]
    for wid, g in agg.groupby("workload_config"):
        apps = g["workload_apps"].iloc[0]
        cols = [(c, label) for c, label, _ in KEY_METRICS if f"{c}__mean" in g and g[f"{c}__mean"].notna().any()]
        lines += [f"## {wid} ({apps})", ""]
        dyn = [(c, label) for c, label in (("cpu_freq_actual_mean", "CPU actual (MHz)"), ("gpu_freq_actual_mean", "GPU actual (MHz)"))
               if f"{c}__mean" in g and (g["cpu_freq_target_mhz"] == "default").any()]
        cols = dyn + cols
        header = ["CPU MHz", "GPU MHz", "runs (used/total, failed, flagged)"] + [label for _, label in cols]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + "---|" * len(header))
        for _, r in g.iterrows():
            used = int(r["runs_used"]) if r.get("runs_used") == r.get("runs_used") else 0
            cells = [r["cpu_freq_target_mhz"], r["gpu_freq_target_mhz"],
                     f"{used}/{int(r['runs_total'])}, {int(r['runs_failed'])}, {int(r['runs_flagged'])}"]
            cells += [_fmt(r.get(f"{c}__mean"), r.get(f"{c}__ci95")) for c, _ in cols]
            lines.append("| " + " | ".join(cells) + " |")
        flagged = g[g["flag_counts"] != "{}"]
        if not flagged.empty:
            lines += ["", "Quality flags:"]
            lines += [f"- CPU {r['cpu_freq_target_mhz']} / GPU {r['gpu_freq_target_mhz']}: {r['flag_counts']}" for _, r in flagged.iterrows()]
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _heatmaps(agg: Any, out_dir: Path) -> None:
    try:
        import matplotlib  # type: ignore

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore
        from matplotlib.colors import LinearSegmentedColormap, to_rgb  # type: ignore
    except ImportError:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQ_RAMP)
    labels = {c: label for c, label, _ in KEY_METRICS}
    for wid, g in agg.groupby("workload_config"):
        g = g[g["cpu_freq_target_mhz"].map(_sort_key).map(lambda k: k[0] == 0) & g["gpu_freq_target_mhz"].map(_sort_key).map(lambda k: k[0] == 0)]
        if g.empty:
            continue
        cpus = sorted(g["cpu_freq_target_mhz"].unique(), key=float)
        gpus = sorted(g["gpu_freq_target_mhz"].unique(), key=float)
        if len(cpus) * len(gpus) < 2:
            continue
        for metric in HEATMAP_METRICS:
            col = f"{metric}__mean"
            if col not in g or g[col].isna().all():
                continue
            grid = [[float("nan")] * len(gpus) for _ in cpus]
            for _, r in g.iterrows():
                grid[cpus.index(r["cpu_freq_target_mhz"])][gpus.index(r["gpu_freq_target_mhz"])] = r[col]
            fig, ax = plt.subplots(figsize=(1.1 * len(gpus) + 2.2, 0.8 * len(cpus) + 1.6))
            im = ax.imshow(grid, cmap=cmap, origin="lower", aspect="auto")
            vmin, vmax = im.get_clim()
            for i in range(len(cpus)):
                for j in range(len(gpus)):
                    v = grid[i][j]
                    if v != v:
                        continue
                    frac = (v - vmin) / (vmax - vmin) if vmax > vmin else 0.5
                    r_, g_, b_ = to_rgb(cmap(frac))
                    ink = "#0b0b0b" if (0.2126 * r_ + 0.7152 * g_ + 0.0722 * b_) > 0.45 else "#ffffff"
                    ax.text(j, i, _fmt(v, None), ha="center", va="center", fontsize=9, color=ink)
            ax.set_xticks(range(len(gpus)), gpus)
            ax.set_yticks(range(len(cpus)), cpus)
            ax.set_xlabel("GPU frequency (MHz)")
            ax.set_ylabel("CPU frequency (MHz)")
            ax.set_title(f"{wid} ({g['workload_apps'].iloc[0]}) — {labels.get(metric, metric)}", fontsize=10, loc="left")
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.tick_params(length=0)
            cbar = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
            cbar.outline.set_visible(False)
            fig.tight_layout()
            fig.savefig(out_dir / f"{wid}_{metric}.png", dpi=150)
            plt.close(fig)

"""Shared loading / safety model for the VLM-only study (used by baselines.py and paper_figs.py).

Measured data (vlm_bench.py):
    data/vlm_matrix.csv       one row per request, inputs x DVFS (cpu=gpu=0 -> stock governors)
    data/vlm_matrix_idle.csv  idle board power per DVFS point

Safety model: the robot keeps moving while the VLM answers, then must stop within D:
    v (L + t_act) + v^2 / (2 a) + margin <= D
    => latency budget tau(v) = (D - margin - v^2/2a) / v - t_act
    => max safe speed v*(L) = a (-T + sqrt(T^2 + 2 (D - margin) / a)),  T = L + t_act
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
NACT = (8, 16, 32)
INP = ["res_w", "res_h", "res", "n_img", "n_txt_target", "inp"]


@dataclass
class Safety:
    D: float = 5.0          # recognition / decision distance [m]
    a: float = 2.0          # braking deceleration [m/s^2]
    t_act: float = 0.05     # actuation delay [s]
    margin: float = 0.5     # [m]
    q: float = 0.95         # latency quantile used for feasibility

    def tau(self, v):
        v = np.asarray(v, float)
        return (self.D - self.margin - v ** 2 / (2 * self.a)) / v - self.t_act

    def v_star(self, lat_s):
        T = np.asarray(lat_s, float) + self.t_act
        r = self.D - self.margin
        return self.a * (-T + np.sqrt(T ** 2 + 2 * r / self.a))


def load(csv=HERE / "data/vlm_matrix.csv", safety=Safety()):
    """Return (raw, agg, p_idle): raw requests, per (cpu, gpu, input) medians + quantiles, idle power dict."""
    raw = pd.read_csv(csv)
    raw["res"] = raw.res_w.astype(str) + "x" + raw.res_h.astype(str)
    raw["inp"] = raw.res + " x" + raw.n_img.astype(str) + " t" + raw.n_txt_target.astype(str)
    g = raw.groupby(["cpu", "gpu"] + INP)
    agg = g.median(numeric_only=True)
    for n in NACT:
        agg[f"q_{n}"] = g[f"tta_{n}"].quantile(safety.q)
    agg = agg.drop(columns="rep").reset_index()
    idle_csv = Path(csv).with_name(Path(csv).stem + "_idle.csv")
    p_idle = pd.read_csv(idle_csv).groupby(["cpu", "gpu"]).idle_w.median().to_dict()
    agg["p_idle"] = [p_idle.get((c, g_), np.nan) for c, g_ in zip(agg.cpu, agg.gpu)]
    for n in NACT:
        agg[f"v_{n}"] = safety.v_star(agg[f"q_{n}"] / 1000)
    return raw, agg, p_idle


def is_gov(df):
    return (df.cpu == 0) & (df.gpu == 0)

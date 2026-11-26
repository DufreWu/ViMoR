"""Trace-driven evaluation of the basic DVFS baselines for the VLM-only, speed-given problem.

Scenario s = (mode, n_act, input x, speed v). Every policy picks one DVFS point per scenario; its cost and
deadline behaviour are replayed from the measured data (vlm_matrix.csv, vlm_matrix_idle.csv).

Modes (how the robot invokes the VLM):
    cont   back-to-back queries, VLM stops after n_act tokens: P = E_req / L, budget B = tau(v)
    event  one query every dq metres (T_q = dq / v), idle in between:
           P = (E_req + P_idle (T_q - L)) / T_q,   budget B = min(tau(v), T_q)
A DVFS point is feasible for s if its p95 TTA(n_act) <= B. Scenarios where no fixed point is feasible
are dropped (no policy can be safe there).

Policies:
    MaxPerf       CPU/GPU at the highest clocks (methods/max_pref.py)
    Governor      stock schedutil (CPU) + nvhost_podgov (GPU)
    MinCPU-MaxGPU CPU lowest, GPU highest (race-to-idle heuristic suggested by the CPU-insensitivity)
    Speed-only    knows v, not the input: one point per (mode, n_act, v), safe for every input
    Input-only    knows the input, not v: one point per (mode, n_act, x), safe for every speed
    Oracle        knows x and v, offline optimum on measured p95 (upper bound for an input+speed-aware policy)
Metrics: average power [W], saving vs MaxPerf [%], deadline-violation rate over the measured requests
of the chosen point [%].

Outputs: speed_vlm/results/baseline_{scenarios,summary}.csv, paper/tables/baselines.tex,
paper/figs/baselines_power_vs_speed.pdf, paper/generated/baseline_numbers.tex
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from vlm_data import HERE, NACT, Safety, load, is_gov

ROOT = HERE.parent
p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--csv", default=str(HERE / "data/vlm_matrix.csv"))
p.add_argument("--dq", type=float, default=2.0, help="event mode: one query per dq metres")
p.add_argument("--speeds", default="0.25:3.0:0.25", help="start:stop:step [m/s]")
p.add_argument("--D", type=float, default=5.0)
p.add_argument("--a-brake", type=float, default=2.0)
p.add_argument("--paper", default=str(ROOT / "paper"))
a = p.parse_args()

S = Safety(D=a.D, a=a.a_brake)
raw, agg, P_IDLE = load(a.csv, S)
s0, s1, st = (float(x) for x in a.speeds.split(":"))
V = np.round(np.arange(s0, s1 + 1e-9, st), 3)
MODES = ["cont", "event"]
POL = ["MaxPerf", "Governor", "MinCPU-MaxGPU", "Speed-only", "Input-only", "Oracle"]

fixed = agg[~is_gov(agg)]
PTS = sorted(set(zip(fixed.cpu, fixed.gpu)))
CMIN, CMAX = min(c for c, _ in PTS), max(c for c, _ in PTS)
GMAX = max(g for _, g in PTS)
INPUTS = sorted(agg.inp.unique(), key=lambda k: (agg[agg.inp == k].n_vis.iloc[0], k))
A = agg.set_index(["cpu", "gpu", "inp"]).sort_index()
SAMPLES = {k: {n: d[f"tta_{n}"].values / 1000 for n in NACT}
           for k, d in raw.groupby(["cpu", "gpu", "inp"])}


def budget(mode, v):
    t = float(S.tau(v))
    return min(t, a.dq / v) if mode == "event" else t


def cost(mode, n, v, r):
    L, E = r[f"tta_{n}"] / 1000, r[f"energy_{n}"]
    if mode == "cont":
        return E / L
    Tq = a.dq / v
    return (E + r.p_idle * max(Tq - L, 0.0)) / max(Tq, L)


# ---- per-scenario table over all DVFS points ------------------------------------------------
rec = []
for mode in MODES:
    for n in NACT:
        for x in INPUTS:
            for v in V:
                B = budget(mode, v)
                if B <= 0:
                    continue
                for (c, g) in PTS + [(0, 0)]:
                    r = A.loc[(c, g, x)]
                    rec.append(dict(mode=mode, n_act=n, inp=x, v=v, cpu=c, gpu=g, B=B,
                                    feas=r[f"q_{n}"] / 1000 <= B, P=cost(mode, n, v, r)))
T = pd.DataFrame(rec)
F = T[T.cpu > 0]
ok_s = F.groupby(["mode", "n_act", "inp", "v"]).feas.any()
T = T.merge(ok_s.rename("scen_ok").reset_index(), on=["mode", "n_act", "inp", "v"])
T = T[T.scen_ok].drop(columns="scen_ok")
F = T[T.cpu > 0]
key = ["mode", "n_act", "inp", "v"]


def pick_group(df, level):
    """One DVFS point per group of scenarios: feasible for all of them, minimum total power.
    If no point is feasible for all, take the one feasible for most (then cheapest)."""
    out = {}
    for k, d in df.groupby(level):
        s = d.groupby(["cpu", "gpu"]).agg(nf=("feas", "sum"), P=("P", "sum"))
        best = s.sort_values(["nf", "P"], ascending=[False, True]).index[0]
        out[k] = best
    return out


choice = {}
scen = F[key].drop_duplicates().reset_index(drop=True)
choice["MaxPerf"] = {tuple(s): (CMAX, GMAX) for s in scen.values}
choice["Governor"] = {tuple(s): (0, 0) for s in scen.values}
choice["MinCPU-MaxGPU"] = {tuple(s): (CMIN, GMAX) for s in scen.values}
sp = pick_group(F, ["mode", "n_act", "v"])
choice["Speed-only"] = {tuple(s): sp[(s[0], s[1], s[3])] for s in scen.values}
ip = pick_group(F, ["mode", "n_act", "inp"])
choice["Input-only"] = {tuple(s): ip[(s[0], s[1], s[2])] for s in scen.values}
orc = F[F.feas].sort_values("P").groupby(key).head(1).set_index(key)
choice["Oracle"] = {k: (int(r.cpu), int(r.gpu)) for k, r in orc.iterrows()}

Ti = T.set_index(key + ["cpu", "gpu"])
rows = []
for pol in POL:
    for s, (c, g) in choice[pol].items():
        mode, n, x, v = s
        r = Ti.loc[s + (c, g)]
        lat = SAMPLES[(c, g, x)][n]
        rows.append(dict(policy=pol, mode=mode, n_act=n, inp=x, v=v, cpu=c, gpu=g, P=r.P, B=r.B,
                         feas_p95=bool(r.feas), viol=float(np.mean(lat > r.B)), n_req=len(lat)))
R = pd.DataFrame(rows)
base = R[R.policy == "MaxPerf"].set_index(key).P
R["saving"] = 100 * (1 - R.P / R.set_index(key).index.map(base).values)

out = HERE / "results"
out.mkdir(exist_ok=True)
R.to_csv(out / "baseline_scenarios.csv", index=False)
summ = R.groupby(["mode", "policy"]).agg(P=("P", "mean"), saving=("saving", "mean"),
                                         viol=("viol", "mean"), unsafe=("feas_p95", lambda s: 100 * (1 - s.mean())),
                                         gpu=("gpu", "median"), n=("P", "size"))
summ["viol"] *= 100
summ = summ.reindex(pd.MultiIndex.from_product([MODES, POL], names=["mode", "policy"]))
summ_n = R.groupby(["mode", "n_act", "policy"]).agg(P=("P", "mean"), saving=("saving", "mean"),
                                                    viol=("viol", "mean")).reset_index()
summ.reset_index().to_csv(out / "baseline_summary.csv", index=False)
summ_n.to_csv(out / "baseline_summary_nact.csv", index=False)
pd.set_option("display.width", 200)
print(f"{len(scen)} feasible scenarios ({len(INPUTS)} inputs x n_act {NACT} x v {V.min()}-{V.max()} x {MODES}),"
      f" dq={a.dq} m, D={a.D} m")
print(summ.round(2).to_string())
print(summ_n.pivot_table(index=["mode", "n_act"], columns="policy", values="saving")[POL].round(1).to_string())

# ---- paper assets -------------------------------------------------------------------------------
paper = Path(a.paper)
for d in ("tables", "figs", "generated"):
    (paper / d).mkdir(parents=True, exist_ok=True)
name = {"cont": "Continuous", "event": f"Event-triggered ($d_q$={a.dq:g}\\,m)"}
pol_tex = {"MinCPU-MaxGPU": "MinCPU-MaxGPU", "Speed-only": "Speed-only", "Input-only": "Input-only",
           "Oracle": "Oracle (input+speed)"}
L = [r"\begin{tabular}{l" + "rrr" * len(MODES) + "}", r"\toprule",
     " & " + " & ".join(rf"\multicolumn{{3}}{{c}}{{{name[m]}}}" for m in MODES) + r" \\",
     "".join(rf"\cmidrule(lr){{{2 + 3 * i}-{4 + 3 * i}}}" for i in range(len(MODES))),
     "Policy" + r" & $\bar P$ [W] & Saving [\%] & Viol. [\%]" * len(MODES) + r" \\", r"\midrule"]
for pol in POL:
    cells = []
    for m in MODES:
        r = summ.loc[(m, pol)]
        cells += [f"{r.P:.2f}", f"{r.saving:.1f}", f"{r.viol:.1f}"]
    L.append(pol_tex.get(pol, pol) + " & " + " & ".join(cells) + r" \\")
    if pol == "Input-only":
        L.append(r"\midrule")
L += [r"\bottomrule", r"\end{tabular}"]
(paper / "tables/baselines.tex").write_text("\n".join(L) + "\n")


def mac(nm, val):
    return rf"\newcommand{{\{nm}}}{{{val}}}"


g = summ
M = [mac("NScen", len(scen)), mac("Dq", f"{a.dq:g}"),
     mac("OracleSaveCont", f"{g.loc[('cont', 'Oracle')].saving:.0f}"),
     mac("OracleSaveEvent", f"{g.loc[('event', 'Oracle')].saving:.0f}"),
     mac("GovSaveCont", f"{g.loc[('cont', 'Governor')].saving:.0f}"),
     mac("GovSaveEvent", f"{g.loc[('event', 'Governor')].saving:.0f}"),
     mac("GovViolCont", f"{g.loc[('cont', 'Governor')].viol:.1f}"),
     mac("GovViolEvent", f"{g.loc[('event', 'Governor')].viol:.1f}"),
     mac("OracleVsGovCont", f"{100 * (1 - g.loc[('cont', 'Oracle')].P / g.loc[('cont', 'Governor')].P):.0f}"),
     mac("OracleVsGovEvent", f"{100 * (1 - g.loc[('event', 'Oracle')].P / g.loc[('event', 'Governor')].P):.0f}"),
     mac("SpeedOnlySaveCont", f"{g.loc[('cont', 'Speed-only')].saving:.0f}"),
     mac("InputOnlySaveCont", f"{g.loc[('cont', 'Input-only')].saving:.0f}"),
     mac("RaceSaveCont", f"{g.loc[('cont', 'MinCPU-MaxGPU')].saving:.0f}"),
     mac("RaceSaveEvent", f"{g.loc[('event', 'MinCPU-MaxGPU')].saving:.0f}")]
(paper / "generated/baseline_numbers.tex").write_text("\n".join(M) + "\n")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import plot_style

    plot_style.apply()
    fig, ax = plt.subplots(1, 2, figsize=plot_style.FIG2, sharey=False)
    title = {"cont": "(a) continuous", "event": f"(b) event-triggered ($d_q$={a.dq:g} m)"}
    for x_, m in zip(ax, MODES):
        d = R[R["mode"] == m].groupby(["policy", "v"]).P.mean().unstack(0)[POL]
        for pol, mk in zip(POL, "os^vDx"):
            x_.plot(d.index, d[pol], marker=mk, ms=5, lw=1.5, label=pol)
        x_.set(xlabel="speed $v$ [m/s]", title=title[m])
        x_.grid(alpha=.3)
    ax[0].set_ylabel("avg. power [W]")
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, frameon=False, columnspacing=1, handletextpad=.3)
    fig.tight_layout(rect=(0, 0, 1, .8))
    fig.savefig(paper / "figs/baselines_power_vs_speed.pdf")
    print("wrote", paper / "figs/baselines_power_vs_speed.pdf")
except ImportError:
    pass
print("wrote", paper / "tables/baselines.tex", paper / "generated/baseline_numbers.tex")

"""VLM-only: task inputs -> TTFT/TTA, and the speed-aware choice of DVFS.

Inputs (decided by the task, not optimised): image resolution, number of images,
text prompt length, and n_act (output tokens needed before the robot can act).
Control knob: DVFS (CPU x GPU clock).
Reads the CSV written by vlm_bench.py (default speed_vlm/data/vlm_matrix.csv).

Speed model: the robot must have the VLM answer and still be able to brake before
a point D metres ahead (D = task recognition / decision distance, a parameter):
    v * (TTA_q(n_act) + t_act) + v^2 / (2 a_brake) + margin <= D
    => v*(input, DVFS) = a (-T + sqrt(T^2 + 2 (D - margin) / a)),  T = TTA_q + t_act
For each input and speed v, the cheapest feasible DVFS point is the speed-aware
choice; it is compared with always running at the highest clocks.

All parameters are assumptions; change them on the command line.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--csv", default=str(ROOT / "speed_vlm/data/vlm_matrix.csv"))
p.add_argument("--n-act", default="8,16,32", help="n_act values (must be measured tta_n columns)")
p.add_argument("--q", type=float, default=0.95, help="latency quantile used for feasibility")
p.add_argument("--D", type=float, default=5.0, help="recognition / decision distance [m]")
p.add_argument("--a-brake", type=float, default=2.0)
p.add_argument("--t-act", type=float, default=0.05)
p.add_argument("--margin", type=float, default=0.5)
p.add_argument("--speeds", default="0.25,0.5,0.75,1,1.25,1.5,2")
p.add_argument("--power", default="mean_w", choices=["mean_w", "energy_j"],
               help="cost to minimise: mean power (VLM runs continuously) or energy per request")
p.add_argument("--dq", type=float, default=2.0,
               help="Mode B: one VLM query per dq metres travelled -> period T_q = dq / v [s]")
p.add_argument("--tq", type=float, default=0.0, help="Mode B: fixed query period [s] (overrides --dq)")
p.add_argument("--idle-drop", action="store_true",
               help="Mode B: between queries drop to the lowest-idle-power DVFS point "
                    "(default: stay at the chosen point)")
p.add_argument("--fig", default=str(ROOT / "speed_vlm/fig_vlm_speed"))
a = p.parse_args()

df = pd.read_csv(a.csv)
df["res"] = df.res_w.astype(str) + "x" + df.res_h.astype(str)
NACT = [int(x) for x in a.n_act.split(",") if f"tta_{x}" in df]
if not NACT:
    raise SystemExit(f"none of n_act={a.n_act} measured; have {[c for c in df if c.startswith('tta_')]}")
INP = ["res_w", "res_h", "res", "n_img", "n_txt_target"]
g = df.groupby(["cpu", "gpu"] + INP)
agg = g.median(numeric_only=True)
for n in NACT:
    agg[f"q_{n}"] = g[f"tta_{n}"].quantile(a.q)
agg = agg.reset_index().sort_values(["res_w", "n_img", "n_txt_target", "cpu", "gpu"])
gov = agg[(agg.cpu == 0) & (agg.gpu == 0)]          # stock governors (baseline, not a candidate)
agg = agg[(agg.cpu != 0) | (agg.gpu != 0)]
V = [float(x) for x in a.speeds.split(",")]
CPUS, GPUS = sorted(agg.cpu.unique()), sorted(agg.gpu.unique())
CMAX, GMAX = CPUS[-1], GPUS[-1]
Q = f"p{int(100 * a.q)}"
pd.set_option("display.width", 220)
pd.set_option("display.max_rows", 500)


def v_star(tta_ms):
    T = np.asarray(tta_ms, dtype=float) / 1000 + a.t_act
    rhs = a.D - a.margin
    return np.where(rhs > 0, a.a_brake * (-T + np.sqrt(T ** 2 + 2 * rhs / a.a_brake)), 0.0)


for n in NACT:
    agg[f"v_{n}"] = v_star(agg[f"q_{n}"])
    gov = gov.assign(**{f"v_{n}": v_star(gov[f"q_{n}"])})

print(f"Data: {a.csv} ({len(df)} requests). CPU MHz {CPUS}, GPU MHz {GPUS}")
print(f"D={a.D} m, a_brake={a.a_brake}, t_act={a.t_act}, margin={a.margin}, latency {Q}, "
      f"cost={a.power}; zero-latency bound v={float(v_star(-1000 * a.t_act)):.2f} m/s\n")

# 1) breakdown per input at the highest clocks -----------------------------------------
top = agg[(agg.cpu == CMAX) & (agg.gpu == GMAX)]
cols = (["res", "n_img", "n_txt_target", "n_vis", "n_txt", "t_pre", "t_enc", "t_img_pf", "t_txt_pf", "ttft", "tpot"]
        + [f"q_{n}" for n in NACT])
print(f"1) Median breakdown [ms] at CPU {CMAX} / GPU {GMAX} MHz (q_n = {Q} TTA(n))")
print(top[cols].round(0).to_string(index=False), "\n")

# 2) sensitivity to the clocks ------------------------------------------------------------
print(f"2a) TTFT vs GPU clock: fit TTFT = a + b*(1000/f_gpu) at CPU {CMAX} (a = clock-independent part)")
fits = []
for keys, d in agg[agg.cpu == CMAX].groupby(INP):
    if d.gpu.nunique() < 2:
        continue
    x = 1000 / d.gpu.values
    b, a0 = np.polyfit(x, d.ttft.values, 1)
    b2, a2 = np.polyfit(x, d.tpot.values, 1)
    fits.append(dict(res=keys[2], n_img=keys[3], n_txt=keys[4], ttft_a=a0, ttft_b=b,
                     R2=np.corrcoef(x, d.ttft)[0, 1] ** 2, tpot_a=a2, tpot_b=b2,
                     **{f"ttft@{int(gq)}": d.loc[d.gpu == gq, "ttft"].iloc[0] for gq in (GPUS[0], GMAX)}))
if fits:
    print(pd.DataFrame(fits).round(2).to_string(index=False), "\n")
if len(CPUS) > 1:
    print(f"2b) CPU effect at GPU {GMAX}: TTFT and TPOT [ms] per CPU MHz (ratio = lowest/highest CPU)")
    t = agg[agg.gpu == GMAX].pivot_table(index=["res_w", "res", "n_img", "n_txt_target"], columns="cpu",
                                         values=["ttft", "tpot"]).droplevel(0)
    t[("ttft", "ratio")] = t[("ttft", CPUS[0])] / t[("ttft", CMAX)]
    t[("tpot", "ratio")] = t[("tpot", CPUS[0])] / t[("tpot", CMAX)]
    print(t.round(2).to_string(), "\n")

# 3) max safe speed per input --------------------------------------------------------------
print(f"3) Max safe speed v* [m/s] per input at the highest clocks, per n_act")
t3 = top[["res", "n_img", "n_txt_target"] + [f"v_{n}" for n in NACT]].rename(
    columns={f"v_{n}": f"n_act={n}" for n in NACT})
print(t3.round(2).to_string(index=False), "\n")
n0 = NACT[0]
print(f"   v* [m/s] vs GPU clock (best CPU), n_act={n0}")
print(agg.pivot_table(index=["res_w", "res", "n_img", "n_txt_target"], columns="gpu", values=f"v_{n0}",
                      aggfunc="max").droplevel(0).round(2).to_string(), "\n")

# 4) speed-aware DVFS: cheapest feasible point per (input, speed) ---------------------------
unit = "W" if a.power == "mean_w" else "J"
for n in NACT:
    print(f"4) n_act={n}: cheapest feasible CPU/GPU [MHz] and {a.power} per speed; "
          f"saving vs highest clocks in brackets" + (", then vs stock governor" if len(gov) else "")
          + "; x = infeasible, (gov x) = governor misses the bound")
    rows, sav = {}, []
    for keys, d in agg.groupby(INP):
        maxp = d[(d.cpu == CMAX) & (d.gpu == GMAX)][a.power]
        maxp = float(maxp.iloc[0]) if len(maxp) else np.nan
        gk = gov[(gov.res_w == keys[0]) & (gov.res_h == keys[1]) & (gov.n_img == keys[3])
                 & (gov.n_txt_target == keys[4])]
        col = []
        for v in V:
            ok = d[d[f"v_{n}"] >= v]
            if ok.empty:
                col.append("x")
                continue
            b = ok.loc[ok[a.power].idxmin()]
            s = 100 * (1 - b[a.power] / maxp)
            sav.append(dict(v=v, saving=s))
            txt = f"{int(b.cpu)}/{int(b.gpu)} {b[a.power]:.1f}{unit} ({s:.0f}%"
            if len(gk):
                gp, gv = float(gk[a.power].iloc[0]), float(gk[f"v_{n}"].iloc[0])
                txt += f", gov {100 * (1 - b[a.power] / gp):.0f}%" + ("" if gv >= v else " x")
            col.append(txt + ")")
        rows[f"{keys[2]} x{keys[3]} txt{keys[4]}"] = col
    print(pd.DataFrame(rows, index=[f"v={v}" for v in V]).T.to_string())
    if sav:
        s = pd.DataFrame(sav).groupby("v").saving.agg(["mean", "min", "max", "size"]).round(0)
        print(f"   saving vs highest clocks over feasible inputs (size = #inputs feasible):\n{s.T.to_string()}")
    print()

# 6) Mode B: event-triggered queries ------------------------------------------------------------
# One query every T_q seconds (T_q = dq / v, or --tq). The VLM stops after n_act tokens, then the
# board idles until the next query. Average power over one period:
#     P_B(f) = [E_req,n(f) + P_idle(f) * (T_q - L_n(f))] / T_q,   needs L_n(f) <= T_q
# with E_req,n = energy_n (board energy over [0, TTA(n)]) and L_n = median TTA(n).
# Both modes must meet the cadence L_q,n(f) <= T_q (and the safety bound v_n >= v).
# Mode A under the same cadence (back-to-back, stop at n_act): P_A = E_req,n / L_n.
idle_path = Path(a.csv).with_name(Path(a.csv).stem + "_idle.csv")
P_IDLE = {}
if idle_path.exists() and all(f"energy_{n}" in df for n in NACT):
    P_IDLE = pd.read_csv(idle_path).groupby(["cpu", "gpu"]).idle_w.median().to_dict()
    fixed = {k: w for k, w in P_IDLE.items() if k != (0, 0)}
    floor_k = min(fixed, key=fixed.get)
    print(f"6) Mode B (event-triggered): idle power [W] per CPU/GPU MHz (0/0 = stock governor):")
    print("   " + ", ".join(f"{c}/{g} {w:.2f}" for (c, g), w in sorted(P_IDLE.items())))
    per = f"T_q = {a.tq} s" if a.tq else f"T_q = dq/v, dq = {a.dq} m"
    print(f"   {per}; between queries: "
          + (f"drop to {floor_k[0]}/{floor_k[1]} ({fixed[floor_k]:.2f} W)" if a.idle_drop else "stay at the chosen point"))

    def p_idle(c, g):
        if a.idle_drop and (c, g) != (0, 0):
            return fixed[floor_k]
        return P_IDLE.get((int(c), int(g)), np.nan)

    def p_event(rows, n, Tq):
        L = rows[f"tta_{n}"].values / 1000
        pi = np.array([p_idle(c, g) for c, g in zip(rows.cpu, rows.gpu)])
        P = (rows[f"energy_{n}"].values + pi * (Tq - L)) / Tq
        return np.where(rows[f"q_{n}"].values / 1000 <= Tq, P, np.nan)

    def p_cont(rows, n, Tq):
        # Mode A under the same cadence: back-to-back queries (period = L_n <= T_q), stop at n_act
        P = rows[f"energy_{n}"].values / (rows[f"tta_{n}"].values / 1000)
        return np.where(rows[f"q_{n}"].values / 1000 <= Tq, P, np.nan)

    figB = []
    for n in NACT:
        print(f"\n6) n_act={n}: Mode B cheapest feasible CPU/GPU and average power per speed; in brackets: "
              f"saving vs Mode B at highest clocks, vs Mode B with stock governor, and vs Mode A optimum")
        rows, sav = {}, []
        for keys, d in agg.groupby(INP):
            gk = gov[(gov.res_w == keys[0]) & (gov.res_h == keys[1]) & (gov.n_img == keys[3])
                     & (gov.n_txt_target == keys[4])]
            col = []
            for v in V:
                Tq = a.tq if a.tq else a.dq / v
                ok = d[d[f"v_{n}"] >= v]
                if ok.empty:
                    col.append("x")
                    continue
                pB = p_event(ok, n, Tq)
                if np.all(np.isnan(pB)):
                    col.append("x (L>T_q)")
                    continue
                i = int(np.nanargmin(pB))
                b, pb = ok.iloc[i], pB[i]
                pA = float(np.nanmin(p_cont(ok, n, Tq)))
                top_r = d[(d.cpu == CMAX) & (d.gpu == GMAX)]
                pmax = float(p_event(top_r, n, Tq)[0]) if len(top_r) else np.nan
                s_max = 100 * (1 - pb / pmax)
                txt = f"{int(b.cpu)}/{int(b.gpu)} {pb:.1f}W (max {s_max:.0f}%"
                s_gov = np.nan
                if len(gk):
                    pg = float(p_event(gk, n, Tq)[0])
                    s_gov = 100 * (1 - pb / pg)
                    txt += f", gov {s_gov:.0f}%" + ("" if float(gk[f"v_{n}"].iloc[0]) >= v else " x")
                s_A = 100 * (1 - pb / pA)
                txt += f", A {s_A:.0f}%)"
                col.append(txt)
                sav.append(dict(v=v, Tq=Tq, gpu=int(b.gpu), vs_max=s_max, vs_gov=s_gov, vs_A=s_A, P_B=pb, P_A=pA))
                if n == NACT[0]:
                    figB.append(dict(inp=f"{keys[2]} x{keys[3]} txt{keys[4]}", v=v, P_A=pA, P_B=pb))
            rows[f"{keys[2]} x{keys[3]} txt{keys[4]}"] = col
        print(pd.DataFrame(rows, index=[f"v={v}" for v in V]).T.to_string())
        if sav:
            s = pd.DataFrame(sav).groupby("v").agg(
                T_q=("Tq", "first"), P_B=("P_B", "mean"), P_A=("P_A", "mean"), vs_A=("vs_A", "mean"),
                vs_max=("vs_max", "mean"), vs_gov=("vs_gov", "mean"),
                gpu_median=("gpu", "median"), n=("gpu", "size")).round(2)
            print(f"   mean over feasible inputs (P in W, savings in %, gpu_median = chosen GPU MHz):\n{s.T.to_string()}")
else:
    print(f"6) Mode B skipped: need {idle_path.name} and energy_<n_act> columns")
print()

# 5) figure -------------------------------------------------------------------------------
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import plot_style

    plot_style.apply()
    fig, ax = plt.subplots(1, 3, figsize=plot_style.FIG3)
    sel = top[top.n_txt_target == top.n_txt_target.min()]
    for k, d in sel.groupby("n_img"):
        ax[0].plot(d.n_vis, d.ttft / 1000, "o-", label=f"{k} img")
    ax[0].set(xlabel="visual tokens", ylabel="median TTFT [s]",
              title=f"(a) TTFT, text {int(sel.n_txt_target.min())} tok, {CMAX}/{GMAX} MHz")
    d = agg[agg.cpu == CMAX]
    for keys, dd in d.groupby(INP):
        if keys[4] == d.n_txt_target.min() and keys[3] == d.n_img.min():
            ax[1].plot(dd[f"v_{n0}"], dd[a.power], "o-", label=keys[2])
    ax[1].set(xlabel=f"$v^*$ [m/s] ($n_{{act}}$={n0})", ylabel="mean power [W]" if a.power == "mean_w"
              else "energy [J]", title=f"(b) one point per GPU clock, {int(d.n_img.min())} img")
    for k, dd in sel.groupby("n_img"):
        ax[2].plot(dd.n_vis, dd[f"v_{n0}"], "o-", label=f"{k} img")
    ax[2].set(xlabel="visual tokens", ylabel="$v^*$ [m/s] at max clocks", title=f"(c) $n_{{act}}$={n0}")
    for x in ax:
        x.grid(alpha=0.3)
        x.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(a.fig + ".pdf")
    print(f"figure: {a.fig}.pdf")
    if P_IDLE and figB:
        fb = pd.DataFrame(figB)
        sel_inp = [i for i in fb.inp.unique() if "txt32" in i and (" x1 " in i or "x4 txt" in i)]
        fig, ax = plt.subplots(figsize=plot_style.FIG1)
        for k, i in enumerate(sel_inp):
            d = fb[fb.inp == i]
            line, = ax.plot(d.v, d.P_A, "o--", lw=1.2, ms=4)
            ax.plot(d.v, d.P_B, "o-", color=line.get_color(), ms=4, label=i.replace(" txt32", ""))
        ax.set(xlabel="speed $v$ [m/s]", ylabel="average power [W]",
               title=f"$n_{{act}}$={NACT[0]}, " + (f"$T_q$={a.tq} s" if a.tq else f"query every {a.dq} m")
               + "; dashed: cont., solid: event")
        ax.grid(alpha=0.3)
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, hi + 0.7 * (hi - lo))
        ax.legend(frameon=False, ncol=3, loc="upper left", columnspacing=.8, handlelength=1.5)
        fig.tight_layout()
        fig.savefig(a.fig + "_modes.pdf")
        print(f"figure: {a.fig}_modes.pdf")
except ImportError:
    pass

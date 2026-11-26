"""Motivation figures + numbers for the paper (paper/figs/*.pdf, paper/generated/motivation_numbers.tex).
Run baselines.py first (mot_oracle_gpu reads results/baseline_scenarios.csv)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import plot_style
from vlm_data import HERE, Safety, load, is_gov

plot_style.apply()
PAPER = HERE.parent / "paper"
(PAPER / "figs").mkdir(parents=True, exist_ok=True)
(PAPER / "generated").mkdir(parents=True, exist_ok=True)
S = Safety()
raw, agg, P_IDLE = load(safety=S)
fx = agg[~is_gov(agg)]
gov = agg[is_gov(agg)].set_index("inp")
CPUS, GPUS = sorted(fx.cpu.unique()), sorted(fx.gpu.unique())
CMIN, CMAX, GMIN, GMAX = CPUS[0], CPUS[-1], GPUS[0], GPUS[-1]
top = fx[(fx.cpu == CMAX) & (fx.gpu == GMAX)].set_index("inp")
ORDER = top.sort_values(["n_vis", "n_txt_target"]).index.tolist()
LS = dict(zip(CPUS, [":", "--", "-"]))
N = {}

# Fig. 1: input -> latency (2 panels) ----------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=plot_style.FIG2, gridspec_kw=dict(width_ratios=[1.5, 1]))
s = top.loc[ORDER]
s = s[s.n_txt_target == 128]
bot = np.zeros(len(s))
for c, lab in [("t_pre", "preproc."), ("t_enc", "encoder"), ("t_img_pf", "img prefill"),
               ("t_txt_pf", "txt prefill")]:
    ax[0].bar(range(len(s)), s[c] / 1000, bottom=bot, label=lab, width=.75)
    bot += s[c].values / 1000
ax[0].set_xticks(range(len(s)), [f"{r.res_w}×{r.n_img}" for r in s.itertuples()], rotation=60, ha="right")
ax[0].set(ylabel="TTFT [s]", xlabel="width × #images", title="(a) TTFT breakdown")
ax[0].legend(frameon=False, handlelength=1, labelspacing=.2)
for n, mk in zip((8, 16, 32), "o^s"):
    ax[1].scatter(top.n_vis, top[f"v_{n}"], s=18, marker=mk, label=f"$n_{{act}}$={n}")
ax[1].set(xlabel="visual tokens $N_{vis}$", ylabel="$v^*$ [m/s]", title="(b) max safe speed")
ax[1].legend(frameon=False, handletextpad=.1)
for x in ax:
    x.grid(alpha=.3)
fig.tight_layout()
fig.savefig(PAPER / "figs/mot_input_latency.pdf")
N.update(TtftMin=f"{top.ttft.min() / 1000:.2f}", TtftMax=f"{top.ttft.max() / 1000:.1f}",
         TtftRatio=f"{top.ttft.max() / top.ttft.min():.0f}",
         VisShareMax=f"{100 * ((top.t_enc + top.t_img_pf) / top.ttft).max():.0f}",
         TpotMs=f"{top.tpot.median():.0f}",
         VstarMin=f"{top.v_8.min():.2f}", VstarMax=f"{top.v_8.max():.2f}",
         VstarRatio=f"{top.v_8.max() / top.v_8.min():.0f}")

# Fig. 2: clocks -> latency and power (3 panels) -----------------------------------------------
fits = []
for k, d in fx[fx.cpu == CMAX].groupby("inp"):
    xx = 1000 / d.gpu.values
    fits.append(dict(R2=np.corrcoef(xx, d.ttft)[0, 1] ** 2, sp=d.ttft.max() / d.ttft.min()))
fits = pd.DataFrame(fits)
cr = fx[fx.gpu == GMAX].pivot_table(index="inp", columns="cpu", values="ttft")
cr = cr[CMIN] / cr[CMAX]
pw = fx.groupby(["cpu", "gpu"]).mean_w.median().unstack()
fx = fx.assign(e_rel=fx.energy_8 / fx.inp.map(top.energy_8))
er = fx.groupby(["cpu", "gpu"]).e_rel.mean().unstack()

fig, ax = plt.subplots(1, 3, figsize=plot_style.FIG3)
for i, k in enumerate([ORDER[0], ORDER[len(ORDER) // 2], ORDER[-1]]):
    for cpu in CPUS:
        d = fx[(fx.inp == k) & (fx.cpu == cpu)]
        ax[0].plot(d.gpu, d.ttft / 1000, LS[cpu], marker="o", ms=4, color=f"C{i}",
                   label=k.replace(" t", ", t") if cpu == CMAX else None)
ax[0].set(xlabel="GPU frequency [MHz]", ylabel="TTFT [s]", yscale="log",
          title="(a) TTFT vs. clocks (line style: CPU)")
ax[0].set_ylim(0.3, 400)
ax[0].legend(frameon=False, loc="upper right")
for cpu in CPUS:
    ax[1].plot(GPUS, pw.loc[cpu], LS[cpu], marker="o", ms=4, color="C0", label=f"CPU {cpu}")
    ax[1].plot(GPUS, [P_IDLE[(cpu, g)] for g in GPUS], LS[cpu], marker="s", ms=3, color="gray",
               label="idle" if cpu == CMAX else None)
    ax[2].plot(GPUS, er.loc[cpu], LS[cpu], marker="o", ms=4, color="C0", label=f"CPU {cpu} MHz")
ax[1].axhline(gov.mean_w.median(), color="C3", lw=1.5, label="governor")
ax[1].set(xlabel="GPU frequency [MHz]", ylabel="board power [W]", title="(b) active / idle power")
ax[1].set_ylim(4, 26)
ax[1].legend(frameon=False, loc="upper left", ncol=2, columnspacing=.8, handlelength=1.5)
ax[2].set(xlabel="GPU frequency [MHz]", ylabel=f"rel. energy / request", title="(c) energy per request")
ax[2].legend(frameon=False)
for x in ax:
    x.grid(alpha=.3)
fig.tight_layout()
fig.savefig(PAPER / "figs/mot_dvfs.pdf")
N.update(FitRtwo=f"{fits.R2.min():.3f}", GpuSpMin=f"{fits.sp.min():.1f}", GpuSpMax=f"{fits.sp.max():.1f}",
         CpuEffect=f"{100 * max(abs(cr.max() - 1), abs(1 - cr.min())):.0f}",
         PActMin=f"{pw.values.min():.1f}", PActMax=f"{pw.values.max():.1f}",
         PIdleMin=f"{min(v for k, v in P_IDLE.items() if k != (0, 0)):.1f}",
         PIdleMax=f"{max(v for k, v in P_IDLE.items() if k != (0, 0)):.1f}",
         EreqBest=f"{100 * (1 - er.values.min()):.0f}",
         GovGpu=f"{gov.gpu_mean.median():.0f}", GovCpu=f"{gov.cpu_mean.median():.0f}",
         GovSlowMax=f"{100 * ((gov.ttft / top.ttft).max() - 1):.0f}")

# Fig. 3: oracle GPU clock over (input, speed) (2 panels) -------------------------------------
R = pd.read_csv(HERE / "results/baseline_scenarios.csv")
o = R[(R.policy == "Oracle") & (R.n_act == 8)]
NV = top.loc[ORDER, "n_vis"].astype(int).values
fig, ax = plt.subplots(1, 2, figsize=plot_style.FIG2, sharey=True, layout="constrained")
for x, m, t in zip(ax, ["cont", "event"], ["(a) continuous", "(b) event-triggered"]):
    h = o[o["mode"] == m].pivot_table(index="inp", columns="v", values="gpu").reindex(ORDER)
    im = x.imshow(h.values, aspect="auto", cmap="magma", vmin=GMIN, vmax=GMAX)
    x.set_xticks(range(h.shape[1])[1::4], [f"{v:g}" for v in h.columns][1::4])
    x.set(xlabel="speed $v$ [m/s]", title=t)
yt = [0, 9, 18, 27, 35]
ax[0].set_yticks(yt, [str(NV[i]) for i in yt])
ax[0].set_ylabel("inputs (by $N_{vis}$)")
cb = fig.colorbar(im, ax=ax, fraction=.05, pad=.02)
cb.set_label("oracle GPU [MHz]")
fig.savefig(PAPER / "figs/mot_oracle_gpu.pdf")
N.update(NInputs=str(len(ORDER)), NReq=str(len(raw)))

(PAPER / "generated/motivation_numbers.tex").write_text(
    "\n".join(rf"\newcommand{{\{k}}}{{{v}}}" for k, v in N.items()) + "\n")
print("\n".join(f"{k} = {v}" for k, v in N.items()))

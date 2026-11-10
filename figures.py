"""Figures for the paper, drawn from the registered runs. Vector PDF (fonts embedded) and PNG previews.

    .venv/bin/python figures.py --out figures

Figure 1: collapses and daytime on cold ground in the frozen check, by childhood and adult world,
          seeds 100-169 (version 1, its replication and the scarce, rich and adult childhoods of
          experiment 2; identical code and settings).
Figure 2: where the trace lives: layer similarity between twins and chimera positions (trace_v1/).
Figure 3: experiment 2: food carried at home and energy in the standard adult world, by childhood.
Means over seeds with bootstrap 95% confidence intervals (10,000 resamples of seeds).
"""
import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

# Inter as TrueType, so the PDFs embed fonts publishers accept. FIG_FONT_DIR points to Inter-Regular.ttf
# and Inter-SemiBold.ttf (from the Inter release, extras/ttf); without it the system's Inter or default is used.
if os.environ.get("FIG_FONT_DIR"):
    for f in Path(os.environ["FIG_FONT_DIR"]).glob("*.ttf"):
        font_manager.fontManager.addfont(str(f))
    font_manager.fontManager.ttflist = [f for f in font_manager.fontManager.ttflist
                                        if not (f.name == "Inter" and f.fname.endswith(".otf"))]

import numpy as np
import pandas as pd

from analyze_v1 import phase_mean

INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
COLOR = {"scarce": "#2a78d6", "adult": "#eb6834", "rich": "#1baf7a"}   # validated: worst CVD dE 9.2
NAME = {"scarce": "scarce childhood", "adult": "childhood in the adult world", "rich": "rich childhood",
        "hungry": "hungry", "unreliable": "unreliable", "none": "no childhood"}
plt.rcParams.update({
    "font.family": "Inter", "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7, "pdf.fonttype": 42,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "axes.titleweight": "semibold",
    "axes.titlelocation": "left", "figure.dpi": 150,
})
RNG = np.random.default_rng(0)


def ci(v):
    v = np.asarray(v, float)
    boot = RNG.choice(v, (10_000, len(v))).mean(1)
    return v.mean(), np.percentile(boot, 2.5), np.percentile(boot, 97.5)


def lives(root, seeds, childhoods):
    out = {}
    for c in childhoods:
        for s in seeds:
            for b in ["standard", "richer", "poorer"]:
                f = Path(root) / f"{c}_s{s}_{b}.csv"
                if f.exists():
                    out[(c, s, b)] = pd.read_csv(f)
    return out


def grid(ax):
    ax.yaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def figure1(L, seeds, out):
    # outside droughts in all three worlds, so the x axis is one quantity: food growth outside droughts
    worlds = [("poorer", "_normal"), ("standard", "_normal"), ("richer", "_normal")]
    xlab = ["poorer\n0.0008", "standard\n0.004", "richer\n0.04"]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), gridspec_kw={"wspace": 0.42})
    for ax, (m, title, ylabel) in zip(axes, [
            ("collapses_per_1k", "a  Collapses right after the move", "collapses per 1,000 steps"),
            ("cold_day", "b  Daytime spent on cold ground", "share of daytime")]):
        x = np.arange(3)
        for c in ["scarce", "adult", "rich"]:
            stats = [ci([phase_mean(L[(c, s, b)], "frozen", m + suf) for s in seeds]) for b, suf in worlds]
            mean = np.array([t[0] for t in stats])
            lo, hi = np.array([t[1] for t in stats]), np.array([t[2] for t in stats])
            ax.plot(x, mean, color=COLOR[c], lw=2, zorder=3)
            ax.vlines(x, lo, hi, color=COLOR[c], lw=1.2, zorder=3)
            ax.scatter(x, mean, s=22, color=COLOR[c], edgecolor="white", linewidth=1.2, zorder=4)
            ax.annotate(NAME[c].replace("childhood in the adult world", "adult-world childhood"),
                        (x[-1], mean[-1]), xytext=(6, 0), textcoords="offset points",
                        va="center", fontsize=6.8, color=INK)
        ax.set_xticks(x, xlab)
        ax.set_xlim(-0.3, 2.95)
        ax.set_ylabel(ylabel)
        ax.set_title(title, color=INK, pad=6)
        ax.set_ylim(bottom=0)
        grid(ax)
    for ax in axes:
        ax.set_xlabel("adult world: food growth per patch per step, outside droughts", color=INK2, labelpad=4)
    fig.savefig(out / "fig1_mismatch.pdf", bbox_inches="tight")
    fig.savefig(out / "fig1_mismatch.png", bbox_inches="tight", dpi=200)
    plt.close(fig)


def figure2(trace_dir, out):
    sim = pd.read_csv(trace_dir / "similarity.csv")
    chim = pd.read_csv(trace_dir / "chimeras.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9), gridspec_kw={"wspace": 0.95, "width_ratios": [1, 1.3]})

    ax = axes[0]
    layers = ["hidden1", "hidden2", "actions"]
    pairs = [("scarce-adult", "scarce", "adult"), ("scarce-rich", "scarce", "rich"), ("rich-adult", "rich", "adult")]
    # pairs are not childhoods, so they get no childhood colour: grey steps plus direct labels
    pair_color = {"scarce-adult": "#0b0b0b", "rich-adult": "#7a7973", "scarce-rich": "#b5b4ad"}
    x = np.arange(3)
    for p, a, b in pairs:
        stats = [ci(sim[(sim.pair == p) & (sim.layer == l)].cka) for l in layers]
        mean = np.array([t[0] for t in stats])
        ax.plot(x, mean, color=pair_color[p], lw=2, zorder=3)
        ax.scatter(x, mean, s=22, color=pair_color[p], edgecolor="white", linewidth=1.2, zorder=4)
        ax.annotate(f"{a} vs {b}", (x[-1], mean[-1]), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=6.8, color=INK)
    ax.set_xticks(x, ["hidden\nlayer 1", "hidden\nlayer 2", "action\noutput"])
    ax.set_xlim(-0.3, 3.4)
    ax.set_ylim(0, 1)
    ax.set_ylabel("similarity of twins (linear CKA)")
    ax.set_title("a  Twins differ from the first layer on", color=INK, pad=6)
    grid(ax)

    ax = axes[1]
    seeds = sorted(chim.seed.unique())
    rows = [("standard", "cold_day", "cold ground, standard"),
            ("richer", "collapses_per_1k_normal", "collapses, richer"),
            ("richer", "thirst_per_1k", "thirst collapses, richer"),
            ("poorer", "starved_per_1k", "hunger collapses, poorer"),
            ("richer", "meals_per_100", "meals, richer"),
            ("richer", "drinks_per_100", "drinks, richer")]

    def mean_of(name, world, m):
        d = chim[(chim.network == name) & (chim.world == world)].set_index("seed").loc[seeds, m]
        return d.mean()

    y = np.arange(len(rows))[::-1]
    for cut, marker, dx in [("cut1", "o", -0.12), ("cut2", "s", 0.12)]:
        for low, high, fill in [("scarce", "rich", True), ("rich", "scarce", False)]:
            pos = []
            for world, m, _ in rows:
                lo_v, hi_v = mean_of(f"intact {low}", world, m), mean_of(f"intact {high}", world, m)
                c = mean_of(f"{cut}: low {low}, high {high}", world, m)
                pos.append((c - hi_v) / (lo_v - hi_v))
            col = "#184f95" if cut == "cut1" else "#6da7ec"     # one hue, two steps: the cut goes deeper
            ax.scatter(pos, y + dx, marker=marker, s=20, zorder=4, linewidth=1.1,
                       facecolor=col if fill else "white", edgecolor=col)
    ax.axvline(0, color=INK2, lw=0.6)
    ax.axvline(1, color=INK2, lw=0.6)
    ax.set_yticks(y, [r[2] for r in rows])
    ax.set_xlim(-0.2, 1.45)
    ax.set_xticks([0, 0.5, 1], ["0\nlike the twin that\ngave the upper layers", "0.5", "1\nlike the twin that\ngave the lower layers"])
    ax.xaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_title("b  Chimeras act mostly like the donor of the lower layers", color=INK, pad=6)
    handles = [plt.Line2D([], [], ls="", marker="o", color="#184f95", ms=4.5, label="only first layer swapped"),
               plt.Line2D([], [], ls="", marker="s", color="#6da7ec", ms=4.5, label="both hidden layers swapped"),
               plt.Line2D([], [], ls="", marker="o", mfc="#52514e", mec="#52514e", ms=4.5, label="lower layers from scarce twin"),
               plt.Line2D([], [], ls="", marker="o", mfc="white", mec="#52514e", ms=4.5, label="lower layers from rich twin")]
    ax.legend(handles=handles, loc="lower right", bbox_to_anchor=(1.02, -0.50), ncol=2, frameon=False,
              handletextpad=0.3, columnspacing=1.0)
    fig.savefig(out / "fig2_trace.pdf", bbox_inches="tight")
    fig.savefig(out / "fig2_trace.png", bbox_inches="tight", dpi=200)
    plt.close(fig)


def figure3(L, seeds, out):
    order = ["hungry", "scarce", "unreliable", "adult", "rich"]
    labels = ["hungry\n0.00045", "scarce\n0.001", "unreliable\n0.001\n(waves)", "adult\n0.0028\n(droughts)", "rich\n0.02"]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), gridspec_kw={"wspace": 0.35})
    for ax, (phase, m, title, ylabel) in zip(axes, [
            ("frozen_home", "carry_sated", "a  Food carried while full, at home", "pieces carried"),
            ("frozen", "energy", "b  Energy kept in the standard adult world", "energy (0-1)")]):
        x = np.arange(len(order))
        for i, c in enumerate(order):
            v = [phase_mean(L[(c, s, "standard")], phase, m) for s in seeds]
            mean, lo, hi = ci(v)
            ax.scatter(np.full(len(v), i) + RNG.uniform(-0.18, 0.18, len(v)), v, s=6, color="#86b6ef",
                       linewidth=0, zorder=2)
            ax.vlines(i, lo, hi, color="#184f95", lw=1.4, zorder=3)
            ax.scatter([i], [mean], s=26, color="#184f95", edgecolor="white", linewidth=1.2, zorder=4)
        ax.set_xticks(x, labels)
        ax.set_xlabel("childhood: food growth per patch per step", color=INK2, labelpad=4)
        ax.set_ylabel(ylabel)
        ax.set_title(title, color=INK, pad=6)
        grid(ax)
    axes[1].set_ylim(0.72, 1.0)
    axes[0].set_ylim(bottom=-0.1)
    fig.savefig(out / "fig3_storing.pdf", bbox_inches="tight")
    fig.savefig(out / "fig3_storing.png", bbox_inches="tight", dpi=200)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--v1", default="runs_v1")
    p.add_argument("--rep", default="runs_v1_rep")
    p.add_argument("--exp2", default="runs_exp2")
    p.add_argument("--trace", default="trace_v1")
    p.add_argument("--out", default="figures")
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    L = {}
    L.update(lives(args.v1, range(100, 120), ["scarce", "rich", "adult"]))
    L.update(lives(args.rep, range(120, 140), ["scarce", "rich", "adult"]))
    E = lives(args.exp2, range(140, 170), ["scarce", "rich", "adult", "hungry", "unreliable"])
    L.update({k: v for k, v in E.items() if k[0] in ("scarce", "rich", "adult")})
    figure1(L, list(range(100, 170)), out)
    figure2(Path(args.trace), out)
    figure3(E, list(range(140, 170)), out)
    print(f"written to {out}")


if __name__ == "__main__":
    main()

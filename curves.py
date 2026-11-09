"""Learning curves for every life in runs/, one colour per childhood.

    python curves.py              draw once
    python curves.py --live       redraw every 20 seconds while lives are running
    python curves.py --save curves.png

Left half of each panel is childhood, then the dashed line marks the move to
the adult world. Thin lines are single seeds, the thick line is their mean.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

COLOURS = {"scarce": "#d62728", "rich": "#2ca02c", "none": "#7f7f7f"}
PANELS = [("wellbeing", "wellbeing (1 = all needs met)"),
          ("collapses_per_1k", "collapses per 1,000 steps"),
          ("carry_sated", "food carried while full"),
          ("cold_day", "share of daytime on cold ground"),
          ("drinks_per_100", "drinks per 100 steps"),
          ("meals_per_100", "meals per 100 steps")]


def load(folder):
    lives = {}
    for f in sorted(Path(folder).glob("*.csv")):
        d = pd.read_csv(f)
        if d.empty:
            continue
        child = d[d.phase == "child"]
        adult = d[d.phase == "adult"].copy()
        offset = child.step.max() if not child.empty else 0
        adult["step"] = adult.step + offset
        d = pd.concat([child, adult])
        d["step_m"] = d.step / 1e6
        moved = offset / 1e6 if (offset and not adult.empty) else None
        lives[f.stem] = (d, moved)
    return lives


def draw(fig, axes, folder, smooth):
    lives = load(folder)
    for ax in axes.flat:
        ax.clear()
    moves = []
    for name, (d, move) in lives.items():
        cond = name.rsplit("_s", 1)[0]
        for ax, (col, title) in zip(axes.flat, PANELS):
            if col in d:
                ax.plot(d.step_m, d[col].rolling(smooth, min_periods=1).mean(),
                        color=COLOURS.get(cond, "black"), lw=0.7, alpha=0.35)
        if move:
            moves.append(move)
    for cond, colour in COLOURS.items():
        runs = [d.set_index("step_m") for n, (d, _) in lives.items() if n.rsplit("_s", 1)[0] == cond]
        if not runs:
            continue
        for ax, (col, title) in zip(axes.flat, PANELS):
            cols = [r[col].rolling(smooth, min_periods=1).mean() for r in runs if col in r]
            if cols:
                m = pd.concat(cols, axis=1).mean(axis=1).sort_index()
                ax.plot(m.index, m.values, color=colour, lw=2, label=f"{cond} ({len(runs)})")
    for ax, (col, title) in zip(axes.flat, PANELS):
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("million steps", fontsize=8)
        for m in set(moves):
            ax.axvline(m, color="black", ls="--", lw=0.8)
    axes.flat[0].legend(fontsize=8)
    fig.tight_layout()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="runs")
    p.add_argument("--live", action="store_true")
    p.add_argument("--save")
    p.add_argument("--smooth", type=int, default=10)
    args = p.parse_args()
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    draw(fig, axes, args.runs, args.smooth)
    if args.save:
        fig.savefig(args.save, dpi=120)
        print("saved", args.save)
    elif args.live:
        plt.ion()
        plt.show()
        while plt.fignum_exists(fig.number):
            plt.pause(20)
            draw(fig, axes, args.runs, args.smooth)
            fig.canvas.draw_idle()
    else:
        plt.show()


if __name__ == "__main__":
    main()

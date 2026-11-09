"""The pre-registered analysis of the pilot (see PREREGISTRATION.md).

    python analyze.py                       seeds 0-9 in runs/
    python analyze.py --seeds 950 951       any other seeds, e.g. to test the script

Prints the primary test, then the secondary results, and writes everything to
analysis.json next to the runs.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon

PRIMARY = ["carry_sated", "cold_day"]
BLOCK = 200_000


def load(folder, seeds):
    lives = {}
    for cond in ["scarce", "rich", "none"]:
        for s in seeds:
            f = Path(folder) / f"{cond}_s{s}.csv"
            if f.exists():
                lives[(cond, s)] = pd.read_csv(f)
    return lives


def frozen_means(df, cols):
    d = df[df.phase == "frozen"]
    return {c: float(np.nanmean(d[c])) if d[c].notna().any() else float("nan") for c in cols}


def paired(scarce, rich, rng):
    diff = np.array(scarce) - np.array(rich)
    diff = diff[~np.isnan(diff)]
    out = {"n": int(len(diff)), "mean_difference": float(diff.mean()) if len(diff) else float("nan")}
    if len(diff) < 2 or np.allclose(diff, 0):
        return out | {"p": float("nan"), "ci95": [float("nan")] * 2, "rank_biserial": float("nan")}
    boots = rng.choice(diff, (10_000, len(diff))).mean(1)
    nz = diff[diff != 0]
    ranks = rankdata(np.abs(nz))
    r = (ranks[nz > 0].sum() - ranks[nz < 0].sum()) / ranks.sum()
    return out | {"p": float(wilcoxon(diff, alternative="two-sided").pvalue),
                  "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
                  "rank_biserial": float(r)}


def holm(ps):
    order = np.argsort(ps)
    adj = np.empty(len(ps))
    running = 0.0
    for k, i in enumerate(order):
        running = max(running, min(1.0, (len(ps) - k) * ps[i]))
        adj[i] = running
    return adj


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="runs")
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    args = p.parse_args()
    rng = np.random.default_rng(0)
    lives = load(args.runs, args.seeds)
    seeds = [s for s in args.seeds if ("scarce", s) in lives and ("rich", s) in lives]
    print(f"paired seeds: {seeds}")
    if not seeds:
        return

    cols = PRIMARY + ["carry_sated_normal", "carry_sated_drought", "cold_day_normal", "cold_day_drought",
                      "collapses_per_1k", "starved_per_1k", "thirst_per_1k", "froze_per_1k", "wellbeing"]
    table = pd.DataFrame([{"condition": c, "seed": s, **frozen_means(df, cols)} for (c, s), df in lives.items()])
    result = {"seeds": seeds, "primary": {}, "secondary": {}}

    print("\nPRIMARY (frozen check, scarce minus rich, paired by seed)")
    stats = {m: paired([table.query("condition=='scarce' and seed==@s")[m].iloc[0] for s in seeds],
                       [table.query("condition=='rich' and seed==@s")[m].iloc[0] for s in seeds], rng)
             for m in PRIMARY}
    adjusted = holm(np.array([stats[m]["p"] for m in PRIMARY]))
    for m, a in zip(PRIMARY, adjusted):
        stats[m]["p_holm"] = float(a)
        s = stats[m]
        print(f"  {m:12s} n={s['n']}  difference {s['mean_difference']:+.3f}  "
              f"95% CI [{s['ci95'][0]:+.3f}, {s['ci95'][1]:+.3f}]  p={s['p']:.4f}  p_holm={a:.4f}  r={s['rank_biserial']:+.2f}")
    result["primary"] = stats

    print("\nSECONDARY: frozen-check means by condition")
    means = table.groupby("condition")[cols].mean().round(3)
    print(means.to_string())
    result["secondary"]["frozen_means"] = means.to_dict()

    print("\nSECONDARY: scarce minus rich in each 200k-step block of adult life (mean over seeds)")
    blocks = []
    for s in seeds:
        for cond in ["scarce", "rich"]:
            d = lives[(cond, s)]
            d = d[d.phase == "adult"].copy()
            d["block"] = (d.step - 1) // BLOCK
            for b, g in d.groupby("block"):
                blocks.append({"seed": s, "condition": cond, "block": int(b),
                               **{m: float(np.nanmean(g[m])) for m in PRIMARY}})
    bt = pd.DataFrame(blocks)
    if not bt.empty:
        wide = bt.pivot_table(index=["seed", "block"], columns="condition", values=PRIMARY)
        course = pd.DataFrame({m: (wide[(m, "scarce")] - wide[(m, "rich")]).groupby("block").mean() for m in PRIMARY})
        course.index = [f"{(b + 1) * BLOCK // 1000}k" for b in course.index]
        print(course.round(3).to_string())
        result["secondary"]["difference_by_block"] = course.to_dict()

    out = Path(args.runs) / "analysis.json"
    out.write_text(json.dumps(result, indent=2, default=float))
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()

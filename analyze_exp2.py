"""The pre-registered analysis of experiment 2, the hungry and the unreliable childhood (see PREREGISTRATION_EXP2.md).

    .venv/bin/python analyze_exp2.py
    .venv/bin/python analyze_v1_rep.py --runs runs_exp2 --seeds $(seq 140 169)    # the second replication of version 1

Each prediction is a two-sided Wilcoxon signed-rank test on the paired differences (one pair per seed), with
Holm over the tests of its family. A prediction holds if it is significant after Holm and the mean difference
has the predicted sign. Writes runs_exp2/analysis_exp2.json.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze import holm, paired
from analyze_v1 import late_mean, phase_mean

CHILDHOODS = ["scarce", "hungry", "unreliable", "rich", "adult"]
BRANCHES = ["standard", "richer", "poorer"]

# (label, measure, phase, branch whose file holds the phase, x, y, predicted sign of x minus y)
FAMILIES = {
    "G: hunger, hungry - scarce": [
        ("food carried while full, at home", "carry_sated", "frozen_home", "standard", "hungry", "scarce", "+"),
        ("food carried while full, standard adult world", "carry_sated", "frozen", "standard", "hungry", "scarce", "+"),
        ("energy, standard adult world", "energy", "frozen", "standard", "hungry", "scarce", "+"),
    ],
    "H: unreliability at equal hardship, unreliable - hungry": [
        ("food carried while full, at home", "carry_sated", "frozen_home", "standard", "unreliable", "hungry", "+"),
        ("food carried while full, standard adult world", "carry_sated", "frozen", "standard", "unreliable", "hungry", "+"),
        ("energy, standard adult world", "energy", "frozen", "standard", "unreliable", "hungry", "+"),
    ],
    "I: unreliable - hungry, droughts of the standard adult world": [
        ("collapses during droughts", "collapses_per_1k_drought", "frozen", "standard", "unreliable", "hungry", "-"),
    ],
}


def load(root, seeds):
    lives = {}
    for c in CHILDHOODS:
        for s in seeds:
            for b in BRANCHES:
                f = Path(root) / f"{c}_s{s}_{b}.csv"
                if f.exists():
                    lives[(c, s, b)] = pd.read_csv(f)
    return lives


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="runs_exp2")
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(140, 170)))
    args = p.parse_args()
    rng = np.random.default_rng(0)
    lives = load(args.runs, args.seeds)
    print(f"{len(lives)} life files found in {args.runs}")
    result = {}
    for name, tests in FAMILIES.items():
        stats = []
        for label, m, phase, b, x, y, sign in tests:
            ok = [s for s in args.seeds if (x, s, b) in lives and (y, s, b) in lives]
            st = paired([phase_mean(lives[(x, s, b)], phase, m) for s in ok],
                        [phase_mean(lives[(y, s, b)], phase, m) for s in ok], rng)
            stats.append({"test": f"{x} - {y}: {label}", "measure": m, "phase": phase, "predicted": sign, **st})
        for t, a in zip(stats, holm(np.array([t["p"] for t in stats]))):
            t["p_holm"] = float(a)
            t["verdict"] = "holds" if a < 0.05 and (t["mean_difference"] > 0) == (t["predicted"] == "+") else "does not hold"
        print(f"\nFAMILY {name} (Holm over {len(stats)} tests)")
        for t in stats:
            print(f"  {t['test']:64s} predicted {t['predicted']}  n={t['n']:2d}  {t['mean_difference']:+.3f}  "
                  f"[{t['ci95'][0]:+.3f}, {t['ci95'][1]:+.3f}]  p={t['p']:.5f}  p_holm={t['p_holm']:.5f}  "
                  f"r={t['rank_biserial']:+.2f}  {t['verdict']}")
        result[name] = stats

    print("\nDESCRIPTIVE: means by childhood")
    rows = []
    for c in CHILDHOODS:
        for s in args.seeds:
            if (c, s, "standard") not in lives:
                continue
            d = lives[(c, s, "standard")]
            child = d[d.phase == "child"]
            row = {"childhood": c, "seed": s}
            for m in ["carry_sated", "energy", "collapses_per_1k", "collapses_per_1k_drought", "cold_day", "wellbeing"]:
                row[f"child_end {m}"] = float(np.nanmean(child.tail(max(1, len(child) // 10))[m])) if len(child) else np.nan
                row[f"home {m}"] = phase_mean(d, "frozen_home", m)
                for b in BRANCHES:
                    if (c, s, b) in lives:
                        row[f"frozen {b} {m}"] = phase_mean(lives[(c, s, b)], "frozen", m)
                row[f"late standard {m}"] = late_mean(d, m)
            rows.append(row)
    t = pd.DataFrame(rows).groupby("childhood").mean(numeric_only=True).drop(columns="seed").reindex(CHILDHOODS)
    with pd.option_context("display.max_rows", 200, "display.width", 200):
        print(t.T.round(3).to_string())
    result["descriptive"] = t.to_dict()
    # food carried is bimodal (most agents carry almost nothing, some carry a lot), so also the share of carriers
    d = pd.DataFrame(rows)
    share = d.assign(carrier=d["home carry_sated"] > 0.5).groupby("childhood").carrier.mean().reindex(CHILDHOODS)
    print("\nDESCRIPTIVE: share of agents carrying more than half a piece while full, at home")
    print(share.round(2).to_string())
    result["share_of_carriers_at_home"] = share.to_dict()
    out = Path(args.runs) / "analysis_exp2.json"
    out.write_text(json.dumps(result, indent=2, default=float))
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()

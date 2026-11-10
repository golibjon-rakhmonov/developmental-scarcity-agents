"""The pre-registered analysis of experiment 3, the varied childhood (see PREREGISTRATION_EXP3.md).

    .venv/bin/python analyze_exp3.py
    .venv/bin/python analyze_v1_rep.py --runs runs_exp3 --seeds $(seq 190 219)    # the third replication of version 1

Directional predictions: two-sided Wilcoxon signed-rank test on the paired differences (one pair per seed),
Holm over the tests of the family; a prediction holds if it is significant after Holm and the mean difference
has the predicted sign. Predictions of no difference hold only if the whole bootstrap 95% confidence interval
lies inside the margin. Writes runs_exp3/analysis_exp3.json.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze import holm, paired
from analyze_v1 import late_mean, phase_mean

CHILDHOODS = ["scarce", "rich", "adult", "varied"]
BRANCHES = ["standard", "richer", "poorer"]
MARGIN = 0.15      # collapses per 1,000 steps, as in the replications of version 1
NORMAL = "collapses_per_1k_normal"
ALL = "collapses_per_1k"

# (label, x, y, branch or "mean of the three worlds", measure, predicted sign of x minus y: "-", "+" or "0")
FAMILIES = {
    "J: variety removes the mismatch": [
        ("varied - scarce, richer world outside droughts", "varied", "scarce", "richer", NORMAL, "-"),
        ("varied - rich, poorer world", "varied", "rich", "poorer", ALL, "-"),
        ("varied - rich, standard world", "varied", "rich", "standard", ALL, "-"),
    ],
    "K: no price for being a generalist": [
        ("varied - rich, richer world outside droughts", "varied", "rich", "richer", NORMAL, "0"),
        ("varied - scarce, poorer world", "varied", "scarce", "poorer", ALL, "0"),
        ("varied - adult, standard world", "varied", "adult", "standard", ALL, "0"),
    ],
    "L: better across worlds (mean of the three adult worlds)": [
        ("varied - scarce", "varied", "scarce", "mean", ALL, "-"),
        ("varied - rich", "varied", "rich", "mean", ALL, "-"),
        ("varied - adult", "varied", "adult", "mean", ALL, "-"),
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


def value(lives, c, s, b, m):
    """A childhood's frozen-check value in one adult world, or its mean over the three adult worlds."""
    if b == "mean":
        return float(np.mean([phase_mean(lives[(c, s, w)], "frozen", m) for w in BRANCHES]))
    return phase_mean(lives[(c, s, b)], "frozen", m)


def verdict(t):
    if t["predicted"] == "0":
        inside = -MARGIN < t["ci95"][0] and t["ci95"][1] < MARGIN
        return f"holds (95% CI inside +-{MARGIN})" if inside else f"DOES NOT HOLD (95% CI leaves +-{MARGIN})"
    ok = t["p_holm"] < 0.05 and ((t["mean_difference"] < 0) == (t["predicted"] == "-"))
    return "holds" if ok else "does not hold"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="runs_exp3")
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(190, 220)))
    args = p.parse_args()
    rng = np.random.default_rng(0)
    lives = load(args.runs, args.seeds)
    print(f"{len(lives)} life files found in {args.runs}")
    complete = lambda c, s: all((c, s, b) in lives for b in BRANCHES)
    result = {}
    for name, tests in FAMILIES.items():
        stats = []
        for label, x, y, b, m, sign in tests:
            ok = [s for s in args.seeds if complete(x, s) and complete(y, s)]
            st = paired([value(lives, x, s, b, m) for s in ok], [value(lives, y, s, b, m) for s in ok], rng)
            stats.append({"test": label, "measure": m, "predicted": sign, **st})
        directional = [t for t in stats if t["predicted"] != "0"]
        for t, a in zip(directional, holm(np.array([t["p"] for t in directional]))):
            t["p_holm"] = float(a)
        print(f"\nFAMILY {name} (Holm over its {len(directional)} directional tests)")
        for t in stats:
            t.setdefault("p_holm", float("nan"))
            t["verdict"] = verdict(t)
            print(f"  {t['test']:48s} predicted {t['predicted']}  n={t['n']:2d}  {t['mean_difference']:+.3f}  "
                  f"[{t['ci95'][0]:+.3f}, {t['ci95'][1]:+.3f}]  p={t['p']:.5f}  p_holm={t['p_holm']:.5f}  "
                  f"r={t['rank_biserial']:+.2f}  {t['verdict']}")
        result[name] = stats

    print("\nDESCRIPTIVE: means by childhood")
    rows = []
    for c in CHILDHOODS:
        for s in args.seeds:
            if not complete(c, s):
                continue
            d = lives[(c, s, "standard")]
            child = d[d.phase == "child"]
            row = {"childhood": c, "seed": s}
            for m in ["collapses_per_1k", "wellbeing", "energy", "cold_day", "carry_sated"]:
                row[f"child_end {m}"] = float(np.nanmean(child.tail(max(1, len(child) // 10))[m]))
                row[f"home {m}"] = phase_mean(d, "frozen_home", m)
                for b in BRANCHES:
                    row[f"frozen {b} {m}"] = phase_mean(lives[(c, s, b)], "frozen", m)
                    row[f"late {b} {m}"] = late_mean(lives[(c, s, b)], m)
            for m in ["collapses_per_1k_normal", "collapses_per_1k_drought", "cold_day_normal", "cold_day_drought"]:
                row[f"home {m}"] = phase_mean(d, "frozen_home", m)
            for b in BRANCHES:
                for m in ["collapses_per_1k_normal", "starved_per_1k", "thirst_per_1k", "froze_per_1k",
                          "meals_per_100", "drinks_per_100"]:
                    row[f"frozen {b} {m}"] = phase_mean(lives[(c, s, b)], "frozen", m)
            per_world = [row[f"frozen {b} collapses_per_1k"] for b in BRANCHES]
            row["frozen mean of worlds collapses_per_1k"] = float(np.mean(per_world))
            row["frozen worst world collapses_per_1k"] = float(np.max(per_world))
            rows.append(row)
    t = pd.DataFrame(rows).groupby("childhood").mean(numeric_only=True).drop(columns="seed").reindex(CHILDHOODS)
    with pd.option_context("display.max_rows", 200, "display.width", 200):
        print(t.T.round(3).to_string())
    result["descriptive"] = t.to_dict()
    out = Path(args.runs) / "analysis_exp3.json"
    out.write_text(json.dumps(result, indent=2, default=float))
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()

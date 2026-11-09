"""The pre-registered analysis of the version 1 replication (see PREREGISTRATION_V1_REPLICATION.md).

    .venv/bin/python analyze_v1_rep.py
    .venv/bin/python analyze_v1_rep.py --runs runs_v1 --seeds 100 101 102     # to test the script

Directional predictions: two-sided Wilcoxon signed-rank test on paired differences, Holm over the
directional tests of each family; a prediction replicates if it is significant after Holm and the
mean difference has the predicted sign. Predictions of no difference: the difference holds as absent
if its bootstrap 95% confidence interval lies entirely inside the margin set below (an equivalence
check, not a missing significance). Writes runs_v1_rep/analysis.json.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from analyze import holm, paired
from analyze_v1 import late_mean, load, phase_mean

MEASURES = ["carry_sated", "cold_day", "collapses_per_1k"]
# margins for "no difference", in collapses per 1,000 steps: a third of the smallest collapse
# difference version 1 found in the frozen check (rich - adult, 0.40), and half of the level of
# late-life collapses (about 0.10)
MARGIN = {"frozen": 0.15, "late": 0.05}
PHASE = {"normal": "_normal", "drought": "_drought", "all": ""}

# (label, measure, x, y, branch, value, predicted sign of x minus y: "-", "+" or "0" for no difference)
FAMILIES = {
    "A: frozen check, standard adult world": [
        ("scarce - rich", "carry_sated", "scarce", "rich", "standard", "frozen", "-"),
        ("scarce - rich", "cold_day", "scarce", "rich", "standard", "frozen", "+"),
        ("scarce - rich", "collapses_per_1k", "scarce", "rich", "standard", "frozen", "-"),
        ("rich - adult", "cold_day", "rich", "adult", "standard", "frozen", "-"),
        ("rich - adult", "collapses_per_1k", "rich", "adult", "standard", "frozen", "+"),
        ("scarce - adult", "collapses_per_1k", "scarce", "adult", "standard", "frozen", "0"),
    ],
    "B: scarce - rich, frozen check, mirror adult worlds": [
        (f"scarce - rich, {b} {ph}", m + PHASE[ph], "scarce", "rich", b, "frozen", sign)
        for b, ph, signs in [("richer", "normal", "-++"), ("richer", "drought", "-+-"), ("poorer", "all", "-+-")]
        for m, sign in zip(MEASURES, signs)
    ],
    "C: collapses in the last 10% of adult life, standard adult world": [
        ("scarce - rich", "collapses_per_1k", "scarce", "rich", "standard", "late", "0"),
        ("scarce - adult", "collapses_per_1k", "scarce", "adult", "standard", "late", "0"),
        ("rich - adult", "collapses_per_1k", "rich", "adult", "standard", "late", "0"),
    ],
    "D: what the collapses are made of, frozen check, mirror adult worlds": [
        ("scarce - rich, richer", "thirst_per_1k", "scarce", "rich", "richer", "frozen", "+"),
        ("scarce - rich, richer", "meals_per_100", "scarce", "rich", "richer", "frozen", "+"),
        ("scarce - rich, richer", "drinks_per_100", "scarce", "rich", "richer", "frozen", "-"),
        ("scarce - rich, poorer", "starved_per_1k", "scarce", "rich", "poorer", "frozen", "-"),
    ],
    "E: each childhood against the childhood in the adult world, frozen check, mirror adult worlds": [
        ("scarce - adult, richer normal", "collapses_per_1k_normal", "scarce", "adult", "richer", "frozen", "+"),
        ("rich - adult, richer normal", "collapses_per_1k_normal", "rich", "adult", "richer", "frozen", "-"),
        ("rich - adult, poorer", "collapses_per_1k", "rich", "adult", "poorer", "frozen", "+"),
        ("scarce - adult, poorer", "collapses_per_1k", "scarce", "adult", "poorer", "frozen", "0"),
    ],
}


def value(d, kind, m):
    return phase_mean(d, "frozen", m) if kind == "frozen" else late_mean(d, m)


def verdict(t):
    if t["predicted"] == "0":
        m = MARGIN[t["kind"]]
        inside = -m < t["ci95"][0] and t["ci95"][1] < m
        return f"holds (95% CI inside +-{m})" if inside else f"DOES NOT HOLD (95% CI leaves +-{m})"
    ok = t["p_holm"] < 0.05 and ((t["mean_difference"] < 0) == (t["predicted"] == "-"))
    return "replicates" if ok else "does not replicate"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="runs_v1_rep")
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(120, 140)))
    args = p.parse_args()
    rng = np.random.default_rng(0)
    lives = load(args.runs, args.seeds)
    print(f"{len(lives)} life files found in {args.runs}")
    result = {}
    for name, tests in FAMILIES.items():
        stats = []
        for label, m, x, y, b, kind, sign in tests:
            ok = [s for s in args.seeds if (x, s, b) in lives and (y, s, b) in lives]
            st = paired([value(lives[(x, s, b)], kind, m) for s in ok], [value(lives[(y, s, b)], kind, m) for s in ok], rng)
            stats.append({"test": label, "measure": m, "kind": kind, "predicted": sign, **st})
        directional = [t for t in stats if t["predicted"] != "0"]
        for t, a in zip(directional, holm(np.array([t["p"] for t in directional]))):
            t["p_holm"] = float(a)
        print(f"\nFAMILY {name} (Holm over its {len(directional)} directional tests)")
        for t in stats:
            t.setdefault("p_holm", float("nan"))
            t["verdict"] = verdict(t)
            print(f"  {t['test']:30s} {t['measure']:24s} predicted {t['predicted']}  n={t['n']:2d}  "
                  f"{t['mean_difference']:+.3f}  [{t['ci95'][0]:+.3f}, {t['ci95'][1]:+.3f}]  "
                  f"p={t['p']:.5f}  p_holm={t['p_holm']:.5f}  r={t['rank_biserial']:+.2f}  {t['verdict']}")
        result[name] = stats
    out = Path(args.runs) / "analysis.json"
    out.write_text(json.dumps(result, indent=2, default=float))
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()

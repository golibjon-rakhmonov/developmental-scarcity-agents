"""The pre-registered analysis of version 1 (see PREREGISTRATION_V1.md).

    .venv/bin/python analyze_v1.py
    .venv/bin/python analyze_v1.py --runs runs_v1_test --seeds 900 901      # to test the script

Prints the three families of tests and the descriptive checks, writes runs_v1/analysis.json.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze import holm, paired

MEASURES = ["carry_sated", "cold_day", "collapses_per_1k"]
CHILDHOODS = ["scarce", "rich", "adult", "none"]
BRANCHES = ["standard", "richer", "poorer"]


def load(root, seeds):
    lives = {}
    for c in CHILDHOODS:
        for s in seeds:
            for b in BRANCHES:
                f = Path(root) / f"{c}_s{s}_{b}.csv"
                if f.exists():
                    lives[(c, s, b)] = pd.read_csv(f)
    return lives


def phase_mean(d, phase, m):
    v = d.loc[d.phase == phase, m]
    return float(np.nanmean(v)) if v.notna().any() else float("nan")


def late_mean(d, m):
    a = d[d.phase == "adult"]
    return float(np.nanmean(a.tail(max(1, len(a) // 10))[m]))


def family(name, tests, lives, seeds, rng, value):
    """tests: list of (label, measure, childhood x, childhood y, branch). Paired x minus y, Holm over the family."""
    stats = []
    for label, m, x, y, b in tests:
        ok = [s for s in seeds if (x, s, b) in lives and (y, s, b) in lives]
        st = paired([value(lives[(x, s, b)], m) for s in ok], [value(lives[(y, s, b)], m) for s in ok], rng)
        stats.append({"test": label, "measure": m, "x": x, "y": y, "branch": b, **st})
    adj = holm(np.array([t["p"] for t in stats]))
    print(f"\n{name} (Holm over {len(stats)} tests)")
    for t, a in zip(stats, adj):
        t["p_holm"] = float(a)
        print(f"  {t['test']:30s} {t['measure']:24s} n={t['n']:2d}  {t['mean_difference']:+.3f}  "
              f"[{t['ci95'][0]:+.3f}, {t['ci95'][1]:+.3f}]  p={t['p']:.5f}  p_holm={a:.5f}  r={t['rank_biserial']:+.2f}")
    return stats


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="runs_v1")
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(100, 120)))
    args = p.parse_args()
    rng = np.random.default_rng(0)
    lives = load(args.runs, args.seeds)
    print(f"{len(lives)} life files found in {args.runs}")
    frozen = lambda d, m: phase_mean(d, "frozen", m)
    result = {}

    contrasts = [("scarce", "rich"), ("scarce", "adult"), ("rich", "adult")]
    result["A_frozen_standard"] = family(
        "FAMILY A: frozen check in the standard adult world", [
            (f"{x} - {y}", m, x, y, "standard") for x, y in contrasts for m in MEASURES],
        lives, args.seeds, rng, frozen)
    # the richer world is richer than both childhoods only outside droughts, so its two phases are
    # tested apart; the poorer world is poorer than both in both phases
    phase_cols = {"normal": ["carry_sated_normal", "cold_day_normal", "collapses_per_1k_normal"],
                  "drought": ["carry_sated_drought", "cold_day_drought", "collapses_per_1k_drought"],
                  "all": MEASURES}
    result["B_frozen_mirror"] = family(
        "FAMILY B: scarce - rich, frozen check in the mirror adult worlds", [
            (f"scarce - rich, {b} {ph}", m, "scarce", "rich", b)
            for b, ph in [("richer", "normal"), ("richer", "drought"), ("poorer", "all")] for m in phase_cols[ph]],
        lives, args.seeds, rng, frozen)
    result["C_late_collapses"] = family(
        "FAMILY C: collapses in the last 10% of adult life, standard adult world", [
            (f"{x} - {y}", "collapses_per_1k", x, y, "standard") for x, y in contrasts],
        lives, args.seeds, rng, late_mean)

    print("\nDESCRIPTIVE: means by childhood (standard branch)")
    rows = []
    for (c, s, b), d in lives.items():
        if b != "standard":
            continue
        child = d[d.phase == "child"]
        row = {"childhood": c, "seed": s}
        for m in MEASURES + ["wellbeing", "energy"]:
            row[f"child_end_{m}"] = float(np.nanmean(child.tail(max(1, len(child) // 10))[m])) if len(child) else float("nan")
            row[f"home_{m}"] = phase_mean(d, "frozen_home", m)
            row[f"frozen_{m}"] = phase_mean(d, "frozen", m)
            row[f"late_{m}"] = late_mean(d, m)
        rows.append(row)
    t = pd.DataFrame(rows).groupby("childhood").mean(numeric_only=True).drop(columns="seed")
    print(t.T.round(3).to_string())
    result["descriptive"] = t.to_dict()

    print("\nDESCRIPTIVE: map check. Collapses per 1,000 steps at the end of childhood (maps it lived on)")
    print("and in the home check (same kind of world, maps never seen). Version 0 showed 0.04 -> 0.55 (scarce)")
    print("and 0.12 -> 1.61 (rich); version 1 should show little change.")
    print(t[["child_end_collapses_per_1k", "home_collapses_per_1k", "frozen_collapses_per_1k"]].round(3).to_string())

    out = Path(args.runs) / "analysis.json"
    out.write_text(json.dumps(result, indent=2, default=float))
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()

"""The pre-registered analysis of the robustness check (see PREREGISTRATION_ROBUSTNESS.md).

    .venv/bin/python analyze_robust.py

Prints the replication, the four variations and the layer-reset comparison,
and writes runs_robust/analysis.json.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze import frozen_means, holm, paired

MEASURES = ["carry_sated", "cold_day", "collapses_per_1k"]
PREDICTED = {"carry_sated": "-", "cold_day": "+", "collapses_per_1k": "-"}   # sign of scarce minus rich
VARIATIONS = ["long_childhood", "short_childhood", "scarcer", "less_scarce"]
BLOCK = 200_000


def load(folder):
    lives = {}
    for f in Path(folder).glob("*.csv"):
        cond, seed = f.stem.rsplit("_s", 1)
        lives[(cond, int(seed))] = pd.read_csv(f)
    return lives


def compare(lives, rng):
    seeds = sorted(s for c, s in lives if c == "scarce" and ("rich", s) in lives)
    means = {k: frozen_means(d, MEASURES) for k, d in lives.items()}
    return seeds, {m: paired([means[("scarce", s)][m] for s in seeds], [means[("rich", s)][m] for s in seeds], rng)
                   for m in MEASURES}


def line(name, m, s, p_adj):
    ok = (s["mean_difference"] < 0) == (PREDICTED[m] == "-") and p_adj < 0.05
    return (f"  {name:16s} {m:17s} n={s['n']:2d}  {s['mean_difference']:+.3f}  "
            f"[{s['ci95'][0]:+.3f}, {s['ci95'][1]:+.3f}]  p={s['p']:.4f}  p_holm={p_adj:.4f}  "
            f"r={s['rank_biserial']:+.2f}  {'as predicted' if ok else 'NOT as predicted'}")


def adult_blocks(lives, seeds, measure, until=1_000_000):
    rows = []
    for s in seeds:
        for c in ["scarce", "rich"]:
            d = lives[(c, s)]
            d = d[(d.phase == "adult") & (d.step <= until)].copy()
            d["block"] = (d.step - 1) // BLOCK
            for b, g in d.groupby("block"):
                rows.append({"seed": s, "c": c, "block": int(b), "v": float(np.nanmean(g[measure]))})
    t = pd.DataFrame(rows).pivot_table(index=["seed", "block"], columns="c", values="v")
    return (t["scarce"] - t["rich"]).groupby("block").mean()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="runs_robust")
    args = p.parse_args()
    root = Path(args.runs)
    rng = np.random.default_rng(0)
    result = {}

    print("REPLICATION (seeds 10-29, frozen check, scarce minus rich)")
    rep = load(root / "replication")
    seeds, stats = compare(rep, rng)
    adj = holm(np.array([stats[m]["p"] for m in MEASURES]))
    for m, a in zip(MEASURES, adj):
        stats[m]["p_holm"] = float(a)
        print(line("replication", m, stats[m], a))
    result["replication"] = {"seeds": seeds, **stats}

    print("\nVARIATIONS (seeds 10-19, Holm over all twelve tests)")
    var_stats = {}
    for v in VARIATIONS:
        if (root / v).exists():
            var_stats[v] = compare(load(root / v), rng)[1]
    flat = [(v, m) for v in var_stats for m in MEASURES]
    adj = holm(np.array([var_stats[v][m]["p"] for v, m in flat])) if flat else []
    for (v, m), a in zip(flat, adj):
        var_stats[v][m]["p_holm"] = float(a)
        print(line(v, m, var_stats[v][m], a))
    result["variations"] = var_stats

    print("\nSECONDARY: does the resilience last? Collapses in the last 10% of adult life, replication seeds")
    late = {}
    for (c, s), d in rep.items():
        a = d[d.phase == "adult"]
        late[(c, s)] = float(a.tail(max(1, len(a) // 10)).collapses_per_1k.mean())
    s_late = paired([late[("scarce", s)] for s in seeds], [late[("rich", s)] for s in seeds], rng)
    print(f"  scarce minus rich {s_late['mean_difference']:+.3f}  [{s_late['ci95'][0]:+.3f}, {s_late['ci95'][1]:+.3f}]  p={s_late['p']:.4f}")
    result["late_collapses"] = s_late

    if (root / "reset").exists():
        print("\nEXPLORATORY: layer reset at the move. Scarce minus rich in the first 1M adult steps, by 200k block")
        res = load(root / "reset")
        rs = sorted(s for c, s in res if c == "scarce" and ("rich", s) in res)
        both = [s for s in rs if ("scarce", s) in rep and ("rich", s) in rep]
        table = {}
        for m in MEASURES:
            table[f"{m} (no reset)"] = adult_blocks(rep, both, m)
            table[f"{m} (reset)"] = adult_blocks(res, both, m)
        t = pd.DataFrame(table)
        t.index = [f"{(b + 1) * BLOCK // 1000}k" for b in t.index]
        print(t.round(3).to_string())
        result["reset"] = {"seeds": both, "by_block": t.to_dict()}

    out = root / "analysis.json"
    out.write_text(json.dumps(result, indent=2, default=float))
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()

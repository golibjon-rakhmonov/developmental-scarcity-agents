"""Exploratory analyses pooled over the registered runs. NOT pre-registered.

Written on 11 November 2026, after every run, while the paper draft was checked. It pools the 100 twin sets
of version 1 and its three replications (seeds 100-169 and 190-219, in runs_v1, runs_v1_rep, runs_exp2 and
runs_exp3) and prints the pooled and descriptive numbers the paper gives. Nothing here tests a registered
prediction; the registered analyses are analyze_v1.py, analyze_v1_rep.py, analyze_exp2.py and analyze_exp3.py.

    .venv/bin/python analyze_pooled.py > exploratory/pooled.txt

Each line is a paired difference over seeds: mean, bootstrap 95% interval (10,000 resamples), two-sided
Wilcoxon p, and the number of pairs below and above zero.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from analyze import paired
from analyze_v1 import late_mean, phase_mean

SETS = {"version 1": ("runs_v1", range(100, 120)), "replication 1": ("runs_v1_rep", range(120, 140)),
        "replication 2": ("runs_exp2", range(140, 170)), "replication 3": ("runs_exp3", range(190, 220)),
        "64 units": ("runs_net64", range(170, 190)), "256 units": ("runs_net256", range(170, 190)),
        "memory": ("runs_memory", range(220, 240))}
POOL = ["version 1", "replication 1", "replication 2", "replication 3"]
WORLDS = ["poorer", "standard", "richer"]
ALL, NORMAL, DROUGHT = "collapses_per_1k", "collapses_per_1k_normal", "collapses_per_1k_drought"
rng = np.random.default_rng(0)
_lives = {}


def life(root, childhood, seed, world):
    key = (root, childhood, seed, world)
    if key not in _lives:
        _lives[key] = pd.read_csv(Path(root) / f"{childhood}_s{seed}_{world}.csv")
    return _lives[key]


def value(root, childhood, seed, world, measure, when):
    """when: 'frozen' (right after the move), 'frozen_home', 'late' (last 10% of adult life) or 'child_end'."""
    d = life(root, childhood, seed, world)
    if when == "late":
        return late_mean(d, measure)
    if when == "child_end":
        child = d[d.phase == "child"]
        return float(np.nanmean(child.tail(max(1, len(child) // 10))[measure]))
    return phase_mean(d, when, measure)


def values(sets, childhood, world, measure, when):
    return np.array([value(SETS[k][0], childhood, s, world, measure, when) for k in sets for s in SETS[k][1]])


def line(label, sets, x, y, world, measure, when):
    a, b = values(sets, x, world, measure, when), values(sets, y, world, measure, when)
    t = paired(a, b, rng)
    d = a - b
    print(f"  {label:58s} n={t['n']:3d}  {t['mean_difference']:+.4f}  [{t['ci95'][0]:+.3f}, {t['ci95'][1]:+.3f}]  "
          f"p={t['p']:.4f}  below zero {int((d < 0).sum())}, above {int((d > 0).sum())}")


def means(label, sets, childhoods, world, measure, when, digits=3):
    print(f"  {label}: " + ", ".join(f"{c} {values(sets, c, world, measure, when).mean():.{digits}f}" for c in childhoods))


print("1. Scarce minus adult, collapses per 1,000 steps right after the move, 100 twin sets")
for w in WORLDS:
    for m, name in [(ALL, "whole check"), (NORMAL, "outside droughts"), (DROUGHT, "during droughts")]:
        line(f"{w} world, {name}", POOL, "scarce", "adult", w, m, "frozen")
    means(f"{w} world, levels", POOL, ["scarce", "adult", "rich"], w, ALL, "frozen")
print("   standard world, whole check, by set")
for k in SETS:
    line(k, [k], "scarce", "adult", "standard", ALL, "frozen")

print("\n2. Standard world right after the move, 100 twin sets (the sizes the end of adult life is compared with)")
line("rich - adult", POOL, "rich", "adult", "standard", ALL, "frozen")
line("scarce - rich", POOL, "scarce", "rich", "standard", ALL, "frozen")

print("\n3. Collapses at the end of adult life (last 10%), 100 twin sets")
for w in WORLDS:
    for x, y in [("scarce", "adult"), ("rich", "adult"), ("scarce", "rich")]:
        line(f"{w} world, {x} - {y}", POOL, x, y, w, ALL, "late")
    means(f"{w} world, levels", POOL, ["scarce", "adult", "rich"], w, ALL, "late")

print("\n4. Share of daytime on cold ground, standard world, 100 twin sets")
for x, y in [("scarce", "rich"), ("scarce", "adult"), ("rich", "adult")]:
    for when, name in [("frozen", "right after the move"), ("late", "end of adult life")]:
        line(f"{x} - {y}, {name}", POOL, x, y, "standard", "cold_day", when)
means("levels at the end of adult life", POOL, ["scarce", "adult", "rich"], "standard", "cold_day", "late", 4)
print("   rich agents spent the least daytime on cold ground of the three childhoods (cases of 300):")
for m in ["cold_day_normal", "cold_day"]:
    no = [(s, w) for k in POOL for s in SETS[k][1] for w in WORLDS
          if not value(SETS[k][0], "rich", s, w, m, "frozen")
          < min(value(SETS[k][0], c, s, w, m, "frozen") for c in ["scarce", "adult"])]
    print(f"     {m}: {300 - len(no)} of 300; exceptions (seed, world): {no}")
    for c in ["scarce", "adult", "rich"]:
        print(f"       {c}: " + ", ".join(f"{w} {100 * values(POOL, c, w, m, 'frozen').mean():.1f}%" for w in WORLDS))

print("\n5. Food carried while full")
for c in ["scarce", "rich", "adult"]:
    v = values(POOL, c, "standard", "carry_sated", "frozen_home")
    print(f"  at home, {c}: mean {v.mean():.2f}, more than half a piece {int((v > 0.5).sum())} of {len(v)}; "
          f"richer world right after the move {values(POOL, c, 'richer', 'carry_sated', 'frozen').mean():.2f}; "
          f"standard world at the end of adult life {values(POOL, c, 'standard', 'carry_sated', 'late').mean():.2f}")
line("scarce - adult, standard world, end of adult life", POOL, "scarce", "adult", "standard", "carry_sated", "late")
for c in ["hungry", "unreliable"]:
    v = values(["replication 2"], c, "standard", "carry_sated", "frozen_home")
    print(f"  experiment 2, {c}: at home mean {v.mean():.3f}, more than half a piece {int((v > 0.5).sum())} of {len(v)}; "
          "largest single agent right after the move: "
          + ", ".join(f"{w} {values(['replication 2'], c, w, 'carry_sated', 'frozen').max():.3f}" for w in WORLDS))
v = values(["replication 3"], "varied", "standard", "carry_sated", "frozen_home")
print(f"  experiment 3, varied: at home mean {v.mean():.3f}, more than half a piece {int((v > 0.5).sum())} of {len(v)}")
v = values(["memory"], "rich", "standard", "carry_sated", "frozen_home")
print(f"  memory, rich: at home mean {v.mean():.2f}, more than half a piece {int((v > 0.5).sum())} of {len(v)}")

print("\n6. Agents with no childhood (version 1 and replication 1), collapses per 1,000 steps, standard world")
for k in ["version 1", "replication 1"]:
    print(f"  {k}: right after the move {values([k], 'none', 'standard', ALL, 'frozen').mean():.2f}, "
          f"end of adult life {values([k], 'none', 'standard', ALL, 'late').mean():.3f}; the three childhoods at the end: "
          + ", ".join(f"{c} {values([k], c, 'standard', ALL, 'late').mean():.3f}" for c in ["scarce", "rich", "adult"]))

print("\n7. Scarce minus rich in the richer world, by drought phase and set (the drought rows are registered, family B)")
for m, name in [(NORMAL, "outside droughts"), (DROUGHT, "during droughts")]:
    line(f"100 twin sets, {name}", POOL, "scarce", "rich", "richer", m, "frozen")
    for k in SETS:
        line(f"{k}, {name}", [k], "scarce", "rich", "richer", m, "frozen")

print("\n8. Experiment 3: comparisons that were not registered (whole frozen check)")
E3 = ["replication 3"]
for w in WORLDS:
    for y in ["adult", "scarce", "rich"]:
        line(f"varied - {y}, {w} world", E3, "varied", y, w, ALL, "frozen")
kinds = ["varied", "scarce", "adult", "rich"]
worst = {c: np.max([values(E3, c, w, ALL, "frozen") for w in WORLDS], axis=0).mean() for c in kinds}
print("  worst of the three worlds (mean over seeds): " + ", ".join(f"{c} {worst[c]:.2f}" for c in kinds))
for w in WORLDS:
    means(f"end of adult life, {w} world", E3, kinds, w, ALL, "late")
print(f"  varied, food carried while full: richer world right after the move "
      f"{values(E3, 'varied', 'richer', 'carry_sated', 'frozen').mean():.3f} "
      f"(largest {values(E3, 'varied', 'richer', 'carry_sated', 'frozen').max():.2f}), "
      f"richer world at the end of adult life {values(E3, 'varied', 'richer', 'carry_sated', 'late').mean():.3f}")

print("\n9. With and without memory (different seeds, so descriptive)")
for label, x, y, w, m in [("scarce - rich, richer world outside droughts", "scarce", "rich", "richer", NORMAL),
                          ("scarce - rich, poorer world", "scarce", "rich", "poorer", ALL),
                          ("scarce - rich, standard world", "scarce", "rich", "standard", ALL),
                          ("rich - adult, standard world", "rich", "adult", "standard", ALL),
                          ("scarce - rich, cold ground, standard world", "scarce", "rich", "standard", "cold_day"),
                          ("scarce - rich, food carried, standard world", "scarce", "rich", "standard", "carry_sated")]:
    a = values(["memory"], x, w, m, "frozen") - values(["memory"], y, w, m, "frozen")
    b = values(POOL, x, w, m, "frozen") - values(POOL, y, w, m, "frozen")
    print(f"  {label:48s} memory {a.mean():+.3f}   no memory {b.mean():+.3f}   ratio {a.mean() / b.mean():.2f}")

print("\n10. How hard each childhood was, and how far the adult-world twin had learned")
for c in ["scarce", "unreliable", "hungry"]:
    print(f"  experiment 2, {c}, end of childhood: " + ", ".join(
        f"{m} {values(['replication 2'], c, 'standard', m, 'child_end').mean():.3f}" for m in ["wellbeing", "energy", ALL]))
print(f"  adult-world childhood, 100 twin sets, collapses in the standard world: home check after childhood "
      f"{values(POOL, 'adult', 'standard', ALL, 'frozen_home').mean():.3f}, end of adult life "
      f"{values(POOL, 'adult', 'standard', ALL, 'late').mean():.3f}")

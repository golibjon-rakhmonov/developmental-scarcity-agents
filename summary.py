"""Quick look at finished lives: one line per condition and stage of life.

    python summary.py            reads runs/*.csv
    python summary.py other_dir

Stages: end of childhood (last 10% of it), right after the move without
learning (frozen), the start of adult life (first 10%) and its end (last 10%).
Numbers are averaged over seeds; the spread between seeds is in brackets.
"""
import sys
from pathlib import Path

import pandas as pd

COLS = ["wellbeing", "collapses_per_1k", "carry_sated", "carry_sated_drought", "cold_day", "cold_night"]


def stage(df, phase, part):
    d = df[df.phase == phase]
    if d.empty:
        return None
    k = max(1, len(d) // 10)
    return {"all": d, "first": d.head(k), "last": d.tail(k)}[part][COLS].mean()


def main():
    folder = Path(sys.argv[1] if len(sys.argv) > 1 else "runs")
    rows = []
    for f in sorted(folder.glob("*.csv")):
        cond, seed = f.stem.rsplit("_s", 1)
        df = pd.read_csv(f)
        for label, phase, part in [("child end", "child", "last"), ("frozen", "frozen", "all"),
                                   ("adult start", "adult", "first"), ("adult end", "adult", "last")]:
            s = stage(df, phase, part)
            if s is not None:
                rows.append({"condition": cond, "seed": int(seed), "stage": label, **s})
    t = pd.DataFrame(rows)
    if t.empty:
        print("no runs found in", folder)
        return
    order = {"child end": 0, "frozen": 1, "adult start": 2, "adult end": 3}
    g = t.groupby(["stage", "condition"])[COLS]
    mean, std = g.mean(), g.std().fillna(0)
    out = pd.DataFrame({c: [f"{m:.3f} ({s:.3f})" for m, s in zip(mean[c], std[c])] for c in COLS}, index=mean.index)
    out["seeds"] = g.size()
    out = out.reset_index().sort_values(["stage", "condition"], key=lambda c: c.map(order) if c.name == "stage" else c)
    pd.set_option("display.width", 250)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()

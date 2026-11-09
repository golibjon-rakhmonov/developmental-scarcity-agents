# Pre-registration: robustness to the size of the network

Written after version 1, its two replications and experiment 2, and before any run with seeds 170-189.
The commit that adds this file fixes the code, the settings and the analysis. It is pushed to the
private GitHub repository before the first run.

    .venv/bin/python run_v1.py --parallel 17 --seeds $(seq 170 189) --childhoods scarce rich adult --hidden 64 --out runs_net64
    .venv/bin/python run_v1.py --parallel 17 --seeds $(seq 170 189) --childhoods scarce rich adult --hidden 256 --out runs_net256
    .venv/bin/python analyze_v1_rep.py --runs runs_net64 --seeds $(seq 170 189)
    .venv/bin/python analyze_v1_rep.py --runs runs_net256 --seeds $(seq 170 189)

## Why

Every registered run so far used one network: two hidden layers of 128 units. A result that holds only
for that size says little about learning agents in general. This check repeats version 1 with a smaller
and a larger network.

## What changes

Only the number of units in each hidden layer: 64 and 256 instead of 128. The sizes were fixed before
any calibration and are not tuned. Everything else is as in version 1: the scarce, rich and adult
childhoods, the three adult worlds, maps that change every 1,000 steps at clock 500, food that moves with
the map, the home check, the frozen checks, 3M childhood steps, 3M adult steps, PPO with the same
settings. The same seeds 170-189 are used for both sizes. With `--hidden 128` (the default) the code
gives exactly the lives of version 1.

## Predictions and tests

The 26 predictions of `PREREGISTRATION_V1_REPLICATION.md`, tested with `analyze_v1_rep.py` unchanged,
separately for each size: two-sided Wilcoxon signed-rank tests on paired differences, Holm within each
family, equivalence margins of ±0.15 (frozen check) and ±0.05 (end of adult life).

Known before this run: one of the 26, "scarce minus adult, collapses in the poorer world, no
difference", failed in both replications (scarce collapsed less, by 0.11 and 0.17 per 1,000 steps). It is
kept unchanged so that the analysis stays the same; a third failure in the same direction is read as
the same finding, not as a new one.

## How the results will be read

| Pattern | Conclusion |
|---|---|
| The collapse predictions of families B, D and E and the cold-ground predictions of families A and B hold at both sizes | The mismatch and the cold-ground habit do not depend on the size of the network. |
| They hold at one size and not at the other | The result depends on the size of the network; reported for each size. |
| They hold at neither size | The results of version 1 are specific to the 128-unit network. |

Food carried and the "no difference" predictions are reported as they come. Each prediction is reported
whether it holds or not.

## Calibration

Seed 900 only, childhood, home check and frozen checks, no adult life: scarce, rich and adult childhood at
64 and 256 units (`calibration/netsize/`). Its only purpose is to check that agents of both sizes learn;
nothing was tuned on it. Both sizes learn: wellbeing at the end of childhood
was 0.963, 0.974 and 0.945 (scarce, rich, adult) at 64 units and 0.975, 0.974 and 0.953 at 256, against
0.956-0.975 for the 128-unit network in `calibration/v1b`.

**Directions already seen** (one seed, recorded so they are not taken for blind predictions): collapses
per 1,000 steps in the richer world outside droughts were 3.36 (scarce) and 0.53 (rich) at 64 units, 2.66
and 0.49 at 256; in the poorer world 0.49 and 1.83 at 64 units, 0.61 and 1.82 at 256. The rich agents
spent the least daytime on cold ground at both sizes (0.012 and 0.007).

## Order of steps

1. This file and the code are committed in one commit.
2. The commit is pushed to the private GitHub repository.
3. Its hash is written into the report.
4. Only then `run_v1.py` is started.

## Exclusions

None. Every life is reported. A life that stops for a technical reason is rerun with the same seed and
settings.

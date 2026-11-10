# Pre-registration: experiment 3, the varied childhood

Written after version 1, its two replications, experiment 2 and the network-size check, and before any run
with seeds 190-219. The commit that adds this file fixes the code, the settings and the analysis. It is
pushed to the private GitHub repository before the first run.

    .venv/bin/python run_v1.py --parallel 17 --seeds $(seq 190 219) --childhoods scarce rich adult varied --out runs_exp3
    .venv/bin/python analyze_exp3.py
    .venv/bin/python analyze_v1_rep.py --runs runs_exp3 --seeds $(seq 190 219)

## Why

Version 1 and its replications found no hardening: a childhood with rare food made agents suited to rare
food, not tougher. Each childhood did worst in the adult world least like it. The open question is what,
if anything, does make an agent do well across worlds. One answer from the literature on people is that
it is not hardship as such but some hardship among good times (Seery et al., 2010). This experiment tests
that idea in the agents: a childhood in which plentiful and lean times take turns.

## What changes

One new childhood, `varied` (`VARIED` in `world.py`). Food grows at the rate of the rich childhood (0.02
per patch per step) in plentiful spells and at the rate of the scarce childhood (0.001) in lean spells.
Spells have random lengths (geometric, as in the unreliable childhood of experiment 2), 5,000 steps on
average for both kinds, so on average the agent spends half of its childhood in each. Each of the 32
worlds has its own spells, drawn from the world's own generator, so they do not depend on what the agent
does. The agent is not told which spell it is in.

Everything else is as in version 1: the scarce, rich and adult-world childhoods as twins of the varied
one, the three adult worlds, maps that change every 1,000 steps at clock 500, the home check, the frozen
checks, 3M childhood steps, 3M adult steps, the 128-unit network, PPO with the same settings. Without
`varied` the code gives exactly the lives of version 1.

**Fixed before calibration.** The two rates are those of the existing childhoods. The mean spell length,
5,000 steps for both kinds, was chosen before any life with this childhood was run, and for one reason:
when a plentiful spell ends, the patches are full (at most 8 pieces of food), and a lean spell should be
long enough that living off these leftovers is a small part of it. It was not tuned.

## Hypothesis and predictions

"Variety hardens": an agent that has lived through both plentiful and lean times does well in every adult
world, without the mismatch of the two one-sided childhoods. All measures are from the frozen check (right
after the move, no learning), in collapses per 1,000 agent-steps.

**Family J, variety removes the mismatch** (three directional tests, Holm over the three):

| Test | Adult world | Predicted |
|---|---|---|
| varied - scarce | richer, outside droughts | - |
| varied - rich | poorer | - |
| varied - rich | standard | - |

**Family K, no price for being a generalist** (three predictions of no difference, margin +-0.15 as in
the replications):

| Test | Adult world | Predicted |
|---|---|---|
| varied - rich | richer, outside droughts | no difference |
| varied - scarce | poorer | no difference |
| varied - adult | standard | no difference |

**Family L, better across worlds** (three directional tests, Holm over the three). The measure is the mean
of the collapses per 1,000 steps in the three adult worlds (whole frozen check of each):

| Test | Predicted |
|---|---|
| varied - scarce | - |
| varied - rich | - |
| varied - adult | - |

Tests: two-sided Wilcoxon signed-rank tests on paired differences, one pair per seed, with bootstrap 95%
confidence intervals (10,000 resamples) and the rank-biserial correlation. A directional prediction holds
if it is significant after Holm and the mean difference has the predicted sign. A prediction of no
difference holds only if the whole 95% interval lies inside +-0.15.

## How the results will be read

| Pattern | Conclusion |
|---|---|
| J holds, and L holds against all three childhoods | Variety hardens: a childhood with both plentiful and lean times gives an agent that survives better across worlds than any one-sided childhood and than a childhood in the adult world. |
| J holds, L holds against scarce and rich but not against adult | Variety removes the mismatch of its two halves, but is no better than growing up in the adult world (which has its own good and bad times). |
| J does not hold | Living through both kinds of times does not protect against mismatch. |
| K holds | The generalist pays no price: in each world it does as well as the childhood that suits that world. |
| K does not hold, varied worse | The generalist pays a price against the specialist. |

Each prediction is reported whether it holds or not. Time on cold ground, food carried, causes of
collapse, collapses at the end of adult life and the worst of the three worlds are reported as
descriptive.

## Third replication of version 1

The scarce, rich and adult-world childhoods on seeds 190-219 are also a third replication of the 26
predictions of `PREREGISTRATION_V1_REPLICATION.md`, tested with `analyze_v1_rep.py` unchanged. Known
before this run: "scarce minus adult, collapses in the poorer world, no difference" failed in both
replications in the same direction; a further failure in that direction is the same finding.

## Calibration

Seeds 900-901 only, childhood, home check and frozen checks, no adult life: the varied childhood, and the
rich and adult-world childhoods for comparison (`calibration/exp3/`; the scarce childhood with the same
seeds is in `calibration/exp2/scarce`). Its only purpose is to check that agents with the varied childhood
learn; nothing was tuned on it. They learn: wellbeing at the end of childhood was 0.963 and 0.965 (scarce
0.959 and 0.974, rich 0.976 and 0.975, adult 0.945 and 0.954), and they spent 49% and 46% of the end of
childhood in lean spells.

**Directions already seen** (two seeds, recorded so they are not taken for blind predictions). Collapses
per 1,000 steps in the frozen checks, seed 900 and seed 901:

| | varied | scarce | rich | adult |
|---|---|---|---|---|
| richer world, outside droughts | 0.43, 0.37 | 4.52, 2.08 | 0.48, 0.54 | 1.54, 1.47 |
| poorer world | 0.93, 0.74 | 0.63, 0.67 | 1.62, 1.66 | 0.99, 0.32 |
| standard world | 0.28, 0.19 | 1.43, 0.20 | 0.61, 0.61 | 0.43, 0.13 |
| mean of the three worlds | 0.57, 0.44 | 1.80, 0.83 | 0.95, 0.99 | 0.92, 0.52 |

So in both seeds the three differences of family J and the three of family L had the predicted sign. For
family K the differences were -0.05 and -0.17 (varied - rich, richer world), +0.29 and +0.07 (varied -
scarce, poorer world) and -0.15 and +0.06 (varied - adult, standard world).

## Order of steps

1. This file and the code are committed in one commit.
2. The commit is pushed to the private GitHub repository.
3. Its hash is written into the report.
4. Only then `run_v1.py` is started.

## Exclusions

None. Every life is reported. A life that stops for a technical reason is rerun with the same seed and
settings.

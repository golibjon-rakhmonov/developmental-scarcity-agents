# Pre-registration: pilot

Written before any run with seeds 0-9. The commit that adds this file fixes the code, the settings
and the analysis below. Anything decided later is reported as exploratory.

## Question

Does a scarce childhood, compared with a rich one, change how an agent stores food and how much
risk it takes once it lives in the same adult world?

## Design

- Conditions: scarce childhood, rich childhood, no childhood (baseline).
- Seeds 0-9, one life per seed and condition, 30 lives in total. Within a seed, the conditions share
  the starting network and the adult world.
- Settings: the defaults of `run_life.py` and `world.py` at this commit (3M childhood steps,
  100k frozen steps, 3M adult steps, 32 worlds, learning rate 3e-4, gamma 0.995).

## Primary measures

Measured in the frozen check, right after the move, with learning off:

1. **Storing**: mean number of food pieces carried while energy is above 0.8 (`carry_sated`).
2. **Risk**: share of daytime steps spent on cold ground (`cold_day`).

## Hypothesis and test

H1: the two childhoods differ on at least one primary measure. The direction is not fixed in
advance.

For each measure: scarce minus rich, paired by seed. Two-sided Wilcoxon signed-rank test, Holm
correction over the two measures, alpha 0.05. Reported with the mean paired difference, its
bootstrap 95% confidence interval (10,000 resamples of seeds) and the matched-pairs rank-biserial
correlation.

## Secondary, reported without a confirmatory claim

- The same measures during droughts and outside them.
- Collapses per 1,000 steps and their causes.
- How long the difference lasts: the paired difference in each 200k-step block of adult life.
- The no-childhood baseline next to both groups.

## What the calibration showed

Calibration used seeds 900 and above and is not part of the analysis. In one calibration seed, the
rich-childhood agent carried more food than the scarce one, the opposite of the naive expectation.
This is why the hypothesis is two-sided.

## Exclusions

None. Every life is reported. A life that crashes for a technical reason is rerun with the same seed.

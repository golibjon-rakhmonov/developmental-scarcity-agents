# Pre-registration: robustness check

Written after the pilot (seeds 0-9, see `PREREGISTRATION.md`) and before any run with seeds 10-29.
The commit that adds this file fixes the code, the settings and the analysis below.

## What the pilot found

Right after the move to the adult world, with learning off, agents from a scarce childhood
carried less food while full (-1.77 pieces), spent more of the day on cold ground (+5.8
percentage points) and collapsed less often (0.73 against 2.32 per 1,000 steps) than agents
from a rich childhood. All ten seeds went the same way on all three. The third measure was not
pre-registered in the pilot.

## Predictions

For scarce minus rich, paired by seed, in the frozen check:

1. **Storing** (`carry_sated`): negative.
2. **Risk** (`cold_day`): positive.
3. **Collapses** (`collapses_per_1k`): negative.

## Runs

All settings are the defaults of `run_life.py` and `world.py` at this commit unless listed.
`run_robust.py` runs everything.

| Variant | Childhoods | Seeds | Change |
|---|---|---|---|
| replication | scarce, rich, none | 10-29 | none |
| long_childhood | scarce, rich | 10-19 | 6M childhood steps instead of 3M |
| short_childhood | scarce, rich | 10-19 | 1M childhood steps |
| scarcer | scarce, rich | 10-19 | scarce regrowth 0.0007 instead of 0.001 |
| less_scarce | scarce, rich | 10-19 | scarce regrowth 0.002 |
| reset | scarce, rich | 10-19 | second hidden layer and both heads re-initialised at the move |

## Tests

`analyze_robust.py` does all of the following.

**Replication (confirmatory).** For each of the three measures: two-sided Wilcoxon signed-rank
test on the 20 paired differences, Holm correction over the three, alpha 0.05. Reported with the
mean difference, its bootstrap 95% confidence interval and the matched-pairs rank-biserial
correlation. A measure replicates if it is significant after correction and in the predicted
direction.

**Variations (confirmatory).** The same test for each measure in each of the four variations
(long and short childhood, scarcer and less scarce), ten pairs each, Holm correction over all
twelve tests. An effect generalises to a variation if it is significant after correction and in
the predicted direction.

**Secondary.** Whether the difference in collapses lasts: the same paired test on collapses in
the last 10% of adult life, replication seeds, no correction, no confirmatory claim.

**Exploratory: where the trace lives.** In the reset variant the frozen check measures a network
whose later layers were just re-initialised, so it is not tested. Instead the scarce minus rich
difference in each 200k-step block of the first million adult steps is shown next to the same
seeds without the reset. If the differences shrink after the reset, the trace was held mostly in
the later layers; if they stay, it was held in the first layer.

## Exclusions

None. Every life is reported. A life that crashes for a technical reason is rerun with the same
seed and settings.

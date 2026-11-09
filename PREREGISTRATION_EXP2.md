# Pre-registration: experiment 2, the hungry and the unreliable childhood

Written after version 1, its replication and the calibration below, and before any run with seeds
140-169. The commit that adds this file fixes the code, the settings and the analysis. It is pushed to
the private GitHub repository before the first run. Anything decided later is reported as exploratory.

    .venv/bin/python run_v1.py --parallel 17 --seeds $(seq 140 169) --childhoods scarce hungry unreliable rich adult --out runs_exp2
    .venv/bin/python analyze_exp2.py
    .venv/bin/python analyze_v1_rep.py --runs runs_exp2 --seeds $(seq 140 169)

## Why

Version 1 and its replication found that a childhood with rare food does not teach the agent to carry
food. Two ideas say what should teach it instead:

- **Hunger.** The everyday idea: whoever grew up hungry will hoard. The scarce childhood of version 1
  was not hungry (energy around 0.86), so it could not test this.
- **Unreliability.** The insurance hypothesis (Nettle, Andrews & Bateson, 2017): storing pays when
  access to food is uncertain, not when food is merely rare. In animals and people the store is body
  fat.

Experiment 2 tests both, and separates them.

## The two new childhoods

- **Unreliable.** As the scarce childhood on average (0.001 food per patch per step), but the food comes
  in waves of random length: harvests at the standard adult world's rate (0.004), on average 1,000 steps
  long, and empty spells with no regrowth, on average 3,000 steps long. The waves come from their own
  random generator, like the maps, so they do not depend on what the agent does. An empty spell counts as
  a drought in every measure.
- **Hungry.** Steady, like the scarce childhood, but with less food: 0.00045 per patch per step. The
  rate was chosen in calibration so that wellbeing at the end of childhood matches the unreliable
  childhood. So the hungry and the unreliable childhood are equally hard, and differ only in whether the
  food is steady.

Two rules of the world matter here and were noticed before any run:

- **Carrying costs.** Each piece carried costs energy every step; in 500 steps a piece costs as much as it
  gives. Carrying food pays only over short stretches. So storing is measured in two ways: food carried
  while full (as before) and energy in the adult world (storing in the body).
- **Harvests are partly lost.** A patch holds at most 1.0, so in a harvest food stops growing on full
  patches. With the same mean regrowth, the unreliable childhood is in fact harder than the scarce one.
  This is why it is compared with the hungry childhood, not with the scarce one.

## Design

- Childhoods: scarce, hungry, unreliable, rich, adult. Seeds 140-169: 30 per childhood, 150 processes,
  each with three adult lives (standard, richer and poorer adult world). Everything else as in version 1:
  maps change every 1,000 steps at clock 500, food moves with the map, the home check and the frozen
  checks, 3M childhood steps, 3M adult steps.
- 30 seeds instead of 20, because food carried is bimodal: most agents carry almost nothing, some carry a
  lot (5 of 20 scarce agents in the replication carried more than one piece).

## Tests

Two-sided Wilcoxon signed-rank tests on paired differences (one pair per seed), Holm within each family,
bootstrap 95% confidence intervals (10,000 resamples of seeds), matched-pairs rank-biserial
correlation. A prediction holds if it is significant after Holm and the mean difference has the
predicted sign. `analyze_exp2.py` does all of this.

| Family | Comparison | Measure | Where | Predicted |
|---|---|---|---|---|
| G, hunger | hungry - scarce | food carried while full | home check | + |
| G | hungry - scarce | food carried while full | frozen check, standard adult world | + |
| G | hungry - scarce | energy | frozen check, standard adult world | + |
| H, unreliability | unreliable - hungry | food carried while full | home check | + |
| H | unreliable - hungry | food carried while full | frozen check, standard adult world | + |
| H | unreliable - hungry | energy | frozen check, standard adult world | + |
| I | unreliable - hungry | collapses during droughts | frozen check, standard adult world | - |

Energy at home is not tested: the hungry and the unreliable childhood were matched on wellbeing, and the
hungry and scarce childhood differ in it by design.

**Second replication of version 1.** The scarce, rich and adult childhoods on seeds 140-169 are tested
with `analyze_v1_rep.py` exactly as in `PREREGISTRATION_V1_REPLICATION.md`: the same predictions, the
same equivalence margins.

Described without tests: the share of agents carrying more than half a piece while full at home, by
childhood; all measures by childhood at the end of childhood, in the home check, in the three frozen
checks and at the end of adult life.

## How the results will be read

| Pattern | Conclusion |
|---|---|
| G holds for food carried | Hunger teaches carrying food, and the habit stays after the move. |
| G holds for energy but not for food carried | Hunger teaches storing in the body, not in the hands. |
| H holds for food carried (at home, or at home and in the adult world) | Unreliability teaches carrying beyond what the same hardship does: the insurance hypothesis holds for a learning agent. If only at home, the habit does not carry over. |
| H does not hold for food carried, neither at home nor in the adult world | At equal hardship, unreliability does not teach carrying food in this world. |
| I holds | An unreliable childhood helps through the droughts of the standard adult world. |

Each prediction is reported whether it holds or not.

## Calibration

Seeds 900-901, childhood, home check and frozen checks only, no adult life. Everything is in
`calibration/exp2/`.

1. **Unreliable childhood, two wave lengths.** Harvests of 250 and empty spells of 750 steps on average,
   and 1,000 and 3,000. The rule written beforehand was: take the longest waves at which wellbeing at the
   end of childhood stays at or above 0.9. Neither did: 0.879 and 0.883 (short), 0.876 and 0.884 (long),
   with energy around 0.6 and frequent collapses from hunger (about 2 per 1,000 steps). The waves make the
   childhood hungry, for the reason given above. **The rule was dropped after this round**, and the design
   changed from "unreliable against scarce" to "unreliable against an equally hard steady childhood". The
   long waves were kept, as the original rule preferred the longest; the two lengths gave the same
   wellbeing.
2. **Steady childhood, rates 0.0004 and 0.0006.** Wellbeing at the end of childhood 0.836 and 0.827
   (0.0004), 0.958 and 0.960 (0.0006).
3. **Steady childhood, rates 0.00045, 0.0005 and 0.00055.** The rule, written before this round: the rate
   whose mean wellbeing at the end of childhood over seeds 900-901 is closest to that of the unreliable
   childhood with long waves (0.880). Wellbeing was 0.893 and 0.876 (0.00045, mean 0.885), 0.953 and
   0.956 (0.0005), 0.956 and 0.956 (0.00055). The rule picks **0.00045**.

Two things calibration showed about the match, recorded so the comparison is read with them in mind:

- **Matched on wellbeing, not on collapses.** At the end of childhood the unreliable agents collapsed
  about 2.4 times per 1,000 steps, the steady hungry agents about 0.16. Steady hunger keeps energy low
  (around 0.6) but rarely kills; an empty spell that starts at any moment does. This is part of what
  unreliability is, and it is not removed.
- **A knife edge.** Between 0.00045 and 0.0005 wellbeing jumps from 0.88 to 0.95: at about 0.0005 the food
  grown just covers what the body uses. So the hardship of the hungry childhood may vary more between
  seeds than that of the unreliable one. Wellbeing at the end of childhood is reported for every
  childhood.

The calibration code reproduces version 1 exactly: the scarce life with seed 901 of `calibration/v1b`
comes out identical, row for row, and the frozen checks of version 1 give the same numbers.

**Directions already seen in calibration** (two seeds, so not evidence, recorded so they are not
mistaken for predictions made blind):

- Steady hungry agents carried nothing at all, at every rate from 0.0004 to 0.0006: food carried while
  full 0.000 at home and in every adult world. They ate much more in the adult worlds (17-19 meals per 100 steps in the standard
  world, against 9-11 for the scarce agents) and kept their energy higher there (0.92-0.95 against
  0.85-0.88).
- Unreliable agents (long waves) carried 0.16 and 0.63 at home; scarce agents 1.33 and 0.01. At
  0.00045 the steady hungry agents carried 0.000 (both seeds).
- In the richer adult world, steady hungry agents collapsed about 4 times per 1,000 steps, more than any
  childhood in version 1.

## Order of steps

1. This file and the code are committed in one commit.
2. The commit is pushed to the private GitHub repository.
3. Its hash is written into the report.
4. Only then `run_v1.py` is started.

## Exclusions

None. Every life is reported. A life that stops for a technical reason is rerun with the same seed and
settings.

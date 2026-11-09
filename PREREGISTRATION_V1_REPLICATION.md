# Pre-registration: replication of version 1

Written after version 1 (seeds 100-119, `PREREGISTRATION_V1.md`, commit 1903050) and before any
run with seeds 120-139. The code and settings of the lives are exactly those of version 1; this
commit only adds this file and `analyze_v1_rep.py`. It is pushed to the private GitHub repository
before the first run.

    .venv/bin/python run_v1.py --parallel 17 --seeds $(seq 120 139) --out runs_v1_rep
    .venv/bin/python analyze_v1_rep.py

Seeds 120-139 have not been used before.

## What version 1 found

1. **Mismatch.** Right after the move, each childhood did worst where the adult world was least
   like it. In the richer world outside droughts agents from the rare-food childhood collapsed more
   often than those from the rich childhood (they ate more, drank less and collapsed from thirst);
   in the poorer world they collapsed less often (agents from the rich childhood starved).
2. **No hardening.** Agents from the rare-food childhood collapsed no less often than agents whose
   childhood was spent in the adult world itself.
3. **Habits that hold in every world.** Agents from the rich childhood carried more food and almost
   never went onto cold ground, in all three adult worlds.
4. **The difference in collapses fades.** At the end of adult life no difference was found.

Points 1 and 2 partly come from comparisons that version 1 only described (families D and E below
were not tested there). This replication tests all of them on new agents.

## Predictions

Directions are the signs found in version 1. "Scarce" is the rare-food childhood, "adult" the
childhood spent in the standard adult world. All comparisons are paired by seed.

| Family | Comparison | World and check | Prediction |
|---|---|---|---|
| A | scarce - rich, food carried while full | standard, frozen check | negative |
| A | scarce - rich, share of daytime on cold ground | standard, frozen check | positive |
| A | scarce - rich, collapses | standard, frozen check | negative |
| A | rich - adult, share of daytime on cold ground | standard, frozen check | negative |
| A | rich - adult, collapses | standard, frozen check | positive |
| A | scarce - adult, collapses | standard, frozen check | no difference |
| B | scarce - rich, food carried / cold ground / collapses | richer world outside droughts | negative / positive / **positive** |
| B | scarce - rich, food carried / cold ground / collapses | richer world during droughts | negative / positive / negative |
| B | scarce - rich, food carried / cold ground / collapses | poorer world | negative / positive / negative |
| C | scarce - rich, scarce - adult, rich - adult, collapses | standard, last 10% of adult life | no difference (all three) |
| D | scarce - rich, collapses from thirst | richer world, frozen check | positive |
| D | scarce - rich, meals per 100 steps | richer world, frozen check | positive |
| D | scarce - rich, drinks per 100 steps | richer world, frozen check | negative |
| D | scarce - rich, collapses from hunger | poorer world, frozen check | negative |
| E | scarce - adult, collapses | richer world outside droughts | positive |
| E | rich - adult, collapses | richer world outside droughts | negative |
| E | rich - adult, collapses | poorer world | positive |
| E | scarce - adult, collapses | poorer world | no difference |

## Tests

- **Directional predictions:** two-sided Wilcoxon signed-rank test on the 20 paired differences,
  Holm over the directional tests within each family. A prediction replicates if it is significant
  after Holm and the mean difference has the predicted sign.
- **"No difference" predictions:** an absence of significance is not evidence of no difference, so
  these are equivalence checks. The prediction holds if the bootstrap 95% confidence interval of the
  mean difference (10,000 resamples of seeds) lies entirely inside a margin: **±0.15 collapses per
  1,000 steps** in the frozen check (a third of the smallest collapse difference version 1 found
  there, rich - adult, 0.40) and **±0.05** at the end of adult life (half the level of late-life
  collapses, about 0.10). The margins were set from version 1 before any replication run.
- Every test is reported with the mean difference, its confidence interval, p, p after Holm and the
  matched-pairs rank-biserial correlation. `analyze_v1_rep.py` does all of this.

## How the results will be read

| Pattern | Conclusion |
|---|---|
| Collapses in B (richer outside droughts positive, poorer negative), D and the directional tests of E all replicate | Mismatch is confirmed: each childhood does worst where the adult world is least like it. |
| B in the poorer world replicates but B in the richer world outside droughts does not | Only "a rich childhood hurts in poor worlds" is confirmed; mismatch in both directions is not. |
| Scarce - adult holds as no difference in A and E, and C holds | No hardening is confirmed, with the margins above. |
| Scarce - adult shows fewer collapses for scarce, outside the margin, in A or E | Hardening appears after all; this contradicts version 1 and is reported as a conflict, without claiming either. |
| The predictions on food carried and cold ground replicate | The habits are confirmed as a lasting trace of childhood. |
| C does not hold | The difference in collapses does not fully fade within 3M adult steps. |

Each prediction is reported whether it replicates or not. No analysis is changed after the run.

## Everything else

As in `PREREGISTRATION_V1.md`: the same childhoods, adult worlds, maps, checks and measures, no
exclusions, a life that stops for a technical reason is rerun with the same seed and gives the same
file.

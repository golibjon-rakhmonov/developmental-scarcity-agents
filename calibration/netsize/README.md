# Calibration for the robustness check on network size

Seed 900 only. Childhood (3M steps), home check and the frozen checks in the three adult worlds, no adult
life. `run_cal.sh` reruns it; the lives were first run from a scratch copy of the code, which is why the
`out` paths in the json files differ. Nothing was tuned: the sizes, 64 and 256 units per hidden layer,
were fixed beforehand, and this only checks that agents of both sizes learn.

| Size | Childhood | Wellbeing, end of childhood | Collapses, end of childhood | Collapses, home check | Collapses, frozen check, standard world | Daytime on cold ground, standard world | Collapses, richer world outside droughts | Collapses, poorer world |
|---|---|---|---|---|---|---|---|---|
| 64 | scarce | 0.963 | 0.06 | 0.05 | 0.29 | 0.121 | 3.36 | 0.49 |
| 64 | rich | 0.974 | 0.27 | 0.18 | 0.67 | 0.012 | 0.53 | 1.83 |
| 64 | adult | 0.945 | 0.61 | 0.51 | 0.58 | 0.034 | 1.59 | 1.12 |
| 256 | scarce | 0.975 | 0.01 | 0.01 | 0.21 | 0.074 | 2.66 | 0.61 |
| 256 | rich | 0.974 | 0.27 | 0.31 | 0.68 | 0.007 | 0.49 | 1.82 |
| 256 | adult | 0.953 | 0.25 | 0.06 | 0.21 | 0.062 | 1.36 | 0.40 |

Collapses are per 1,000 steps; "end of childhood" is its last 10%.

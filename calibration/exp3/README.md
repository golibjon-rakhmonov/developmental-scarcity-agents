# Calibration for experiment 3

Seeds 900-901 only. Childhood (3M steps), home check and the frozen checks in the three adult worlds, no
adult life. `run_cal.sh` reruns it. The scarce lives with the same seeds and settings are in
`calibration/exp2/scarce`. Nothing was tuned: the varied childhood (plentiful spells at 0.02, lean spells
at 0.001, 5,000 steps each on average) was fixed beforehand, and this only checks that the agents learn
and live through both kinds of spells.

| Childhood | Seed | Wellbeing, end of childhood | Collapses, end of childhood | Collapses, home check | Collapses, standard world | Collapses, richer world outside droughts | Collapses, poorer world | Collapses, mean of the three worlds | Daytime on cold ground, standard world |
|---|---|---|---|---|---|---|---|---|---|
| varied | 900 | 0.963 | 0.13 | 0.06 | 0.28 | 0.43 | 0.93 | 0.57 | 0.027 |
| varied | 901 | 0.965 | 0.18 | 0.17 | 0.19 | 0.37 | 0.74 | 0.44 | 0.043 |
| scarce | 900 | 0.959 | 0.14 | 0.23 | 1.43 | 4.52 | 0.63 | 1.80 | 0.177 |
| scarce | 901 | 0.974 | 0.01 | 0.01 | 0.20 | 2.08 | 0.67 | 0.83 | 0.060 |
| rich | 900 | 0.976 | 0.25 | 0.31 | 0.61 | 0.48 | 1.62 | 0.95 | 0.021 |
| rich | 901 | 0.975 | 0.27 | 0.32 | 0.61 | 0.54 | 1.66 | 0.99 | 0.018 |
| adult | 900 | 0.945 | 0.58 | 0.30 | 0.43 | 1.54 | 0.99 | 0.92 | 0.039 |
| adult | 901 | 0.954 | 0.17 | 0.06 | 0.13 | 1.47 | 0.32 | 0.52 | 0.105 |

Collapses are per 1,000 steps; "end of childhood" is its last 10%; the adult-world columns are from the
frozen checks. At the end of childhood the varied agents spent 49% and 46% of their steps in lean spells.

Details and the directions already seen are in `PREREGISTRATION_EXP3.md`.

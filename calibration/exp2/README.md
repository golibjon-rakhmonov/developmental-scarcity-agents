# Calibration for experiment 2

Seeds 900-901 only. Childhood (3M steps), home check and the frozen checks in the three adult worlds,
no adult life. `run_cal.sh` reruns all three rounds; the lives were first run from a scratch copy of the
code, which is why the `out` paths in the json files differ. The scarce life with seed 901 is identical,
row for row, to `calibration/v1b/n1000/scarce_s901_standard.csv`.

| Childhood | Seed | Wellbeing, end of childhood | Energy, end of childhood | Collapses per 1,000 steps, end of childhood | Food carried while full, home check | Food carried while full, frozen check, standard world |
|---|---|---|---|---|---|---|
| unreliable, waves 250/750 | 900 | 0.879 | 0.606 | 2.08 | 1.112 | 1.586 |
| unreliable, waves 250/750 | 901 | 0.883 | 0.626 | 1.92 | 0.047 | 0.089 |
| unreliable, waves 1,000/3,000 | 900 | 0.876 | 0.567 | 2.53 | 0.155 | 0.179 |
| unreliable, waves 1,000/3,000 | 901 | 0.884 | 0.600 | 2.36 | 0.628 | 0.752 |
| scarce (0.001) | 900 | 0.959 | 0.838 | 0.14 | 1.333 | 3.037 |
| scarce (0.001) | 901 | 0.974 | 0.868 | 0.01 | 0.007 | 0.017 |
| steady 0.0004 | 900 | 0.836 | 0.494 | 0.68 | 0.000 | 0.000 |
| steady 0.0004 | 901 | 0.827 | 0.478 | 0.68 | 0.000 | 0.000 |
| steady 0.00045 | 900 | 0.893 | 0.624 | 0.17 | 0.000 | 0.000 |
| steady 0.00045 | 901 | 0.876 | 0.598 | 0.16 | 0.000 | 0.000 |
| steady 0.0005 | 900 | 0.953 | 0.801 | 0.03 | 0.000 | 0.000 |
| steady 0.0005 | 901 | 0.956 | 0.808 | 0.01 | 0.000 | 0.000 |
| steady 0.00055 | 900 | 0.956 | 0.835 | 0.03 | 0.000 | 0.000 |
| steady 0.00055 | 901 | 0.956 | 0.808 | 0.05 | 0.000 | 0.000 |
| steady 0.0006 | 900 | 0.958 | 0.836 | 0.02 | 0.000 | 0.000 |
| steady 0.0006 | 901 | 0.960 | 0.835 | 0.04 | 0.000 | 0.000 |

"End of childhood" is the last 10% of childhood. The steady childhoods are the scarce childhood with a
lower regrowth rate (`--scarce-regrow`); in experiment 2 the rate chosen, 0.00045, is the hungry
childhood.

**Round 1.** The rule written before it, take the longest waves at which wellbeing at the end of
childhood stays at or above 0.9, was met by neither wave length. The rule was dropped, and the design
changed to comparing the unreliable childhood with an equally hard steady one. The long waves were kept.

**Rounds 2 and 3.** The rule written before round 3: the steady rate whose mean wellbeing at the end of
childhood over the two seeds is closest to that of the unreliable childhood with long waves (0.880).
It picks 0.00045 (mean 0.885).

Details and the directions already seen are in `PREREGISTRATION_EXP2.md`.

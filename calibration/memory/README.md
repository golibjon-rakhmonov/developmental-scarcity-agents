# Calibration for the robustness check with memory

Seed 900 only. Childhood (3M steps), home check and the frozen checks in the three adult worlds, no adult
life, with `--memory`. `run_cal.sh` reruns it. Nothing was tuned: the network and the training were fixed
beforehand, and this only checks that the agent with memory learns.

| Childhood | Wellbeing, end of childhood | Collapses, end of childhood | Collapses, home check | Collapses, frozen check, standard world | Daytime on cold ground, standard world | Collapses, richer world outside droughts | Collapses, poorer world |
|---|---|---|---|---|---|---|---|
| scarce | 0.969 | 0.05 | 0.08 | 0.23 | 0.062 | 2.99 | 0.53 |
| rich | 0.981 | 0.15 | 0.16 | 0.42 | 0.005 | 0.14 | 1.55 |
| adult | 0.956 | 0.39 | 0.33 | 0.33 | 0.018 | 1.31 | 0.98 |

Collapses are per 1,000 steps; "end of childhood" is its last 10%. One childhood with memory took about
four times as long as one without (about 900 s against 200-240 s on the same machine).

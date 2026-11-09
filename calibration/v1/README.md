# Calibration of version 1

Seeds 900 and above, with the code of the commit that adds this folder. `run_cal.sh` reruns it
(`PYTHON=.venv/bin/python calibration/v1/run_cal.sh`). Maps change every 1,000 steps in every world.

| Run | What it checks | Result |
|---|---|---|
| scarce, seed 901 | the agent still learns with changing maps; the gap between the maps it lived on and new maps | wellbeing 0.972 at the end of childhood; collapses per 1,000 steps 0.03 at the end of childhood, 0.10 in the home check, 0.25 in the adult frozen check |
| rich, seed 901 | the same | wellbeing 0.975; collapses 0.29, 0.54 in the home check, 0.88 in the adult frozen check |
| adult, seed 902 | the new childhood in the adult world | wellbeing 0.952; collapses 0.18, 0.13 in the home check, 0.12 in the frozen check |
| none, seed 903, richer and poorer adult worlds, 1.5M steps | whether an agent can learn to live in the mirror worlds | collapses fall from about 3 to 0.5 per 1,000 steps in both; wellbeing 0.95 (richer) and 0.92 (poorer) in the last fifth |

For comparison, in version 0 (seeds 10-29) the same gap was 0.04 -> 0.55 (scarce) and 0.12 -> 1.61
(rich). With changing maps it is much smaller but not zero.

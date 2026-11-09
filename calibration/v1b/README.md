# Calibration of version 1, second round

The same four lives as in `../v1`, seeds 900 and above, with both fixes from the second review:
maps change when the clock modulo 1,000 equals 500, and food moves with the map within each class of
patches. `PYTHON=.venv/bin/python calibration/v1b/run_cal.sh` reruns it.

Collapses per 1,000 steps (in brackets: outside droughts / during droughts):

| Life | End of childhood | Home check (unseen maps) | Frozen check, standard adult world | Wellbeing, end of childhood |
|---|---|---|---|---|
| scarce, seed 901 | 0.01 | 0.01 | 0.20 (0.03 / 0.56) | 0.974 |
| rich, seed 901 | 0.27 | 0.32 | 0.61 (0.29 / 1.25) | 0.975 |
| adult, seed 902 | 0.22 | 0.30 | 0.34 (0.21 / 0.62) | 0.956 |

An agent with no childhood (seed 903), learning for 1.5M steps in a mirror world, gets from about 2.5-3
down to about 0.6 collapses per 1,000 steps in both (wellbeing in the last fifth 0.947 richer, 0.913
poorer).

One seed per life, and the adult childhood has a different seed from the other two, so none of this
says anything reliable about differences between childhoods.

#!/usr/bin/env bash
# Experiment 2 calibration, seeds 900-901: childhood, home check and the frozen checks in the three adult
# worlds, no adult life. Run from anywhere; results land next to this script. Two lives at a time.
#   round 1: the unreliable childhood with short waves (harvest 250, empty 750 steps on average) and long
#            waves (1,000 and 3,000), and the scarce childhood for comparison
#   round 2: a steady childhood with less food than the scarce one, rates 0.0004 and 0.0006
#   round 3: rates 0.00045, 0.0005 and 0.00055
cd "$(dirname "$0")/../.."
OUT=calibration/exp2
PY=${PYTHON:-python3}
C="--branches standard richer poorer --new-map-every 1000 --new-map-offset 500 --adult-steps 0"
{
for s in 900 901; do
  echo "--childhood unreliable --seed $s --wave-on 250 --wave-off 750 --out $OUT/short $C"
  echo "--childhood unreliable --seed $s --wave-on 1000 --wave-off 3000 --out $OUT/long $C"
  echo "--childhood scarce --seed $s --out $OUT/scarce $C"
done
for s in 900 901; do for r in 0.0004 0.0006; do
  echo "--childhood scarce --scarce-regrow $r --seed $s --out $OUT/hungry_$r $C"
done; done
for s in 900 901; do for r in 0.00045 0.0005 0.00055; do
  echo "--childhood scarce --scarce-regrow $r --seed $s --out $OUT/hungry_$r $C"
done; done
} | xargs -P 2 -L 1 "$PY" run_life.py
echo all done

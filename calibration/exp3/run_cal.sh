#!/usr/bin/env bash
# Experiment 3 calibration, seeds 900-901: childhood, home check and the frozen checks in the three adult
# worlds, no adult life. The varied childhood, and the rich and the adult-world childhood for comparison
# (the scarce childhood with the same seeds and settings is in calibration/exp2/scarce).
# Run from anywhere; results land next to this script. Two lives at a time.
cd "$(dirname "$0")/../.."
OUT=calibration/exp3
PY=${PYTHON:-python3}
C="--branches standard richer poorer --new-map-every 1000 --new-map-offset 500 --adult-steps 0"
{
for s in 900 901; do
  echo "--childhood varied --seed $s --out $OUT/varied $C"
done
for s in 900 901; do
  echo "--childhood rich --seed $s --out $OUT/rich $C"
  echo "--childhood adult --seed $s --out $OUT/adult $C"
done
} | xargs -P 2 -L 1 "$PY" run_life.py
echo all done

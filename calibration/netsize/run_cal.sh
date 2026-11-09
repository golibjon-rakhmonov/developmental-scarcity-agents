#!/usr/bin/env bash
# Robustness calibration: hidden layers of 64 and 256 units, scarce, rich and adult childhood, seed 900.
# Childhood, home check and frozen checks in the three adult worlds, no adult life. Two lives at a time.
cd "$(dirname "$0")/../.."
OUT=calibration/netsize
PY=${PYTHON:-python3}
C="--branches standard richer poorer --new-map-every 1000 --new-map-offset 500 --adult-steps 0"
{
for h in 64 256; do for c in scarce rich adult; do
  echo "--childhood $c --seed 900 --hidden $h --out $OUT/net$h $C"
done; done
} | xargs -P 2 -L 1 "$PY" run_life.py
echo all done

#!/usr/bin/env bash
# Calibration for the agent with memory: scarce, rich and adult childhood, seed 900. Childhood, home check
# and frozen checks in the three adult worlds, no adult life. Two lives at a time.
cd "$(dirname "$0")/../.."
OUT=calibration/memory
PY=${PYTHON:-python3}
C="--branches standard richer poorer --new-map-every 1000 --new-map-offset 500 --adult-steps 0 --memory"
{
for c in scarce rich adult; do
  echo "--childhood $c --seed 900 --out $OUT $C"
done
} | xargs -P 2 -L 1 "$PY" run_life.py
echo all done

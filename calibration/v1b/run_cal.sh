#!/usr/bin/env bash
# version-1 calibration, second round: maps change at clock 500 modulo 1,000 and food moves with
# the map within each class of patches. Seeds 900+ only.
cd "$(dirname "$0")/../.."
OUT=calibration/v1b
PY=${PYTHON:-python3}
M="--new-map-every 1000 --new-map-offset 500"
(
"$PY" run_life.py --childhood scarce --seed 901 $M --branches standard --adult-steps 0 --out $OUT/n1000 > $OUT/scarce_901.log 2>&1
"$PY" run_life.py --childhood none --seed 903 $M --branches poorer richer --adult-steps 1500000 --out $OUT/mirror > $OUT/none_903_mirror.log 2>&1
) &
(
"$PY" run_life.py --childhood rich --seed 901 $M --branches standard --adult-steps 0 --out $OUT/n1000 > $OUT/rich_901.log 2>&1
"$PY" run_life.py --childhood adult --seed 902 $M --branches standard --adult-steps 0 --out $OUT/n1000 > $OUT/adult_902.log 2>&1
) &
wait
echo all done

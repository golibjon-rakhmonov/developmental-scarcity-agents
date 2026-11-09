#!/usr/bin/env bash
# version-1 calibration, seeds 900+ only
cd "$(dirname "$0")/../.."
OUT=calibration/v1
PY=${PYTHON:-python3}
(
"$PY" run_life.py --childhood scarce --seed 901 --new-map-every 1000 --branches standard --adult-steps 0 --out $OUT/n1000 > $OUT/scarce_901_n1000.log 2>&1
"$PY" run_life.py --childhood none --seed 903 --new-map-every 1000 --branches poorer richer --adult-steps 1500000 --out $OUT/mirror > $OUT/none_903_mirror.log 2>&1
) &
(
"$PY" run_life.py --childhood rich --seed 901 --new-map-every 1000 --branches standard --adult-steps 0 --out $OUT/n1000 > $OUT/rich_901_n1000.log 2>&1
"$PY" run_life.py --childhood adult --seed 902 --new-map-every 1000 --branches standard --adult-steps 0 --out $OUT/n1000 > $OUT/adult_902_n1000.log 2>&1
) &
wait
echo all done

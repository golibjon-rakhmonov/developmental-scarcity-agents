#!/usr/bin/env bash
# The pilot: every seed lives three lives (scarce, rich, no childhood).
# Several lives run at once, one CPU core each.
#
#   ./run_pilot.sh                 seeds 0-9, 8 lives at a time
#   ./run_pilot.sh 100 102 6       seeds 100-102, 6 at a time
set -e
FIRST=${1:-0}
LAST=${2:-9}
PARALLEL=${3:-8}
PY=${PYTHON:-python3}
for s in $(seq "$FIRST" "$LAST"); do
  for c in scarce rich none; do
    [ -f "runs/${c}_s${s}_end.pt" ] && continue    # already finished
    echo "$c $s"
  done
done | xargs -P "$PARALLEL" -n 2 sh -c "$PY run_life.py --childhood \"\$0\" --seed \"\$1\""

#!/usr/bin/env bash
# Track every poke split_pokes.py found in one bench clip, in parallel.
#   scripts/track_pokes.sh <export> <name e.g. 230g> <mass kg> [extra track_video flags, e.g. --body-v 0.55]
set -uo pipefail
cd "$(dirname "$0")/.."
EXP=$1 NAME=$2 MASS=$3; shift 3
OUT=data/runs/real/bench; mkdir -p "$OUT" logs/bench
i=0
for r in $(jq -r '.pokes[] | "\(.[0]):\(.[1])"' "$EXP/pokes.json"); do
  i=$((i + 1)); p=$(printf "%s_p%02d" "$NAME" $i)
  .venv/bin/python scripts/track_video.py "$EXP" --label free --mass "$MASS" --frames ${r%:*} ${r#*:} "$@" \
    --out "$OUT/$p.npz" > "logs/bench/$p.log" 2>&1 &
done
wait
for f in logs/bench/${NAME}_p*.log; do echo "$(basename $f .log): $(grep -v Warn $f | grep -E 'wrote|Error|failed' | tail -1)"; done

#!/usr/bin/env bash
# Overnight 2026-10-07 (detached; survives the SSH session closing). Log: logs/overnight_1007.log
#  1. side-by-side demo videos (real | sim-IRL same camera | wide UR5e), one+ clip per straight-poke weight
#  2. wait for the straight estimator run to finish, then test it (and the old angled one) on the real clips
set -uo pipefail
cd "$(dirname "$0")/.."
export MUJOCO_GL=egl
PY=.venv/bin/python
RUN=gru-n5000-1007-1935
echo "$(date) start"

for c in straight_100g/b1798fd13e straight_160g/55b53993ad straight_230g/9fde79fbac straight_345g/7c7e3afa1d \
         straight_345g/307756464f straight_160g/18cb28b069; do
  name=$(echo "$c" | sed 's|straight_||; s|/|_|')
  $PY scripts/render_side_by_side.py "data/usable/$c/track.npz" --out "out/demo/$name.mp4" 2>&1 | grep -v Warn \
    || echo "RENDER FAILED: $c"
done
echo "$(date) demos done"

# wait for training (the python process) to exit; give up after 6 h
for i in $(seq 1 720); do
  pgrep -f "train_estimator.py --n 5000 --name $RUN" > /dev/null || break
  sleep 30
done
CKPT=checkpoints/estimator/$RUN/best.pt
if [ ! -f "$CKPT" ]; then
  echo "$(date) NO CHECKPOINT at $CKPT (training failed or still running): see logs/$RUN.log"
  exit 1
fi
echo "$(date) training finished; final metrics:"
grep -A12 "^FINAL" "logs/$RUN.log" | head -20
mkdir -p out/estimator_straight
TEST=$($PY -c "import json; print(' '.join('data/runs/real/'+c+'.npz' for c in json.load(open('data/replays/straight_split.json'))['test']))")
TRAIN=$($PY -c "import json; print(' '.join('data/runs/real/'+c+'.npz' for c in json.load(open('data/replays/straight_split.json'))['train']))")
echo "== new straight estimator, HELD-OUT test clips"
$PY scripts/estimate_real.py $TEST --ckpt "$CKPT" --json out/estimator_straight/test.json 2>&1 | grep -v Warn
echo "== new straight estimator, train-replay clips (seen as grip paths in training: not a clean test)"
$PY scripts/estimate_real.py $TRAIN --ckpt "$CKPT" --json out/estimator_straight/train_clips.json 2>&1 | grep -v Warn
echo "== old angled estimator (gru-n5000-1006-1818) on the same held-out straight clips, for comparison"
$PY scripts/estimate_real.py $TEST --json out/estimator_straight/test_old_angled_estimator.json 2>&1 | grep -v Warn
echo "$(date) ALL DONE"

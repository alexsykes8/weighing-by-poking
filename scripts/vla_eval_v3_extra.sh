#!/usr/bin/env bash
# Extra evals of the v3 policies (2026-10-13), after scripts/vla_straight_v4.sh frees the GPU:
#   1-2. sim-IRL: the measured needle, friction 0.55 and the real box set (100/160/230/345 g), full + camera-only
#   3.   oracle: v3 full in the training-sim with true masses in place of the estimator's (sensing ceiling)
# Same seeds as the training-sim evals (100000-100199). Log: logs/vla_eval_v3_extra.log
set -uo pipefail
cd "$(dirname "$0")/.."
while pgrep -f scripts/vla_straight_v4.sh > /dev/null; do sleep 60; done
echo "$(date) v4 finished; GPU free"
ev() {  # ev <variant> <out> [flags...]
  local V=$1 OUT=$2; shift 2
  local FLAG=""; [ "$V" = images_only ] && FLAG="--images-only"
  echo "$(date) eval $V -> $OUT ($*)"
  MUJOCO_GL=egl .venv-vla/bin/python scripts/eval_vla.py \
    --policy "outputs/vla/straight_v3_${V}/checkpoints/last/pretrained_model" --n 200 --videos 4 --style straight \
    --out "$OUT" $FLAG "$@" || echo "EVAL FAILED: $OUT"
}
ev full        out/vla_eval/straight_v3_full_simirl        --world sim-irl
ev images_only out/vla_eval/straight_v3_images_only_simirl --world sim-irl
ev full        out/vla_eval/straight_v3_full_oracle        --oracle
echo "$(date) ALL DONE"

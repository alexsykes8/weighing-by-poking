#!/usr/bin/env bash
# Eval-only test of the straight_v1 "no choice" episodes (2026-10-10): replan every 10 actions instead of 50 and allow
# 45 s instead of 30 (success_rate_30s scores the same run at the old timeout). Same seeds as the original evals.
# Log: logs/vla_eval_replan_straight_v1.log
set -uo pipefail
cd "$(dirname "$0")/.."
TAG=straight_v1
for V in full images_only; do
  FLAG=""; [ "$V" = images_only ] && FLAG="--images-only"
  echo "$(date) evaluating $V"
  MUJOCO_GL=egl .venv-vla/bin/python scripts/eval_vla.py --policy "outputs/vla/${TAG}_${V}/checkpoints/last/pretrained_model" \
    --n 200 --videos 4 --style straight --n-action-steps 10 --max-t 45 --out "out/vla_eval/${TAG}_${V}_replan10_t45" $FLAG \
    || echo "EVAL FAILED: $V"
done
echo "$(date) ALL DONE"

#!/usr/bin/env bash
# v3 policies on instructions never seen in training (2026-10-14): same seeds/scenes as the training-sim evals
# (100000-100199), only the sentence changes (eval_vla.py UNSEEN_PHRASINGS). Log: logs/vla_eval_v3_phrasings.log
set -uo pipefail
cd "$(dirname "$0")/.."
ev() {  # ev <variant> <out>
  local V=$1 OUT=$2; local FLAG=""; [ "$V" = images_only ] && FLAG="--images-only"
  echo "$(date) eval $V -> $OUT"
  MUJOCO_GL=egl .venv-vla/bin/python scripts/eval_vla.py \
    --policy "outputs/vla/straight_v3_${V}/checkpoints/last/pretrained_model" --n 200 --videos 4 --style straight \
    --unseen-phrasings --out "$OUT" $FLAG || echo "EVAL FAILED: $OUT"
}
ev full        out/vla_eval/straight_v3_full_phrasings
ev images_only out/vla_eval/straight_v3_images_only_phrasings
echo "$(date) ALL DONE"

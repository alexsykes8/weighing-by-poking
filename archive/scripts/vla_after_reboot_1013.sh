#!/usr/bin/env bash
# After the 2026-10-13 ~15:45 power-off: run the queued v3 extra evals first (sim-IRL full + camera-only, then the
# oracle), then finish v4 camera-only by resuming from its 005000 checkpoint (the run died at step 9574) and
# evaluate it as scripts/vla_straight_v4.sh would have. Log: logs/vla_after_reboot_1013.log
set -uo pipefail
cd "$(dirname "$0")/.."
scripts/vla_eval_v3_extra.sh
OUT=outputs/vla/straight_v4_images_only
echo "$(date) resuming v4 images_only from $OUT/checkpoints/last"
MUJOCO_GL=egl .venv-vla/bin/lerobot-train \
  --config_path="$OUT/checkpoints/last/pretrained_model/train_config.json" --resume=true \
  || { echo "TRAIN FAILED: v4 images_only (resume)"; exit 1; }
echo "$(date) evaluating v4 images_only"
MUJOCO_GL=egl .venv-vla/bin/python scripts/eval_vla.py --policy "$OUT/checkpoints/last/pretrained_model" \
  --n 200 --videos 4 --style straight --max-t 45 --out out/vla_eval/straight_v4_images_only --images-only \
  || echo "EVAL FAILED: v4 images_only"
echo "$(date) ALL DONE"

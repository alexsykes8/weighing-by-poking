#!/usr/bin/env bash
# Finish the straight_v1 VLA chain after the 2026-10-08 20:35 machine crash: resume the full-state SmolVLA run from
# its last checkpoint (step 10000 of 20000), train the images-only ablation, then evaluate both in sim-IRL.
# Same settings as scripts/vla_overnight.sh. Log: logs/vla_resume_straight_v1.log
set -uo pipefail
cd "$(dirname "$0")/.."
DS=data/vla/straight_v1; TAG=straight_v1; STEPS=20000; EVAL_N=200
echo "$(date) resuming full from $(readlink outputs/vla/${TAG}_full/checkpoints/last)"
MUJOCO_GL=egl .venv-vla/bin/lerobot-train \
  --config_path="outputs/vla/${TAG}_full/checkpoints/last/pretrained_model/train_config.json" --resume=true \
  || { echo "TRAIN FAILED: full (resume)"; exit 1; }
OUT="outputs/vla/${TAG}_images_only"
echo "$(date) training images_only -> $OUT"
MUJOCO_GL=egl .venv-vla/bin/lerobot-train --policy.path=lerobot/smolvla_base \
  --dataset.repo_id="local/needle_heaviest_images_only" --dataset.root="$DS/images_only" --dataset.video_backend=pyav \
  --policy.empty_cameras=2 --batch_size=16 --steps="$STEPS" --log_freq=200 --save_freq=5000 \
  --output_dir="$OUT" --job_name="smolvla_${TAG}_images_only" --policy.device=cuda --policy.push_to_hub=false \
  --wandb.enable=true --wandb.project=stick-probe --num_workers=8 || echo "TRAIN FAILED: images_only"
for V in full images_only; do
  FLAG=""; [ "$V" = images_only ] && FLAG="--images-only"
  echo "$(date) evaluating $V"
  MUJOCO_GL=egl .venv-vla/bin/python scripts/eval_vla.py --policy "outputs/vla/${TAG}_${V}/checkpoints/last/pretrained_model" \
    --n "$EVAL_N" --videos 4 --style straight --out "out/vla_eval/${TAG}_${V}" $FLAG || echo "EVAL FAILED: $V"
done
echo "$(date) ALL DONE"

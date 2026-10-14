#!/usr/bin/env bash
# straight_v2: regenerate demos (shuffled probe order, 30% recovery demos from hover poses, wrong expert demos
# dropped), fine-tune each straight_v1 model from its archived 20k checkpoint for 10k more steps, then evaluate
# both in sim-IRL exactly like straight_v1 (200 eps, seeds 100000+, trained chunk length, 30 s timeout).
# Log: logs/vla_straight_v2.log
set -uo pipefail
cd "$(dirname "$0")/.."
DS=data/vla/straight_v2; TAG=straight_v2; STEPS=10000; EVAL_N=200
BEST=outputs/vla/_best_straight_v1
echo "$(date) generating $DS"
MUJOCO_GL=egl .venv-vla/bin/python scripts/gen_vla_episodes.py --style straight --n 1000 --workers 6 \
  --shuffle-order --recovery-frac 0.3 --drop-wrong --out "$DS" || { echo "GENERATION FAILED"; exit 1; }
N_EPS="$(grep -c '"ok": true' "$DS/episodes.jsonl")"
echo "$(date) generation finished: $N_EPS episodes"
if [ "$N_EPS" -lt 700 ] || [ ! -f "$DS/full/meta/info.json" ]; then echo "$(date) ABORT: too few episodes"; exit 1; fi
for V in full images_only; do
  OUT="outputs/vla/${TAG}_${V}"
  rm -rf "$OUT"
  echo "$(date) fine-tuning $V from $BEST/$V/020000 -> $OUT"
  MUJOCO_GL=egl .venv-vla/bin/lerobot-train --policy.path="$BEST/$V/020000/pretrained_model" \
    --dataset.repo_id="local/needle_heaviest_${V}" --dataset.root="$DS/$V" --dataset.video_backend=pyav \
    --policy.empty_cameras=2 --batch_size=16 --steps="$STEPS" --log_freq=200 --save_freq=5000 \
    --output_dir="$OUT" --job_name="smolvla_${TAG}_${V}" --policy.device=cuda --policy.push_to_hub=false \
    --wandb.enable=true --wandb.project=stick-probe --num_workers=8 || echo "TRAIN FAILED: $V"
done
for V in full images_only; do
  FLAG=""; [ "$V" = images_only ] && FLAG="--images-only"
  echo "$(date) evaluating $V"
  MUJOCO_GL=egl .venv-vla/bin/python scripts/eval_vla.py --policy "outputs/vla/${TAG}_${V}/checkpoints/last/pretrained_model" \
    --n "$EVAL_N" --videos 4 --style straight --out "out/vla_eval/${TAG}_${V}" $FLAG || echo "EVAL FAILED: $V"
done
echo "$(date) ALL DONE"

#!/usr/bin/env bash
# straight_v4: v3's recipe (drop wrong expert demos, 10k fine-tune steps from the archived straight_v1 20k
# checkpoints) plus real-poke probes: each of the expert's three probes replays one of my recorded straight pokes
# (train-split clips) with probability 0.3, instead of the synthetic push. The only change from v3, so v3 is the
# control. Real pokes are longer, so the eval allows 45 s; summary.json also has success_rate_30s (v3's limit).
# Full model first, images_only after. Log: logs/vla_straight_v4.log
set -uo pipefail
cd "$(dirname "$0")/.."
DS=data/vla/straight_v4; TAG=straight_v4; STEPS=10000; EVAL_N=200
BEST=outputs/vla/_best_straight_v1

echo "$(date) generating $DS"
MUJOCO_GL=egl .venv-vla/bin/python scripts/gen_vla_episodes.py --style straight --n 1000 --workers 6 \
  --drop-wrong --replay-frac 0.3 --out "$DS" > logs/gen_straight_v4.log 2>&1
echo "$(date) generation rc $?"
N_EPS="$(grep -c '"ok": true' "$DS/episodes.jsonl" 2>/dev/null || echo 0)"
echo "$(date) $N_EPS episodes"
if [ "$N_EPS" -lt 700 ] || [ ! -f "$DS/full/meta/info.json" ]; then
  echo "$(date) ABORT v4: too few episodes"; exit 1
fi

for V in full images_only; do
  FLAG=""; [ "$V" = images_only ] && FLAG="--images-only"
  OUT="outputs/vla/${TAG}_${V}"
  rm -rf "$OUT"
  echo "$(date) fine-tuning $V from $BEST/$V/020000 -> $OUT"
  MUJOCO_GL=egl .venv-vla/bin/lerobot-train --policy.path="$BEST/$V/020000/pretrained_model" \
    --dataset.repo_id="local/needle_heaviest_${V}" --dataset.root="$DS/$V" --dataset.video_backend=pyav \
    --policy.empty_cameras=2 --batch_size=16 --steps="$STEPS" --log_freq=200 --save_freq=5000 \
    --output_dir="$OUT" --job_name="smolvla_${TAG}_${V}" --policy.device=cuda --policy.push_to_hub=false \
    --wandb.enable=true --wandb.project=stick-probe --num_workers=8 || echo "TRAIN FAILED: $V"
  echo "$(date) evaluating $V"
  MUJOCO_GL=egl .venv-vla/bin/python scripts/eval_vla.py --policy "$OUT/checkpoints/last/pretrained_model" \
    --n "$EVAL_N" --videos 4 --style straight --max-t 45 --out "out/vla_eval/${TAG}_${V}" $FLAG \
    || echo "EVAL FAILED: $V"
done
echo "$(date) ALL DONE"

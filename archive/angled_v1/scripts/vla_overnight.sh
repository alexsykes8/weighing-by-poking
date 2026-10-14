#!/usr/bin/env bash
# Chain: wait for episode generation -> fine-tune SmolVLA (full, images-only) -> closed-loop eval of both in sim-IRL.
#   scripts/vla_overnight.sh <dataset_dir> <gen_pid> [steps] [style] [eval_n] [min_episodes]
set -uo pipefail
cd "$(dirname "$0")/.."
DS="$1"; GEN_PID="$2"; STEPS="${3:-20000}"; STYLE="${4:-angled}"; EVAL_N="${5:-50}"; MIN_EPS="${6:-0}"
TAG="$(basename "$DS")"
while kill -0 "$GEN_PID" 2>/dev/null; do sleep 60; done
N_EPS="$(grep -c '"ok": true' "$DS/episodes.jsonl")"
echo "$(date) generation finished: $N_EPS episodes"
if [ "$N_EPS" -lt "$MIN_EPS" ] || [ ! -f "$DS/full/meta/info.json" ]; then
  echo "$(date) ABORT: generation died early ($N_EPS < $MIN_EPS episodes, or dataset not finalised)"; exit 1
fi
for V in full images_only; do
  OUT="outputs/vla/${TAG}_${V}"
  rm -rf "$OUT"
  echo "$(date) training $V -> $OUT"
  MUJOCO_GL=egl .venv-vla/bin/lerobot-train --policy.path=lerobot/smolvla_base \
    --dataset.repo_id="local/needle_heaviest_${V}" --dataset.root="$DS/$V" --dataset.video_backend=pyav \
    --policy.empty_cameras=2 --batch_size=16 --steps="$STEPS" --log_freq=200 --save_freq=5000 \
    --output_dir="$OUT" --job_name="smolvla_${TAG}_${V}" --policy.device=cuda --policy.push_to_hub=false \
    --wandb.enable=true --wandb.project=stick-probe --num_workers=8 || echo "TRAIN FAILED: $V"
done
for V in full images_only; do
  FLAG=""; [ "$V" = images_only ] && FLAG="--images-only"
  echo "$(date) evaluating $V"
  MUJOCO_GL=egl .venv-vla/bin/python scripts/eval_vla.py --policy "outputs/vla/${TAG}_${V}/checkpoints/last/pretrained_model" \
    --n "$EVAL_N" --videos 4 --style "$STYLE" --out "out/vla_eval/${TAG}_${V}" $FLAG || echo "EVAL FAILED: $V"
done
echo "$(date) ALL DONE"

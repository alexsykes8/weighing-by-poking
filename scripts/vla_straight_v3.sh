#!/usr/bin/env bash
# straight_v3: control for straight_v2's regression. Same fine-tune recipe as v2 (10k steps from the archived
# straight_v1 20k checkpoints) but the data only drops wrong expert demos: fixed L->R probe order, no recovery
# demos. Also evaluates v2's 5k checkpoints (does shorter fine-tuning hurt less?). Generation (CPU) starts at
# once; GPU work waits for scripts/vla_straight_v2.sh to finish. Full model first, images_only after.
# Log: logs/vla_straight_v3.log
set -uo pipefail
cd "$(dirname "$0")/.."
DS=data/vla/straight_v3; TAG=straight_v3; STEPS=10000; EVAL_N=200
BEST=outputs/vla/_best_straight_v1

echo "$(date) generating $DS (background)"
( MUJOCO_GL=egl .venv-vla/bin/python scripts/gen_vla_episodes.py --style straight --n 1000 --workers 6 \
    --drop-wrong --out "$DS" > logs/gen_straight_v3.log 2>&1; echo $? > "$DS.gen_rc" ) &
GEN_PID=$!

while pgrep -f scripts/vla_straight_v2.sh > /dev/null; do sleep 60; done
echo "$(date) straight_v2 finished; GPU free"

evaluate() {  # evaluate <policy dir> <out dir> <variant>
  local FLAG=""; [ "$3" = images_only ] && FLAG="--images-only"
  echo "$(date) evaluating $1 -> $2"
  MUJOCO_GL=egl .venv-vla/bin/python scripts/eval_vla.py --policy "$1" --n "$EVAL_N" --videos 4 --style straight \
    --out "$2" $FLAG || echo "EVAL FAILED: $1"
}

evaluate outputs/vla/straight_v2_full/checkpoints/005000/pretrained_model out/vla_eval/straight_v2_full_5k full

wait "$GEN_PID"
N_EPS="$(grep -c '"ok": true' "$DS/episodes.jsonl" 2>/dev/null || echo 0)"
echo "$(date) generation finished (rc $(cat "$DS.gen_rc")): $N_EPS episodes"
if [ "$N_EPS" -lt 700 ] || [ ! -f "$DS/full/meta/info.json" ]; then
  echo "$(date) ABORT v3: too few episodes"
else
  for V in full images_only; do
    OUT="outputs/vla/${TAG}_${V}"
    rm -rf "$OUT"
    echo "$(date) fine-tuning $V from $BEST/$V/020000 -> $OUT"
    MUJOCO_GL=egl .venv-vla/bin/lerobot-train --policy.path="$BEST/$V/020000/pretrained_model" \
      --dataset.repo_id="local/needle_heaviest_${V}" --dataset.root="$DS/$V" --dataset.video_backend=pyav \
      --policy.empty_cameras=2 --batch_size=16 --steps="$STEPS" --log_freq=200 --save_freq=5000 \
      --output_dir="$OUT" --job_name="smolvla_${TAG}_${V}" --policy.device=cuda --policy.push_to_hub=false \
      --wandb.enable=true --wandb.project=stick-probe --num_workers=8 || echo "TRAIN FAILED: $V"
    evaluate "$OUT/checkpoints/last/pretrained_model" "out/vla_eval/${TAG}_${V}" "$V"
  done
fi

evaluate outputs/vla/straight_v2_images_only/checkpoints/005000/pretrained_model \
  out/vla_eval/straight_v2_images_only_5k images_only
echo "$(date) ALL DONE"

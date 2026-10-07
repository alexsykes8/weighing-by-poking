#!/usr/bin/env bash
# Launch generation + training in the background so it survives the terminal closing.
#   scripts/launch_estimator.sh [N_ROLLOUTS] [extra train_estimator.py args...]
# Logs: logs/<name>.log ; attach with: tmux attach -t <name>  (or: tail -f logs/<name>.log)
set -euo pipefail
cd "$(dirname "$0")/.."
N="${1:-5000}"; shift || true
NAME="gru-n${N}-$(date +%m%d-%H%M)"
LOG="logs/${NAME}.log"
mkdir -p logs
# one BLAS thread per sim worker (30 parallel rollouts); torch sets its own thread count for training
CMD="OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -u scripts/train_estimator.py --n ${N} --name ${NAME} $* 2>&1 | tee ${LOG}"
if command -v tmux >/dev/null; then
  tmux new-session -d -s "${NAME}" "bash -lc '${CMD}'"
  echo "started tmux session ${NAME}; log ${LOG}"
else
  setsid nohup bash -c "${CMD}" >/dev/null 2>&1 < /dev/null &
  echo "started (nohup, pid $!) ${NAME}; log ${LOG}"
fi
echo "${NAME}" > logs/latest_estimator_run

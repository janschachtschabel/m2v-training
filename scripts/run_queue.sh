#!/usr/bin/env bash
# Runs several configs one after another (e.g. the teacher comparison), detached from the terminal:
#   nohup ./scripts/run_queue.sh configs/rtx5060ti-0*.json > runs/queue.log 2>&1 &
# Pin the queue to one GPU with CUDA_VISIBLE_DEVICES=0 (or 1). The queue ends early once runs/queue.stop exists
# (written by `scripts/server.sh stop`).
set -uo pipefail
cd "$(dirname "$0")/.."
for config in "$@"; do
  if [ -e runs/queue.stop ]; then echo "$(date -Is) stopped before $config"; exit 0; fi
  echo "$(date -Is) start $config (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-all})"
  .venv/bin/python -m m2vtrain run "$config" || echo "$(date -Is) FAILED $config"
done
echo "$(date -Is) queue done"

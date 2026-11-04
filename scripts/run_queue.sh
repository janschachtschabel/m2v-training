#!/usr/bin/env bash
# Runs several configs one after another (e.g. the teacher comparison), detached from the terminal:
#   nohup ./scripts/run_queue.sh configs/rtx5060ti-0*.json > runs/queue.log 2>&1 &
set -uo pipefail
for config in "$@"; do
  echo "$(date -Is) start $config"
  .venv/bin/python -m m2vtrain run "$config" || echo "$(date -Is) FAILED $config"
done
echo "$(date -Is) queue done"

#!/usr/bin/env bash
# Control script for the GPU PC (Linux), meant to be called over SSH from another computer:
#   ssh user@gpu-pc '~/m2v-training/scripts/server.sh status'
# or through scripts/remote.sh / scripts/remote.ps1 on the other computer.
#
# Actions:
#   update                 git pull (fast-forward only) and reinstall the package
#   check                  GPUs, CUDA build, keyword list
#   start-compare          teacher comparison, one queue per GPU (GPU 0: gte, bge-m3, arctic; GPU 1: Qwen3-4B, Qwen3-0.6B)
#   start-final [config]   final 5-fold run, teacher embeddings on both GPUs (default: configs/dual-gpu/final-5fold-qwen3-4b.json)
#   start <config> [gpu]   any config, optionally pinned to GPU 0 or 1
#   status                 runs, GPU load, last queue lines
#   log [lines]            end of the newest run log
#   results [run ...]      comparison table of finished runs (Markdown)
#   stop                   stop all queues and running runs
#   ui                     start the Streamlit UI on localhost:8501 (reach it through an SSH tunnel)
#
# Started jobs are detached (setsid + nohup), so they keep running after the SSH session ends.
# Training data (data/keywords.txt, exports) is never part of the repository: copy it to data/ yourself.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
mkdir -p runs

COMPARE_GPU0=(configs/rtx5060ti-01-gte-reference.json configs/rtx5060ti-02-bge-m3.json configs/rtx5060ti-03-arctic-embed-l-v2.json)
COMPARE_GPU1=(configs/rtx5060ti-05-qwen3-embedding-4b.json configs/rtx5060ti-04-qwen3-embedding-0.6b.json)
FINAL_CONFIG=configs/dual-gpu/final-5fold-qwen3-4b.json

die() { echo "error: $*" >&2; exit 1; }

queue_running() {
  local pidfile
  for pidfile in runs/queue-*.pid; do
    [ -e "$pidfile" ] || continue
    if kill -0 "$(cat "$pidfile")" 2>/dev/null; then echo "$pidfile"; return 0; fi
  done
  return 1
}

preflight() {
  [ -x "$PY" ] || die "no .venv here; run ./scripts/setup_linux.sh first"
  if running=$(queue_running); then die "a queue is still running ($running); use 'stop' first or wait"; fi
  local config
  for config in "$@"; do
    [ -f "$config" ] || die "config not found: $config"
    if grep -q '"data/keywords.txt"' "$config" && [ ! -f data/keywords.txt ]; then
      die "data/keywords.txt is missing; copy the keyword list to $(pwd)/data/ (it is not in the repository)"
    fi
  done
  rm -f runs/queue.stop
}

# start_queue <name> <gpus or ""> <config...>
start_queue() {
  local name=$1 gpus=$2
  shift 2
  if [ -n "$gpus" ]; then
    CUDA_VISIBLE_DEVICES=$gpus setsid nohup ./scripts/run_queue.sh "$@" > "runs/queue-$name.log" 2>&1 < /dev/null &
  else
    setsid nohup ./scripts/run_queue.sh "$@" > "runs/queue-$name.log" 2>&1 < /dev/null &
  fi
  echo $! > "runs/queue-$name.pid"
  echo "started queue $name (PID $!, GPU ${gpus:-all}): $*"
  echo "  log: runs/queue-$name.log"
}

action=${1:-status}
shift || true
case "$action" in
  update)
    git pull --ff-only
    "$PY" -m pip install -q -e ".[dev]"
    git log -1 --oneline
    ;;
  check)
    "$PY" scripts/check_gpu.py
    if [ -f data/keywords.txt ]; then echo "data/keywords.txt: $(wc -l < data/keywords.txt) lines"; else echo "data/keywords.txt: MISSING"; fi
    ;;
  start-compare)
    preflight "${COMPARE_GPU0[@]}" "${COMPARE_GPU1[@]}"
    start_queue gpu0 0 "${COMPARE_GPU0[@]}"
    start_queue gpu1 1 "${COMPARE_GPU1[@]}"
    ;;
  start-final)
    config=${1:-$FINAL_CONFIG}
    preflight "$config"
    start_queue final "" "$config"
    ;;
  start)
    [ $# -ge 1 ] || die "usage: start <config> [gpu]"
    preflight "$1"
    start_queue "$(basename "$1" .json)" "${2:-}" "$1"
    ;;
  status)
    "$PY" -m m2vtrain status
    echo
    nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null \
      || echo "nvidia-smi not available"
    for log in runs/queue-*.log; do
      [ -e "$log" ] || continue
      echo; echo "== $log"; tail -n 3 "$log"
    done
    ;;
  log)
    newest=$(ls -t runs/*/run.log 2>/dev/null | head -n 1 || true)
    [ -n "$newest" ] || die "no run log yet"
    echo "== $newest"; tail -n "${1:-40}" "$newest"
    ;;
  results)
    "$PY" -m m2vtrain compare "$@"
    ;;
  stop)
    touch runs/queue.stop   # queues end before their next config; removed by the next start
    "$PY" -m m2vtrain stop || true
    ;;
  ui)
    setsid nohup "$PY" -m streamlit run ui/app.py --server.address localhost --server.port 8501 --server.headless true \
      > runs/ui.log 2>&1 < /dev/null &
    echo "UI started (PID $!) on localhost:8501; from the other computer: ssh -N -L 8501:localhost:8501 <user>@<gpu-pc>"
    ;;
  *)
    sed -n '2,20p' "$0"; exit 2
    ;;
esac

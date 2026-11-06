#!/usr/bin/env bash
# Controls training on the GPU PC from another computer over SSH (Linux, macOS, WSL or Git Bash).
#
#   export M2V_HOST=jan@gpu-pc            # SSH target (key login recommended)
#   export M2V_SERVER_OS=linux            # or windows
#   export M2V_REPO=~/m2v-training        # repository folder on the GPU PC; on Windows e.g. C:/Users/jan/m2v-training
#
#   ./scripts/remote.sh status
#   ./scripts/remote.sh push-data data/keywords.txt   # copies a data file to <repo>/data/ on the GPU PC (never via git)
#   ./scripts/remote.sh update                       # git pull + reinstall on the GPU PC
#   ./scripts/remote.sh start-compare                # teacher comparison, one queue per GPU
#   ./scripts/remote.sh start-final                  # final 5-fold run on both GPUs
#   ./scripts/remote.sh log 60 | results | stop | check | ui
#   ./scripts/remote.sh tunnel                       # then open http://localhost:8501 (after 'ui')
#   ./scripts/remote.sh fetch models/<run>           # copies a folder from the GPU PC into the current folder
#
# All other actions are passed to scripts/server.sh or scripts/server.ps1 on the GPU PC; see there.
set -euo pipefail
HOST=${M2V_HOST:?set M2V_HOST, e.g. export M2V_HOST=jan@gpu-pc}
OS=${M2V_SERVER_OS:-linux}
if [ "$OS" = windows ]; then
  REPO=${M2V_REPO:?set M2V_REPO to the repository folder on the GPU PC, e.g. C:/Users/jan/m2v-training}
  REPO=${REPO//\\//}
else
  REPO=${M2V_REPO:-'~/m2v-training'}
fi
action=${1:-status}
shift || true

case "$action" in
  tunnel)
    echo "UI: http://localhost:8501 (Ctrl+C ends the tunnel, not the UI)"
    exec ssh -N -L 8501:localhost:8501 "$HOST"
    ;;
  push-data)
    [ $# -ge 1 ] || { echo "usage: push-data <file> [...]" >&2; exit 2; }
    exec scp "$@" "$HOST:$REPO/data/"
    ;;
  fetch)
    [ $# -ge 1 ] || { echo "usage: fetch <path in repository, e.g. models/m2v-qwen3-4b-256-edu>" >&2; exit 2; }
    exec scp -r "$HOST:$REPO/$1" .
    ;;
esac

if [ "$OS" = windows ]; then
  exec ssh "$HOST" "powershell -NoProfile -ExecutionPolicy Bypass -File \"$REPO/scripts/server.ps1\" $action $*"
else
  exec ssh "$HOST" "$REPO/scripts/server.sh $action $(printf '%q ' "$@")"
fi

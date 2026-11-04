#!/usr/bin/env bash
# Sets up m2v-training on Linux with an NVIDIA GPU (e.g. RTX 5060 Ti 16 GB).
# Usage: ./scripts/setup_linux.sh [python] [torch-index-url]
# RTX 50xx (Blackwell, sm_120) needs a PyTorch build for CUDA 12.8 or newer.
set -euo pipefail
PYTHON="${1:-python3.12}"
TORCH_INDEX="${2:-https://download.pytorch.org/whl/cu128}"

[ -d .venv ] || "$PYTHON" -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install torch --index-url "$TORCH_INDEX"
.venv/bin/python -m pip install -e ".[dev]"
.venv/bin/python scripts/check_gpu.py

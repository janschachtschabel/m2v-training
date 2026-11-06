# Sets up m2v-training on Windows with an NVIDIA GPU (e.g. RTX 5060 Ti 16 GB).
# Run from the repository folder in PowerShell:  .\scripts\setup_windows.ps1
# RTX 50xx (Blackwell, sm_120) needs a PyTorch build for CUDA 12.8 or newer.
param(
    [string]$Python = "py -3.12",
    [string]$TorchIndex = "https://download.pytorch.org/whl/cu128"
)
$ErrorActionPreference = "Stop"

if (-not (Test-Path .venv)) {
    Invoke-Expression "$Python -m venv .venv"
}
& .venv\Scripts\python.exe -m pip install --upgrade pip
& .venv\Scripts\python.exe -m pip install torch --index-url $TorchIndex
& .venv\Scripts\python.exe -m pip install -e ".[dev]"
& .venv\Scripts\python.exe scripts\check_gpu.py

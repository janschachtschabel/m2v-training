# Control script for the GPU PC (Windows), meant to be called over SSH from another computer:
#   ssh jan@gpu-pc "powershell -NoProfile -ExecutionPolicy Bypass -File C:\Users\jan\m2v-training\scripts\server.ps1 status"
# or through scripts\remote.ps1 / scripts/remote.sh on the other computer.
#
# Actions:
#   update                 git pull (fast-forward only) and reinstall the package
#   check                  GPUs, CUDA build, keyword list
#   start-compare          teacher comparison, one queue per GPU (GPU 0: gte, bge-m3, arctic; GPU 1: Qwen3-4B, Qwen3-0.6B)
#   start-final [config]   final 5-fold run, teacher embeddings on both GPUs (default: configs\dual-gpu\final-5fold-qwen3-4b.json)
#   start <config> [gpu]   any config, optionally pinned to GPU 0 or 1
#   status                 runs, GPU load, last queue lines
#   log [lines]            end of the newest run log
#   results [run ...]      comparison table of finished runs (Markdown)
#   stop                   stop all queues and running runs
#   ui                     start the Streamlit UI on localhost:8501 (reach it through an SSH tunnel)
#
# Jobs are started through WMI, so they keep running after the SSH session ends (processes started directly in
# an SSH session are killed with it). Training data (data\keywords.txt, exports) is never part of the
# repository: copy it to data\ yourself.
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$python = Join-Path $root ".venv\Scripts\python.exe"
New-Item -ItemType Directory -Force -Path (Join-Path $root "runs") | Out-Null

$compareGpu0 = @("configs\rtx5060ti-01-gte-reference.json", "configs\rtx5060ti-02-bge-m3.json", "configs\rtx5060ti-03-arctic-embed-l-v2.json")
$compareGpu1 = @("configs\rtx5060ti-05-qwen3-embedding-4b.json", "configs\rtx5060ti-04-qwen3-embedding-0.6b.json")
$finalConfig = "configs\dual-gpu\final-5fold-qwen3-4b.json"

function Fail([string]$message) { Write-Host "error: $message"; exit 1 }

function Get-RunningQueue {
    foreach ($file in Get-ChildItem -Path "runs" -Filter "queue-*.pid" -ErrorAction SilentlyContinue) {
        $queuePid = [int](Get-Content $file.FullName -Raw).Trim()
        if (Get-Process -Id $queuePid -ErrorAction SilentlyContinue) { return $file.Name }
    }
    return $null
}

function Test-Preflight([string[]]$configs) {
    if (-not (Test-Path $python)) { Fail "no .venv here; run .\scripts\setup_windows.ps1 first" }
    $running = Get-RunningQueue
    if ($running) { Fail "a queue is still running ($running); use 'stop' first or wait" }
    foreach ($config in $configs) {
        if (-not (Test-Path $config)) { Fail "config not found: $config" }
        if ((Get-Content $config -Raw).Contains('"data/keywords.txt"') -and -not (Test-Path "data\keywords.txt")) {
            Fail "data\keywords.txt is missing; copy the keyword list to $root\data\ (it is not in the repository)"
        }
    }
    Remove-Item "runs\queue.stop" -ErrorAction SilentlyContinue
}

# Starts `$inner` through WMI; output goes to runs\<stem>.log, the process id to runs\<stem>.pid.
function Start-Detached([string]$stem, [string]$gpus, [string]$inner) {
    $log = Join-Path $root "runs\$stem.log"
    $set = if ($gpus) { "set CUDA_VISIBLE_DEVICES=$gpus&& " } else { "" }
    $command = "cmd.exe /c `"$set$inner > `"$log`" 2>&1`""
    $result = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{ CommandLine = $command; CurrentDirectory = $root }
    if ($result.ReturnValue -ne 0) { Fail "start failed (code $($result.ReturnValue))" }
    Set-Content -Path (Join-Path $root "runs\$stem.pid") -Value $result.ProcessId
    $shown = if ($gpus) { $gpus } else { "all" }
    Write-Host "started $stem (PID $($result.ProcessId), GPU $shown)"
    Write-Host "  log: runs\$stem.log"
}

function Start-Queue([string]$name, [string]$gpus, [string[]]$configs) {
    $queue = Join-Path $root "scripts\run_queue.ps1"
    Start-Detached "queue-$name" $gpus "powershell -NoProfile -ExecutionPolicy Bypass -File `"$queue`" $($configs -join ' ')"
}

$action = if ($args.Count -ge 1) { $args[0] } else { "status" }
$rest = @($args | Select-Object -Skip 1)

switch ($action) {
    "update" {
        git pull --ff-only
        if ($LASTEXITCODE -ne 0) { Fail "git pull failed" }
        & $python -m pip install -q -e ".[dev]"
        git log -1 --oneline
    }
    "check" {
        & $python scripts\check_gpu.py
        if (Test-Path "data\keywords.txt") { Write-Host "data\keywords.txt: $((Get-Content data\keywords.txt).Count) lines" }
        else { Write-Host "data\keywords.txt: MISSING" }
    }
    "start-compare" {
        Test-Preflight ($compareGpu0 + $compareGpu1)
        Start-Queue "gpu0" "0" $compareGpu0
        Start-Queue "gpu1" "1" $compareGpu1
    }
    "start-final" {
        $config = if ($rest.Count -ge 1) { $rest[0] } else { $finalConfig }
        Test-Preflight @($config)
        Start-Queue "final" "" @($config)
    }
    "start" {
        if ($rest.Count -lt 1) { Fail "usage: start <config> [gpu]" }
        $gpu = if ($rest.Count -ge 2) { [string]$rest[1] } else { "" }
        Test-Preflight @($rest[0])
        Start-Queue ([IO.Path]::GetFileNameWithoutExtension($rest[0])) $gpu @($rest[0])
    }
    "status" {
        & $python -m m2vtrain status
        Write-Host ""
        try { nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader }
        catch { Write-Host "nvidia-smi not available" }
        foreach ($log in Get-ChildItem -Path "runs" -Filter "queue-*.log" -ErrorAction SilentlyContinue) {
            Write-Host ""; Write-Host "== runs\$($log.Name)"
            Get-Content $log.FullName -Tail 3
        }
    }
    "log" {
        $lines = if ($rest.Count -ge 1) { [int]$rest[0] } else { 40 }
        $newest = Get-ChildItem -Path "runs" -Filter "run.log" -Recurse -Depth 1 -ErrorAction SilentlyContinue |
            Sort-Object LastWriteTime -Descending | Select-Object -First 1
        if (-not $newest) { Fail "no run log yet" }
        Write-Host "== $($newest.FullName)"
        Get-Content $newest.FullName -Tail $lines
    }
    "results" {
        & $python -m m2vtrain compare @rest
    }
    "stop" {
        New-Item -ItemType File -Force -Path "runs\queue.stop" | Out-Null  # queues end before their next config
        & $python -m m2vtrain stop
    }
    "ui" {
        Start-Detached "ui" "" "`"$python`" -m streamlit run ui\app.py --server.address localhost --server.port 8501 --server.headless true"
        Write-Host "UI on localhost:8501; from the other computer: ssh -N -L 8501:localhost:8501 <user>@<gpu-pc>"
    }
    default {
        Get-Content $PSCommandPath -TotalCount 19 | Select-Object -Skip 1
        exit 2
    }
}

# Runs several configs one after another, e.g. the teacher comparison:
#   .\scripts\run_queue.ps1 configs\rtx5060ti-01-gte-reference.json configs\rtx5060ti-02-bge-m3.json
# To keep it running after closing the terminal, start it detached with scripts\server.ps1 (or through WMI:
#   Invoke-CimMethod Win32_Process Create @{ CommandLine = "powershell -File scripts\run_queue.ps1 configs\a.json"; CurrentDirectory = (Get-Location).Path }).
# Pin the queue to one GPU by setting $env:CUDA_VISIBLE_DEVICES = "0" (or "1") first. The queue ends early once
# runs\queue.stop exists (written by scripts\server.ps1 stop).
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$python = Join-Path $root ".venv\Scripts\python.exe"
$gpus = if ($env:CUDA_VISIBLE_DEVICES) { $env:CUDA_VISIBLE_DEVICES } else { "all" }
foreach ($config in $args) {
    if (Test-Path "runs\queue.stop") { Write-Host "$(Get-Date -Format s) stopped before $config"; exit 0 }
    Write-Host "$(Get-Date -Format s) start $config (CUDA_VISIBLE_DEVICES=$gpus)"
    & $python -m m2vtrain run $config
    if ($LASTEXITCODE -ne 0) { Write-Host "$(Get-Date -Format s) FAILED $config" }
}
Write-Host "$(Get-Date -Format s) queue done"

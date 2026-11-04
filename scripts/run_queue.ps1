# Runs several configs one after another, e.g. the teacher comparison:
#   .\scripts\run_queue.ps1 configs\rtx5060ti-01-gte-reference.json configs\rtx5060ti-02-bge-m3.json
# To keep it running after closing the terminal, start this script itself detached, e.g.
#   Invoke-CimMethod Win32_Process Create @{ CommandLine = "powershell -File scripts\run_queue.ps1 configs\a.json configs\b.json"; CurrentDirectory = (Get-Location).Path }
$python = Join-Path (Get-Location).Path ".venv\Scripts\python.exe"
foreach ($config in $args) {
    Write-Host "$(Get-Date -Format s) start $config"
    & $python -m m2vtrain run $config
    if ($LASTEXITCODE -ne 0) { Write-Host "$(Get-Date -Format s) FAILED $config" }
}
Write-Host "$(Get-Date -Format s) queue done"

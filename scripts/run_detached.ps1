# Starts a training run as a process that is independent of this terminal, IDE or remote session.
# Runs started as child processes of an IDE terminal or remote session die when that parent is closed or
# restarted; a process created through WMI keeps running. Output goes to runs\<name>-console.log.
#
# Usage (from the repository folder):  .\scripts\run_detached.ps1 configs\rtx5060ti-02-bge-m3.json
param([Parameter(Mandatory = $true)][string]$Config)
$ErrorActionPreference = "Stop"

$root = (Get-Location).Path
$python = Join-Path $root ".venv\Scripts\python.exe"
$name = [IO.Path]::GetFileNameWithoutExtension($Config)
New-Item -ItemType Directory -Force -Path (Join-Path $root "runs") | Out-Null
$log = Join-Path $root "runs\$name-console.log"
$command = "cmd.exe /c `"`"$python`" -m m2vtrain run `"$Config`" > `"$log`" 2>&1`""
$result = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{ CommandLine = $command; CurrentDirectory = $root }
if ($result.ReturnValue -ne 0) { throw "Start failed (code $($result.ReturnValue))" }
Write-Host "Started (PID $($result.ProcessId)). Log: $log. Status: runs\<timestamp>-<name>\status.json or the UI."

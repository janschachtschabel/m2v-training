# Controls training on the GPU PC from another Windows computer over SSH (OpenSSH client, built into Windows).
#
#   $env:M2V_HOST = "jan@gpu-pc"                  # SSH target (key login recommended)
#   $env:M2V_SERVER_OS = "windows"                # or "linux"
#   $env:M2V_REPO = "C:/Users/jan/m2v-training"   # repository folder on the GPU PC; Linux e.g. ~/m2v-training
#
#   .\scripts\remote.ps1 status
#   .\scripts\remote.ps1 push-data data\keywords.txt   # copies a data file to <repo>/data/ on the GPU PC (never via git)
#   .\scripts\remote.ps1 update                       # git pull + reinstall on the GPU PC
#   .\scripts\remote.ps1 start-compare                # teacher comparison, one queue per GPU
#   .\scripts\remote.ps1 start-final                  # final 5-fold run on both GPUs
#   .\scripts\remote.ps1 log 60 | results | stop | check | ui
#   .\scripts\remote.ps1 tunnel                       # then open http://localhost:8501 (after 'ui')
#   .\scripts\remote.ps1 fetch models/<run>           # copies a folder from the GPU PC into the current folder
#
# All other actions are passed to scripts/server.ps1 or scripts/server.sh on the GPU PC; see there.
$ErrorActionPreference = "Stop"
if (-not $env:M2V_HOST) { throw 'Set $env:M2V_HOST first, e.g. $env:M2V_HOST = "jan@gpu-pc"' }
$sshHost = $env:M2V_HOST
$os = if ($env:M2V_SERVER_OS) { $env:M2V_SERVER_OS } else { "windows" }
if ($env:M2V_REPO) { $repo = $env:M2V_REPO.Replace("\", "/") }
elseif ($os -eq "linux") { $repo = "~/m2v-training" }
else { throw 'Set $env:M2V_REPO to the repository folder on the GPU PC, e.g. "C:/Users/jan/m2v-training"' }

$action = if ($args.Count -ge 1) { [string]$args[0] } else { "status" }
$rest = @($args | Select-Object -Skip 1 | ForEach-Object { [string]$_ })

switch ($action) {
    "tunnel" {
        Write-Host "UI: http://localhost:8501 (Ctrl+C ends the tunnel, not the UI)"
        ssh -N -L 8501:localhost:8501 $sshHost
    }
    "push-data" {
        if ($rest.Count -lt 1) { throw "usage: push-data <file> [...]" }
        scp @rest "${sshHost}:$repo/data/"
    }
    "fetch" {
        if ($rest.Count -lt 1) { throw "usage: fetch <path in repository, e.g. models/m2v-qwen3-4b-256-edu>" }
        scp -r "${sshHost}:$repo/$($rest[0])" .
    }
    default {
        $arguments = ($rest | ForEach-Object { $_.Replace("\", "/") }) -join " "
        if ($os -eq "linux") {
            ssh $sshHost "$repo/scripts/server.sh $action $arguments"
        } else {
            ssh $sshHost "powershell -NoProfile -ExecutionPolicy Bypass -File $repo/scripts/server.ps1 $action $arguments"
        }
    }
}
exit $LASTEXITCODE

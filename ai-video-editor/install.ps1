# Windows 10/11 installer (PowerShell).  Usage:  powershell -ExecutionPolicy Bypass -File install.ps1 [-WithModels]
param([switch]$WithModels)
$ErrorActionPreference = "Stop"
if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
  Write-Host "ffmpeg not found. Installing with winget..."
  winget install --id Gyan.FFmpeg -e --accept-source-agreements --accept-package-agreements
  Write-Host "Close and reopen PowerShell so PATH updates, then run this script again."; exit 0
}
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
python -m pip install -r "$here\requirements.txt"
python "$here\scripts\env_check.py" --strict | Out-Null
if ($LASTEXITCODE -eq 0) { Write-Host "Core install OK." }
if ($WithModels) {
  python "$here\scripts\setup_voice.py"
  python "$here\scripts\setup_models.py" --matting --whisper tiny
}
python "$here\scripts\env_check.py"

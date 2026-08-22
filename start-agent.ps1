# DaNa Agent Startup Script (PowerShell)

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
if (Test-Path (Join-Path $ScriptDir "dana-agent")) {
    Set-Location (Join-Path $ScriptDir "dana-agent")
} elseif (Test-Path ".\dana-agent") {
    Set-Location ".\dana-agent"
}

if (Test-Path ".venv\Scripts\activate.ps1") {
    & ".venv\Scripts\activate.ps1"
}

uv run python -m dana_agent --bypass-wake --control

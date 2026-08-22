# DaNa Agent Startup Script (PowerShell)

if (Test-Path ".venv\Scripts\activate.ps1") {
    & ".venv\Scripts\activate.ps1"
}

uv run python -m dana_agent --bypass-wake --control

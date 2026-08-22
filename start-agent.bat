@echo off
rem DaNa Agent Startup Script (Batch / Command Prompt)

if exist "%~dp0dana-agent" (
    cd /d "%~dp0dana-agent"
) else if exist ".\dana-agent" (
    cd .\dana-agent
)

if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
)

uv run python -m dana_agent --bypass-wake --control

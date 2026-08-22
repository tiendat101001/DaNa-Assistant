@echo off
rem DaNa Agent Startup Script (Batch / Command Prompt)

if exist "%~dp0.venv\Scripts\activate.bat" (
    call "%~dp0.venv\Scripts\activate.bat"
) else if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
)

uv run python -m dana_agent --bypass-wake --control

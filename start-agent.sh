#!/usr/bin/env bash
# DaNa Agent Startup Script (Bash / Git Bash / WSL)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -d "$SCRIPT_DIR/dana-agent" ]; then
    cd "$SCRIPT_DIR/dana-agent" || exit 1
elif [ -d "./dana-agent" ]; then
    cd ./dana-agent || exit 1
fi

if [ -f ".venv/Scripts/activate" ]; then
    source .venv/Scripts/activate
elif [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

uv run python -m dana_agent --bypass-wake --control

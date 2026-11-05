# Run scripts/serve_sglang.sh inside WSL (SGLang has no native Windows build).
# Usage: .\scripts\serve_sglang.ps1 [-Model Qwen/Qwen3-1.7B] [-Port 30000] [-Parser qwen]
# WSL2 forwards localhost, so DaNa on Windows reaches http://127.0.0.1:<Port>/v1.
param(
    [string]$Model = "Qwen/Qwen2.5-1.5B-Instruct",
    [int]$Port = 30000,
    [string]$Parser = "qwen25"
)
$script = (Resolve-Path (Join-Path $PSScriptRoot "serve_sglang.sh")).Path
$wslPath = (wsl wslpath -a ($script -replace '\\', '/')).Trim()
wsl env "MODEL=$Model" "PORT=$Port" "PARSER=$Parser" bash $wslPath

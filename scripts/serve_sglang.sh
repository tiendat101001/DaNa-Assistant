#!/usr/bin/env bash
# Serve a lightweight chat model with SGLang (OpenAI-compatible API) for DaNa.
#
#   MODEL=Qwen/Qwen3-1.7B PARSER=qwen ./scripts/serve_sglang.sh
#
# Matching DaNa config (env or dana.toml):
#   DANA_CONVERSATION__LLM__PROVIDER=sglang
#   DANA_CONVERSATION__LLM__MODEL=${MODEL}
#   DANA_CONVERSATION__LLM__BASE_URL=http://127.0.0.1:${PORT}/v1   # default for sglang
#   DANA_CONVERSATION__LLM__MAX_TOKENS=160
#
# Tool-call parsers: qwen25 (Qwen2.5), qwen (Qwen3), llama3 (Llama 3.x), mistral.
#
# Low-latency notes:
#   * Keep the model resident and warm (DaNa warms it at startup).
#   * Radix prefix caching is on by default: keep the system prompt stable so
#     it is cached between turns.
#   * Small models + --mem-fraction-static 0.6 leaves headroom for STT/TTS on
#     the same GPU. Raise to 0.8 on a dedicated GPU.
#   * For single-user voice, batch size 1 is the target; do NOT raise
#     --max-running-requests. Add `--enable-torch-compile` (slower start,
#     faster decode) or `--cuda-graph-max-bs 1` once things work.
#   * CPU-only: SGLang has a CPU backend for Intel Xeon with AMX
#     (bf16/int8, `--device cpu`, needs the sglang CPU build). On consumer CPUs
#     without AMX expect seconds of latency: prefer a cloud API or Ollama.
#   * HF gated models (Llama) need HF_TOKEN exported first.
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
PORT="${PORT:-30000}"
PARSER="${PARSER:-qwen25}"
HOST="${HOST:-127.0.0.1}" # keep loopback: the server has no auth

exec python -m sglang.launch_server \
  --model-path "${MODEL}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --tool-call-parser "${PARSER}" \
  --mem-fraction-static 0.6 \
  "$@"

"""Brain LLM provider names and aliases — dependency-free so config validation
never has to import LangChain. ``build_chat_model`` (model.py) consumes these.
"""

from __future__ import annotations

# Alias -> init_chat_model provider. OpenAI-compatible endpoints ride the OpenAI
# client with a base_url (DeepSeek, Grok, Ollama, vLLM, LM Studio).
ALIASES: dict[str, str] = {
    "gemini": "google_genai",
    "openai_compatible": "openai",
    "grok": "openai",
    "deepseek": "openai",
    "ollama": "openai",
    "lmstudio": "openai",
    "sglang": "openai",
    "vllm": "openai",
    # Together AI — hosted open-weight models behind an OpenAI-compatible API.
    "together": "openai",
}

# Self-hosted OpenAI-compatible servers: default endpoint when ``base_url`` is
# unset, and no real API key needed.
DEFAULT_BASE_URLS: dict[str, str] = {
    "sglang": "http://127.0.0.1:30000/v1",
    "vllm": "http://127.0.0.1:8000/v1",
    "ollama": "http://127.0.0.1:11434/v1",
    "lmstudio": "http://127.0.0.1:1234/v1",
}
SELF_HOSTED_PROVIDERS: frozenset[str] = frozenset(DEFAULT_BASE_URLS)

# Hosted OpenAI-compatible clouds: default endpoint only. These are NOT
# self-hosted (data leaves the machine), so privacy.local_only still rejects them.
CLOUD_BASE_URLS: dict[str, str] = {
    "together": "https://api.together.xyz/v1",
}

# Providers the brain can build (plus the offline "mock").
BRAIN_PROVIDERS: frozenset[str] = frozenset(
    {"mock", "openai", "anthropic", "google_genai", *ALIASES}
)

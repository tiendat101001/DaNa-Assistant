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
}

# Providers the brain can build (plus the offline "mock").
BRAIN_PROVIDERS: frozenset[str] = frozenset(
    {"mock", "openai", "anthropic", "google_genai", *ALIASES}
)

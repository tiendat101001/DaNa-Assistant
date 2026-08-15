---
name: provider-plugin-architecture
description: How to design a clean, swappable multi-provider (plugin) architecture in Python so any backend — LLM, STT, TTS, wake word, embeddings, storage, actions — can be selected purely from config without touching call sites. Use this skill whenever the user wants "multiple providers to choose from", "swappable backends", pluggable adapters, an abstraction layer over vendors (OpenAI vs Gemini vs DeepSeek vs self-hosted, AssemblyAI vs Whisper, ElevenLabs vs Piper, etc.), or asks for abstract base classes / interfaces / a factory or registry so components can be interchanged. Consult it before writing any provider interface so all capabilities are abstracted the same consistent way and selection stays config-driven.
---

# Provider / Plugin Architecture

The goal: adding, removing, or switching a backend is a **config change**, never an edit to business logic. Every capability (LLM, STT, TTS, wake word, actions, storage) follows the *same* pattern so the codebase is predictable.

Common failure modes this skill prevents: each capability abstracted differently; provider names hard-coded in call sites; `if provider == "openai": ... elif ...` ladders; config scattered as loose kwargs; no way to validate that a chosen combination actually works.

## The four pieces

Every swappable capability has exactly these four:

1. **An interface (ABC)** — the narrow contract call sites depend on.
2. **Concrete providers** — one class per backend, implementing the ABC.
3. **A registry + factory** — maps a config key to a provider class and constructs it.
4. **Typed config** — one schema per capability, validated, with secrets handled.

Keep these consistent across *all* capabilities. If wake-word providers and LLM providers are structured differently, the abstraction has already failed.

## 1. Interface (ABC)

Design the interface around what call sites need, not around any one vendor's SDK. Keep it minimal — a fat interface forces every provider to implement things some backends can't do.

```python
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

class LLMProvider(ABC):
    """Contract every LLM backend must satisfy."""

    @abstractmethod
    async def stream(self, messages: list[Message], tools: list[Tool] | None = None
                     ) -> AsyncIterator[Chunk]:
        """Yield response chunks as they arrive. Streaming is the default —
        voice and UX depend on it; a non-streaming provider wraps its result
        in a single-chunk iterator."""

    @property
    @abstractmethod
    def capabilities(self) -> "Capabilities":
        """Declare what this provider supports so combinations can be validated."""
```

Guidance:
- **Prefer async and streaming** as the default shape for LLM/STT/TTS — real-time systems need it, and a synchronous backend is trivially wrapped, whereas retrofitting streaming later is painful.
- Use plain typed data classes (or Pydantic models) for inputs/outputs — `Message`, `Chunk`, `Transcript` — so providers are interchangeable at the boundary. Do not leak a vendor SDK's types through the interface.
- One ABC per capability: `LLMProvider`, `STTProvider`, `TTSProvider`, `WakeWordProvider`, `S2SProvider`, `ActionProvider`. Same shape, same conventions.

## 2. Capability flags — declare, then validate

Providers differ in what they can do. Make differences **data**, not special-case code, so the app can validate a chosen stack at startup and fail fast with a clear message instead of crashing mid-conversation.

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class Capabilities:
    streaming: bool
    tool_calling: bool = False
    self_hosted: bool = False
    languages: frozenset[str] = frozenset({"en"})
    max_context: int | None = None
```

At startup, check the assembled configuration against declared capabilities (e.g. "the chosen brain needs tool calling but provider X reports `tool_calling=False`") and refuse to start with an actionable error.

## 3. Registry + factory (no if/elif ladders)

A registry maps a string key from config to a class; a factory builds the instance from config. Providers self-register via a decorator so adding one never touches the factory.

```python
_LLM_REGISTRY: dict[str, type[LLMProvider]] = {}

def register_llm(name: str):
    def wrap(cls: type[LLMProvider]) -> type[LLMProvider]:
        _LLM_REGISTRY[name] = cls
        return cls
    return wrap

def build_llm(cfg: "LLMConfig") -> LLMProvider:
    try:
        cls = _LLM_REGISTRY[cfg.provider]
    except KeyError:
        known = ", ".join(sorted(_LLM_REGISTRY))
        raise ConfigError(f"unknown llm provider {cfg.provider!r}; known: {known}")
    return cls(cfg)

@register_llm("openai")
class OpenAILLM(LLMProvider): ...

@register_llm("gemini")
class GeminiLLM(LLMProvider): ...
```

- Providers self-register at import time — the factory has zero knowledge of specific vendors.
- For optional/heavy deps, import the SDK **inside** the provider module (or lazily) so a missing package only breaks the one provider, not the whole app. Consider Python entry points if providers should be distributed as separate packages.
- "OpenAI-compatible / self-hosted" is usually the *same* provider class with a configurable `base_url` — not a new class. Support a base URL override on the OpenAI provider and you cover vLLM, Ollama, LM Studio, DeepSeek, and most self-hosted endpoints at once.

## 4. Typed, layered config (single source of truth)

Use `pydantic-settings`. One schema for the whole app, with a sub-model per capability. This schema is also the contract any management UI reads and writes.

```python
from pydantic import BaseModel, SecretStr
from pydantic_settings import BaseSettings

class LLMConfig(BaseModel):
    provider: str = "openai"
    model: str = "gpt-4o"
    base_url: str | None = None          # set this for self-hosted / compatible
    api_key: SecretStr | None = None
    temperature: float = 0.7

class AppConfig(BaseSettings):
    llm: LLMConfig = LLMConfig()
    stt: "STTConfig" = ...
    tts: "TTSConfig" = ...
    class Config:
        env_nested_delimiter = "__"      # LLM__PROVIDER=gemini
```

Rules:
- **Precedence: environment > config file > defaults.** State it and implement it once.
- **Secrets** as `SecretStr`, sourced from env / a secret store, never written to logs or serialized configs. Redact on display.
- Config is **validated on load**; a bad provider name or missing key fails at startup, not at first request.
- Support **hot-reload** where feasible (rebuild affected providers on change) so the UI can apply changes without a restart — but only for safe fields.

## Cross-cutting behavior belongs in the base, not each provider

Retries, timeouts, rate-limit backoff, and metrics should not be re-implemented per vendor. Put them in the ABC (template methods) or a decorator/wrapper that every provider passes through, so behavior is uniform and providers stay thin.

```python
class LLMProvider(ABC):
    async def complete(self, messages, tools=None):
        return await self._with_retry(lambda: self._raw(messages, tools))
    # providers implement _raw; retry/timeout/metrics live here once
```

## Testing the abstraction

- Write **one shared test suite against the ABC** and run every provider through it (parametrize over the registry) so all providers are verified to honor the contract.
- Ship a **mock/fake provider** per capability (deterministic, no network) as the default in tests and local dev — this is what makes the rest of the system testable offline.
- Test the factory's error paths: unknown provider name, missing key, incompatible capability combination.

## Review checklist

- [ ] One ABC per capability, all following the same conventions.
- [ ] No `if provider == ...` ladders anywhere; selection goes through the registry.
- [ ] Providers self-register; the factory names no specific vendor.
- [ ] Capabilities are declared and validated at startup with clear errors.
- [ ] Config is one typed schema; precedence env > file > defaults; secrets are `SecretStr`.
- [ ] Self-hosted/compatible endpoints handled via `base_url`, not a fork of the class.
- [ ] Retry/timeout/metrics live once in the base, not per provider.
- [ ] A mock provider exists for every capability and is the test default.

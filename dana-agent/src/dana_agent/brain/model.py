"""Build the brain's chat model.

``init_chat_model`` is itself the provider-swap mechanism the brief wants for the
brain: one call selects Gemini / OpenAI / Anthropic / any OpenAI-compatible or
self-hosted endpoint from config, with ``base_url`` covering DeepSeek, Grok,
Ollama, vLLM, and LM Studio at once. So ``LLMConfig`` (provider, model, api_key,
base_url, temperature) maps straight onto it — no bespoke per-vendor classes.

For offline dev/tests (``provider = "mock"``) we return an echo ``MockChatModel``
so the whole stack assembles and runs with no key and no network.
"""

from __future__ import annotations

import time
from collections.abc import Iterator, Sequence
from typing import Any, cast

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable

from dana_agent.brain.providers import (
    ALIASES,
    BRAIN_PROVIDERS,
    CLOUD_BASE_URLS,
    DEFAULT_BASE_URLS,
    SELF_HOSTED_PROVIDERS,
)
from dana_agent.config import LLMConfig
from dana_agent.errors import ConfigError
from dana_agent.observability import get_logger

log = get_logger(__name__)

__all__ = ["BRAIN_PROVIDERS", "MockChatModel", "build_chat_model", "warmup_chat_model"]


class MockChatModel(BaseChatModel):  # type: ignore[misc]  # BaseChatModel is Any under strict
    """Deterministic offline chat model: echoes the last human message as
    streamed word chunks. Never calls tools. Used when ``provider = "mock"``."""

    @property
    def _llm_type(self) -> str:
        return "mock"

    def _reply(self, messages: list[BaseMessage]) -> str:
        last = ""
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                last = _to_text(msg.content)
                break
        return f"mock reply to: {last}".strip()

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        message = AIMessage(content=self._reply(messages))
        return ChatResult(generations=[ChatGeneration(message=message)])

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        for word in self._reply(messages).split():
            yield ChatGenerationChunk(message=AIMessageChunk(content=word + " "))

    def bind_tools(
        self,
        tools: Sequence[Any],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        # The mock never calls tools; binding is a no-op so the graph builds.
        return cast("Runnable[LanguageModelInput, AIMessage]", self)


def _to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            p if isinstance(p, str) else p.get("text", "")
            for p in content
            if isinstance(p, str) or isinstance(p, dict)
        ]
        return "".join(parts)
    return str(content)


def build_chat_model(cfg: LLMConfig) -> BaseChatModel:
    """Construct the brain's chat model from config (unbound; the brain binds
    its gated tools)."""
    if cfg.provider not in BRAIN_PROVIDERS:
        known = ", ".join(sorted(BRAIN_PROVIDERS))
        raise ConfigError(f"unknown brain llm provider {cfg.provider!r}; known: {known}")
    if cfg.provider == "mock":
        return MockChatModel()
    return _build_real(cfg)


def _build_real(cfg: LLMConfig) -> BaseChatModel:
    try:
        from langchain.chat_models import init_chat_model
    except ImportError as exc:  # pragma: no cover - only without the brain extra
        raise ConfigError(
            "the brain requires the 'brain' extra: install dana-agent[brain] "
            "(and the matching provider extra, e.g. brain-openai)"
        ) from exc

    provider = ALIASES.get(cfg.provider, cfg.provider)
    self_hosted = cfg.provider in SELF_HOSTED_PROVIDERS
    params: dict[str, Any] = {
        "model": cfg.model,
        "model_provider": provider,
        "temperature": cfg.temperature,
    }
    if cfg.api_key is not None:
        params["api_key"] = cfg.api_key.get_secret_value()
    elif self_hosted:
        params["api_key"] = "EMPTY"  # the OpenAI client insists on a non-empty key
    base_url = (
        cfg.base_url or DEFAULT_BASE_URLS.get(cfg.provider) or CLOUD_BASE_URLS.get(cfg.provider)
    )
    if base_url is not None:
        params["base_url"] = base_url
    if cfg.max_tokens is not None:
        params["max_tokens"] = cfg.max_tokens
    if cfg.timeout_s is not None:
        params["timeout"] = cfg.timeout_s
    if provider == "google_genai" and cfg.thinking_budget is not None:
        params["thinking_budget"] = cfg.thinking_budget
    if provider == "openai":
        params["stream_usage"] = True  # token usage in streaming responses
        if cfg.reasoning_effort is not None:
            params["reasoning_effort"] = cfg.reasoning_effort
    params.update(cfg.extra)
    return cast(BaseChatModel, init_chat_model(**params))


async def warmup_chat_model(model: BaseChatModel) -> float:
    """Fire one tiny request so connection setup / model load / first-call
    overhead is paid at startup, not on the user's first turn. Returns elapsed
    ms, or -1.0 on any failure (never raises)."""
    t0 = time.perf_counter()
    try:
        bound = model
        try:
            bound = cast(BaseChatModel, model.bind(max_tokens=1))
        except Exception:  # noqa: BLE001 - binding is best-effort
            bound = model
        await bound.ainvoke("hi")
    except Exception as exc:  # noqa: BLE001 - warmup must never break startup
        log.warning("llm_warmup_failed", error=str(exc)[:200])
        return -1.0
    ms = (time.perf_counter() - t0) * 1000
    log.info("llm_warmup", ms=round(ms, 1))
    return ms

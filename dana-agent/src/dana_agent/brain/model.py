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

from collections.abc import Iterator, Sequence
from typing import Any, cast

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable

from dana_agent.brain.providers import ALIASES, BRAIN_PROVIDERS
from dana_agent.config import LLMConfig
from dana_agent.errors import ConfigError

__all__ = ["BRAIN_PROVIDERS", "MockChatModel", "build_chat_model"]


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
    params: dict[str, Any] = {
        "model": cfg.model,
        "model_provider": provider,
        "temperature": cfg.temperature,
    }
    if cfg.api_key is not None:
        params["api_key"] = cfg.api_key.get_secret_value()
    if cfg.base_url is not None:
        params["base_url"] = cfg.base_url
    if provider == "openai":
        params["stream_usage"] = True  # token usage in streaming responses
    return cast(BaseChatModel, init_chat_model(**params))

"""Startup capability validation: passes for a coherent stack, fails fast and
actionably for unknown providers, unsupported languages, and missing tools."""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from dana_agent.capabilities import validate_app_config
from dana_agent.config import AppConfig
from dana_agent.errors import CapabilityError, ConfigError
from dana_agent.providers.base import Capabilities, LLMProvider, STTProvider
from dana_agent.providers.registry import register
from dana_agent.types import AudioFrame, LLMChunk, Message, Tool, Transcript


def test_default_stack_validates() -> None:
    validate_app_config(AppConfig())


def test_s2s_mode_validates() -> None:
    validate_app_config(AppConfig.model_validate({"conversation": {"mode": "s2s"}}))


def test_unknown_provider_fails() -> None:
    cfg = AppConfig.model_validate({"conversation": {"stt": {"provider": "nope"}}})
    with pytest.raises(ConfigError, match="unknown stt provider 'nope'"):
        validate_app_config(cfg)


def test_language_not_supported_fails(registry_snapshot: None) -> None:
    @register("stt", "english_only")
    class EnglishOnlySTT(STTProvider):
        def __init__(self, cfg: object) -> None: ...

        async def stream(self, audio: AsyncIterator[AudioFrame]) -> AsyncIterator[Transcript]:
            yield Transcript("", is_final=True)  # pragma: no cover

        @property
        def capabilities(self) -> Capabilities:
            return Capabilities(streaming=True, languages=frozenset({"en"}))

    cfg = AppConfig.model_validate(
        {"language": "vi", "conversation": {"stt": {"provider": "english_only"}}}
    )
    with pytest.raises(CapabilityError, match="does not support configured language"):
        validate_app_config(cfg)


def test_llm_without_tool_calling_fails(registry_snapshot: None) -> None:
    @register("llm", "no_tools")
    class NoToolsLLM(LLMProvider):
        def __init__(self, cfg: object) -> None: ...

        async def stream(
            self, messages: list[Message], tools: list[Tool] | None = None
        ) -> AsyncIterator[LLMChunk]:
            yield LLMChunk(done=True)  # pragma: no cover

        @property
        def capabilities(self) -> Capabilities:
            return Capabilities(
                streaming=True, languages=frozenset({"en", "vi"}), tool_calling=False
            )

    cfg = AppConfig.model_validate({"conversation": {"llm": {"provider": "no_tools"}}})
    with pytest.raises(CapabilityError, match="tool calling"):
        validate_app_config(cfg)

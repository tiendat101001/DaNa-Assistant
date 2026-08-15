"""Startup capability validation: passes for a coherent stack, fails fast and
actionably for unknown providers, unsupported languages, and missing tools."""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from dana_agent.capabilities import validate_app_config
from dana_agent.config import AppConfig
from dana_agent.errors import CapabilityError, ConfigError
from dana_agent.providers.base import Capabilities, STTProvider
from dana_agent.providers.registry import register
from dana_agent.types import AudioFrame, Transcript


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


def test_unknown_brain_llm_provider_fails() -> None:
    # The brain's LLM is built by init_chat_model; validate its provider against
    # the brain's supported set, not the LLMProvider registry.
    cfg = AppConfig.model_validate({"conversation": {"llm": {"provider": "frankenmodel"}}})
    with pytest.raises(ConfigError, match="unknown brain llm provider 'frankenmodel'"):
        validate_app_config(cfg)


def test_default_brain_provider_is_supported() -> None:
    # mock is a valid brain provider (offline echo model).
    validate_app_config(AppConfig())

"""Startup validation: check the assembled config against declared provider
capabilities and fail fast with an actionable error, instead of crashing
mid-conversation (provider-plugin-architecture skill).
"""

from __future__ import annotations

import dana_agent.providers  # noqa: F401  ensure all providers are registered
from dana_agent.brain.providers import BRAIN_PROVIDERS
from dana_agent.config import AppConfig
from dana_agent.errors import CapabilityError, ConfigError
from dana_agent.providers.base import Capabilities
from dana_agent.providers.registry import (
    build_actions,
    build_s2s,
    build_stt,
    build_tts,
    build_vad,
    build_wake,
)


def _require_language(caps: Capabilities, label: str, language: str) -> None:
    if language not in caps.languages:
        raise CapabilityError(
            f"{label} does not support configured language {language!r}; "
            f"it supports {sorted(caps.languages)}"
        )


def _require(condition: bool, message: str) -> None:  # noqa: FBT001
    if not condition:
        raise CapabilityError(message)


def validate_app_config(cfg: AppConfig) -> None:
    """Validate that the selected provider stack can actually serve the config.

    Raises ConfigError/CapabilityError on any unknown provider, unsupported
    language, or missing required capability. Constructing a provider here is
    cheap and side-effect-free (no network/model load at __init__).
    """
    lang = cfg.language

    # Wake gate must exist; language-agnostic phrase spotting, so no lang check.
    build_vad(cfg.vad)
    build_wake(cfg.wake)

    conv = cfg.conversation
    if conv.mode == "cascade":
        stt = build_stt(conv.stt)
        tts = build_tts(conv.tts)
        _require_language(stt.capabilities, f"STT provider {conv.stt.provider!r}", lang)
        _require_language(tts.capabilities, f"TTS provider {conv.tts.provider!r}", lang)
        # The brain's LLM is built by init_chat_model, so validate its provider
        # against the brain's supported set rather than the LLMProvider registry.
        if conv.llm.provider not in BRAIN_PROVIDERS:
            known = ", ".join(sorted(BRAIN_PROVIDERS))
            raise ConfigError(f"unknown brain llm provider {conv.llm.provider!r}; known: {known}")
    else:  # s2s
        s2s = build_s2s(conv.s2s)
        _require_language(s2s.capabilities, f"S2S provider {conv.s2s.provider!r}", lang)
        _require(
            s2s.capabilities.streaming,
            f"S2S provider {conv.s2s.provider!r} must stream",
        )

    # Action provider must exist (allowlists/gate policy validated by the schema).
    build_actions(cfg.actions)

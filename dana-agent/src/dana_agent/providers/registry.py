"""Registry + factory — config keys to provider classes, no if/elif ladders.

Providers self-register via ``@register(capability, name)`` at import time, so
adding a backend never touches the factory and the factory names no vendor.
Typed ``build_*`` helpers construct the right provider from its sub-config and
assert the ABC, turning a registry mistake into a loud startup error.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, TypeVar

from dana_agent.errors import ConfigError
from dana_agent.providers.base import (
    ActionProvider,
    LLMProvider,
    Provider,
    S2SProvider,
    STTProvider,
    TTSProvider,
    VADProvider,
    WakeWordProvider,
)

if TYPE_CHECKING:
    from dana_agent.config import (
        ActionsConfig,
        LLMConfig,
        S2SConfig,
        STTConfig,
        TTSConfig,
        VADConfig,
        WakeConfig,
    )

# capability -> {provider name -> provider class}
_REGISTRY: dict[str, dict[str, type[Provider]]] = {}

# Capabilities whose default "mock" provider is offline-safe. The shared
# contract test suite parametrizes over these.
CAPABILITIES: tuple[str, ...] = ("vad", "wake", "stt", "llm", "tts", "s2s", "actions")

P = TypeVar("P", bound=Provider)


def register(capability: str, name: str) -> Callable[[type[P]], type[P]]:
    """Class decorator: register ``cls`` under ``capability``/``name``."""

    def wrap(cls: type[P]) -> type[P]:
        bucket = _REGISTRY.setdefault(capability, {})
        if name in bucket and bucket[name] is not cls:
            raise ConfigError(f"duplicate provider registration {capability!r}/{name!r}")
        bucket[name] = cls
        return cls

    return wrap


def known(capability: str) -> list[str]:
    """Sorted list of registered provider names for a capability."""
    return sorted(_REGISTRY.get(capability, {}))


def get_class(capability: str, name: str) -> type[Provider]:
    """Look up a provider class, or raise an actionable ConfigError."""
    try:
        return _REGISTRY[capability][name]
    except KeyError:
        avail = ", ".join(known(capability)) or "(none registered)"
        raise ConfigError(f"unknown {capability} provider {name!r}; known: {avail}") from None


def _construct(capability: str, name: str, cfg: object) -> Provider:
    cls = get_class(capability, name)
    return cls(cfg)  # type: ignore[call-arg]  # providers take their sub-config


def _wrong_type(capability: str, name: str, expected: str) -> ConfigError:
    return ConfigError(f"{capability} provider {name!r} does not implement {expected}")


# --- typed factories -----------------------------------------------------
# Each narrows the constructed Provider to its ABC via isinstance, turning a
# registry mistake (wrong ABC under a name) into a loud, typed startup error.


def build_vad(cfg: VADConfig) -> VADProvider:
    p = _construct("vad", cfg.provider, cfg)
    if not isinstance(p, VADProvider):
        raise _wrong_type("vad", cfg.provider, "VADProvider")
    return p


def build_wake(cfg: WakeConfig) -> WakeWordProvider:
    p = _construct("wake", cfg.provider, cfg)
    if not isinstance(p, WakeWordProvider):
        raise _wrong_type("wake", cfg.provider, "WakeWordProvider")
    return p


def build_stt(cfg: STTConfig) -> STTProvider:
    p = _construct("stt", cfg.provider, cfg)
    if not isinstance(p, STTProvider):
        raise _wrong_type("stt", cfg.provider, "STTProvider")
    return p


def build_llm(cfg: LLMConfig) -> LLMProvider:
    p = _construct("llm", cfg.provider, cfg)
    if not isinstance(p, LLMProvider):
        raise _wrong_type("llm", cfg.provider, "LLMProvider")
    return p


def build_tts(cfg: TTSConfig) -> TTSProvider:
    p = _construct("tts", cfg.provider, cfg)
    if not isinstance(p, TTSProvider):
        raise _wrong_type("tts", cfg.provider, "TTSProvider")
    return p


def build_s2s(cfg: S2SConfig) -> S2SProvider:
    p = _construct("s2s", cfg.provider, cfg)
    if not isinstance(p, S2SProvider):
        raise _wrong_type("s2s", cfg.provider, "S2SProvider")
    return p


def build_actions(cfg: ActionsConfig) -> ActionProvider:
    p = _construct("actions", cfg.provider, cfg)
    if not isinstance(p, ActionProvider):
        raise _wrong_type("actions", cfg.provider, "ActionProvider")
    return p

"""Provider catalog for ``GET /providers``.

The dashboard renders capability-aware provider dropdowns from this: for each
capability it lists every registered provider and its declared
:class:`~dana_agent.providers.base.Capabilities` (streaming, languages,
self_hosted, tool_calling, latency_class). Constructing a provider to read its
``capabilities`` is cheap and side-effect-free (no network/model load at
``__init__`` — the same guarantee ``capabilities.validate_app_config`` relies
on). A provider that can't be introspected offline is still listed, with
``capabilities: null`` rather than being dropped.

The brain's chat-model providers are a separate axis (they go through
``init_chat_model``, not the ``LLMProvider`` registry), so they are exposed under
``brain_llm`` for the LLM dropdown.
"""

from __future__ import annotations

from typing import Any

import dana_agent.providers  # noqa: F401  ensure providers self-register
from dana_agent.brain.model import BRAIN_PROVIDERS
from dana_agent.config import (
    ActionsConfig,
    LLMConfig,
    S2SConfig,
    STTConfig,
    TTSConfig,
    VADConfig,
    WakeConfig,
)
from dana_agent.observability import get_logger
from dana_agent.providers.base import Provider
from dana_agent.providers.registry import _REGISTRY, CAPABILITIES

log = get_logger(__name__)

# How to build a default sub-config for each capability so a provider can be
# constructed and its capabilities read.
_CONFIG_FACTORY: dict[str, Any] = {
    "vad": VADConfig,
    "wake": WakeConfig,
    "stt": STTConfig,
    "llm": LLMConfig,
    "tts": TTSConfig,
    "s2s": S2SConfig,
    "actions": ActionsConfig,
}


def _capabilities_of(capability: str, name: str, cls: type[Provider]) -> dict[str, Any] | None:
    factory = _CONFIG_FACTORY.get(capability)
    if factory is None:
        return None
    try:
        provider = cls(factory(provider=name))  # type: ignore[call-arg]
        return provider.capabilities.model_dump(mode="json")
    except Exception as exc:  # noqa: BLE001 — a provider that needs an extra to
        # introspect is still worth listing; the UI just won't show its flags.
        log.debug("catalog_introspect_failed", capability=capability, provider=name, error=str(exc))
        return None


def build_provider_catalog() -> dict[str, Any]:
    """Return ``{capability: [{name, capabilities}], "brain_llm": [names]}``."""
    catalog: dict[str, Any] = {}
    for capability in CAPABILITIES:
        entries: list[dict[str, Any]] = []
        for name in sorted(_REGISTRY.get(capability, {})):
            cls = _REGISTRY[capability][name]
            entries.append({"name": name, "capabilities": _capabilities_of(capability, name, cls)})
        catalog[capability] = entries
    # Brain chat-model providers (init_chat_model axis) drive the LLM dropdown.
    catalog["brain_llm"] = sorted(BRAIN_PROVIDERS)
    return catalog

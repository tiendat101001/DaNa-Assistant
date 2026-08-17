"""Composition root: the one place that reads config and constructs concrete
providers. Everything else receives its dependencies (python-service-scaffold
skill), keeping modules unit-testable and swap-friendly.

Assembles and validates the provider stack, the wake gate, the action gate, and
(in cascade mode) the brain. Wiring these into the Pipecat pipeline and the
control API lands in later phases.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

import dana_agent.providers  # noqa: F401  register providers
from dana_agent.actions import ActionGate
from dana_agent.brain import Brain, HistoryPolicy, build_chat_model, open_checkpointer
from dana_agent.capabilities import validate_app_config
from dana_agent.config import AppConfig, get_settings
from dana_agent.observability import get_logger
from dana_agent.providers.base import (
    ActionProvider,
    S2SProvider,
    STTProvider,
    TTSProvider,
    VADProvider,
    WakeWordProvider,
)
from dana_agent.providers.registry import (
    build_actions,
    build_s2s,
    build_stt,
    build_tts,
    build_vad,
    build_wake,
)
from dana_agent.wake import WakeGate

log = get_logger(__name__)

# Map language codes to natural-language names the LLM will recognise.
_LANGUAGE_NAMES: dict[str, str] = {
    "en": "English",
    "vi": "Vietnamese (Tiếng Việt)",
}


def _effective_system_prompt(base: str, language: str) -> str:
    """Append a language instruction to the system prompt when not English."""
    if language == "en":
        return base
    lang_name = _LANGUAGE_NAMES.get(language, language)
    return f"{base}\nAlways reply in {lang_name}. Use natural, conversational tone."


@dataclass(slots=True)
class AssembledStack:
    """The constructed, validated set of providers for one run. In cascade mode
    ``s2s`` is None and the brain is present; in s2s mode ``stt``/``tts`` and the
    brain are None (the s2s model is its own turn). The brain owns the LLM via
    ``init_chat_model``, so there is no separate cascade LLM object here."""

    config: AppConfig
    vad: VADProvider
    wake: WakeWordProvider
    wake_gate: WakeGate
    actions: ActionProvider
    gate: ActionGate
    stt: STTProvider | None = None
    tts: TTSProvider | None = None
    s2s: S2SProvider | None = None
    brain: Brain | None = None
    _close_checkpointer: Callable[[], Awaitable[None]] | None = field(default=None, repr=False)

    async def aclose(self) -> None:
        for provider in (self.vad, self.wake, self.actions, self.stt, self.tts, self.s2s):
            if provider is not None:
                await provider.aclose()
        if self._close_checkpointer is not None:
            await self._close_checkpointer()


async def build_stack(cfg: AppConfig | None = None) -> AssembledStack:
    """Validate the config, then construct every selected provider. Async because
    the brain's SQLite checkpointer is opened here (it owns an async connection,
    closed by ``AssembledStack.aclose``). Defaults to the shared settings."""
    cfg = cfg or get_settings()
    validate_app_config(cfg)

    vad = build_vad(cfg.vad)
    wake = build_wake(cfg.wake)
    actions = build_actions(cfg.actions)
    gate = ActionGate(actions, policy=cfg.actions.gate_policy)
    stack = AssembledStack(
        config=cfg,
        vad=vad,
        wake=wake,
        wake_gate=WakeGate(vad, wake, inactivity_timeout_s=cfg.wake.inactivity_timeout_s),
        actions=actions,
        gate=gate,
    )
    if cfg.conversation.mode == "cascade":
        stack.stt = build_stt(cfg.conversation.stt)
        stack.tts = build_tts(cfg.conversation.tts)
        checkpointer, close_checkpointer = await open_checkpointer(cfg.brain.memory)
        stack._close_checkpointer = close_checkpointer
        stack.brain = Brain(
            build_chat_model(cfg.conversation.llm),
            gate,
            system_prompt=_effective_system_prompt(cfg.brain.system_prompt, cfg.language),
            max_steps=cfg.brain.max_steps,
            checkpointer=checkpointer,
            history=HistoryPolicy(
                strategy=cfg.brain.history_strategy,
                trigger_messages=cfg.brain.history_trigger_messages,
                keep_last=cfg.brain.history_keep_last,
            ),
        )
    else:
        stack.s2s = build_s2s(cfg.conversation.s2s)

    log.info(
        "stack_assembled",
        language=cfg.language,
        mode=cfg.conversation.mode,
        vad=cfg.vad.provider,
        wake=cfg.wake.provider,
        stt=cfg.conversation.stt.provider if stack.stt else None,
        llm=cfg.conversation.llm.provider if stack.brain else None,
        tts=cfg.conversation.tts.provider if stack.tts else None,
        s2s=cfg.conversation.s2s.provider if stack.s2s else None,
        actions=cfg.actions.provider,
        gate_policy=cfg.actions.gate_policy,
        brain=stack.brain is not None,
        memory=cfg.brain.memory.backend if stack.brain else None,
        local_only=cfg.privacy.local_only,
    )
    return stack

"""Composition root: the one place that reads config and constructs concrete
providers. Everything else receives its dependencies (python-service-scaffold
skill), keeping modules unit-testable and swap-friendly.

Phase 1 assembles and validates the provider stack. Wiring these into the
Pipecat pipeline, wake gate, brain, and control API lands in later phases.
"""

from __future__ import annotations

from dataclasses import dataclass

import dana_agent.providers  # noqa: F401  register providers
from dana_agent.actions import ActionGate
from dana_agent.brain import Brain
from dana_agent.capabilities import validate_app_config
from dana_agent.config import AppConfig
from dana_agent.observability import get_logger
from dana_agent.providers.base import (
    ActionProvider,
    LLMProvider,
    S2SProvider,
    STTProvider,
    TTSProvider,
    VADProvider,
    WakeWordProvider,
)
from dana_agent.providers.registry import (
    build_actions,
    build_llm,
    build_s2s,
    build_stt,
    build_tts,
    build_vad,
    build_wake,
)
from dana_agent.wake import WakeGate

log = get_logger(__name__)


@dataclass(slots=True)
class AssembledStack:
    """The constructed, validated set of providers for one run. In cascade mode
    ``s2s`` is None and the brain is present; in s2s mode ``stt``/``llm``/``tts``
    and the brain are None (the s2s model is its own turn)."""

    config: AppConfig
    vad: VADProvider
    wake: WakeWordProvider
    wake_gate: WakeGate
    actions: ActionProvider
    gate: ActionGate
    stt: STTProvider | None = None
    llm: LLMProvider | None = None
    tts: TTSProvider | None = None
    s2s: S2SProvider | None = None
    brain: Brain | None = None

    async def aclose(self) -> None:
        for provider in (
            self.vad,
            self.wake,
            self.actions,
            self.stt,
            self.llm,
            self.tts,
            self.s2s,
        ):
            if provider is not None:
                await provider.aclose()


def build_stack(cfg: AppConfig) -> AssembledStack:
    """Validate the config, then construct every selected provider."""
    validate_app_config(cfg)

    vad = build_vad(cfg.vad)
    wake = build_wake(cfg.wake)
    actions = build_actions(cfg.actions)
    gate = ActionGate(actions, policy=cfg.actions.gate_policy)
    stack = AssembledStack(
        config=cfg,
        vad=vad,
        wake=wake,
        wake_gate=WakeGate(
            vad,
            wake,
            inactivity_timeout_s=cfg.wake.inactivity_timeout_s,
        ),
        actions=actions,
        gate=gate,
    )
    if cfg.conversation.mode == "cascade":
        stack.stt = build_stt(cfg.conversation.stt)
        stack.llm = build_llm(cfg.conversation.llm)
        stack.tts = build_tts(cfg.conversation.tts)
        stack.brain = Brain(
            stack.llm,
            gate,
            system_prompt=cfg.brain.system_prompt,
            max_steps=cfg.brain.max_steps,
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
        llm=cfg.conversation.llm.provider if stack.llm else None,
        tts=cfg.conversation.tts.provider if stack.tts else None,
        s2s=cfg.conversation.s2s.provider if stack.s2s else None,
        actions=cfg.actions.provider,
        gate_policy=cfg.actions.gate_policy,
        brain=stack.brain is not None,
        local_only=cfg.privacy.local_only,
    )
    return stack

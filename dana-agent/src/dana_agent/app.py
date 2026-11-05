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
from typing import Any, TypeVar

import dana_agent.providers  # noqa: F401  register providers
from dana_agent.actions import ActionGate
from dana_agent.actions.audit import AuditLog, JsonlAuditSink
from dana_agent.brain import (
    Brain,
    HistoryPolicy,
    build_action_tools,
    build_chat_model,
    open_checkpointer,
)
from dana_agent.brain.tools import build_gated_tools
from dana_agent.capabilities import validate_app_config
from dana_agent.config import AppConfig, STTConfig, TTSConfig, get_settings
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
    # Plug-and-play MCP tool servers (own ActionGate sharing the audit log).
    mcp: ActionProvider | None = None
    mcp_gate: ActionGate | None = None
    _close_checkpointer: Callable[[], Awaitable[None]] | None = field(default=None, repr=False)

    async def aclose(self) -> None:
        for provider in (
            self.vad,
            self.wake,
            self.actions,
            self.stt,
            self.tts,
            self.s2s,
            self.mcp,
        ):
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
    # Durable, append-only JSONL audit trail alongside the in-memory snapshot.
    audit = AuditLog(sink=JsonlAuditSink(cfg.actions.audit_path))
    gate = ActionGate(actions, policy=cfg.actions.gate_policy, audit=audit)
    stack = AssembledStack(
        config=cfg,
        vad=vad,
        wake=wake,
        wake_gate=WakeGate(vad, wake, inactivity_timeout_s=cfg.wake.inactivity_timeout_s),
        actions=actions,
        gate=gate,
    )
    if cfg.conversation.mode == "cascade":
        stack.stt = build_stt(_with_local_model_path(cfg.conversation.stt, cfg))
        stack.tts = build_tts(_with_local_model_path(cfg.conversation.tts, cfg))
        checkpointer, close_checkpointer = await open_checkpointer(cfg.brain.memory)
        stack._close_checkpointer = close_checkpointer
        tools = build_action_tools(cfg.actions, gate)
        tools += await _build_mcp_tools(cfg, stack, audit)
        stack.brain = Brain(
            build_chat_model(cfg.conversation.llm),
            gate,
            system_prompt=_effective_system_prompt(cfg.brain.system_prompt, cfg.language),
            tools=tools,
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
        mcp_servers=len([s for s in cfg.actions.mcp_servers if s.enabled]),
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


async def _build_mcp_tools(cfg: AppConfig, stack: AssembledStack, audit: AuditLog) -> list[Any]:
    """Discover tools from configured MCP servers and wrap each one in a gated
    LangChain tool. MCP tools get their *own* ActionGate (same policy, same
    audit log) because the gate wraps exactly one provider. A server that fails
    to connect is skipped by ``MCPActions.connect`` — it never blocks startup."""
    servers = [s for s in cfg.actions.mcp_servers if s.enabled]
    if not servers:
        return []
    from dana_agent.providers.actions.mcp import MCPActions  # noqa: PLC0415

    mcp = MCPActions(servers)
    await mcp.connect()
    mcp_gate = ActionGate(mcp, policy=cfg.actions.gate_policy, audit=audit)
    stack.mcp = mcp
    stack.mcp_gate = mcp_gate
    specs = mcp.tool_specs()
    log.info("mcp_tools_ready", count=len(specs), names=[s.name for s in specs][:50])
    return list(build_gated_tools(specs, mcp_gate))


# Providers whose ``model`` may be a model-registry id (HF repo id) on local disk.
_REGISTRY_BACKED = frozenset({"litert", "piper"})

_SubCfg = TypeVar("_SubCfg", STTConfig, TTSConfig)


def _with_local_model_path(sub: _SubCfg, cfg: AppConfig) -> _SubCfg:
    """Resolve a registry id (``litert-community/whisper-tiny``) to its pulled
    directory under ``models.models_dir`` so local providers honour the
    configured models dir. Returns a *copy* — the live config (and what the UI
    shows) is untouched. An explicit ``model_path`` always wins."""
    if sub.provider not in _REGISTRY_BACKED or sub.model_path or "/" not in sub.model:
        return sub
    from dana_agent.models.store import model_dir  # noqa: PLC0415

    path = model_dir(cfg.models.models_dir, sub.model)
    if not path.is_dir():
        return sub  # provider reports a clear "pull it first" error at warm-up
    return sub.model_copy(update={"model_path": str(path)})

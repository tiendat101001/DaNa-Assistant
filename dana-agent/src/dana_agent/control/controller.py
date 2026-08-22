"""The ``Controller`` abstraction the API calls into.

FastAPI routes never touch ``ConversationSession``, the brain, or the audio
pipeline directly — they call this interface. That keeps the HTTP layer thin and
unit-testable: the route tests drive a fake controller, and the real
:class:`AgentController` wires the same calls into the assembled stack.

The controller owns the mutable runtime state the UI observes/controls: whether a
session is running, the mic gate, the FSM state, and the set of pending
confirmations awaiting a UI decision.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from dana_agent.brain.events import ConfirmationRequired, TurnComplete
from dana_agent.config import AppConfig
from dana_agent.control.config_apply import (
    ChangeStatus,
    FieldChange,
    diff_and_label,
    get_path,
    set_path,
)
from dana_agent.control.events import EventBus, EventType
from dana_agent.conversation.mute import MicGate
from dana_agent.observability import get_logger
from dana_agent.wake.state_machine import State

if TYPE_CHECKING:
    from dana_agent.app import AssembledStack

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class StatusInfo:
    """A snapshot of the agent for ``GET /status``."""

    state: str
    session_running: bool
    muted: bool
    uptime_s: float
    session_id: str | None
    language: str
    mode: str
    providers: dict[str, str | None]

    def to_json(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "session_running": self.session_running,
            "muted": self.muted,
            "uptime_s": round(self.uptime_s, 3),
            "session_id": self.session_id,
            "language": self.language,
            "mode": self.mode,
            "providers": self.providers,
        }


@dataclass(frozen=True, slots=True)
class ConfirmationInfo:
    """A destructive action paused on a confirmation (mirrors the voice prompt)."""

    id: str
    tool: str
    args_resolved: dict[str, Any]
    reversibility: str
    preview: str

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "tool": self.tool,
            "args_resolved": self.args_resolved,
            "reversibility": self.reversibility,
            "preview": self.preview,
        }


class Controller(ABC):
    """What the control API needs from the agent. Both the real controller and
    the test fake implement this."""

    @abstractmethod
    def config(self) -> AppConfig: ...

    @abstractmethod
    def status(self) -> StatusInfo: ...

    @abstractmethod
    async def start_session(self) -> None: ...

    @abstractmethod
    async def stop_session(self) -> None: ...

    @abstractmethod
    async def activate(self) -> None:
        """Trip the wake gate to ACTIVE (the UI Talk button — like saying the
        wake phrase, so speech goes straight to STT)."""

    @abstractmethod
    async def submit_text(self, text: str) -> None:
        """Run a typed user turn through the live pipeline (brain answers, TTS
        speaks the reply)."""

    @abstractmethod
    async def mute(self) -> None: ...

    @abstractmethod
    async def unmute(self) -> None: ...

    @abstractmethod
    def is_muted(self) -> bool: ...

    @abstractmethod
    async def resolve_confirmation(self, confirmation_id: str, *, approved: bool) -> bool:
        """Approve/deny a pending confirmation. Returns False if unknown."""

    @abstractmethod
    async def resolve_bot_check(self, *, bypassed: bool = True) -> bool:
        """Signal that an active bot check / CAPTCHA has been resolved or bypassed."""

    @abstractmethod
    def pending_confirmations(self) -> list[ConfirmationInfo]: ...

    @abstractmethod
    def audit_entries(self, *, since: float | None = None) -> list[dict[str, Any]]: ...

    @abstractmethod
    def apply_config(self, new_config: AppConfig) -> list[FieldChange]:
        """Validate+diff a new config, hot-apply safe fields, label the rest."""


class AgentController(Controller):
    """Real controller backed by an :class:`AssembledStack`.

    Session start/stop launch/stop a :class:`ConversationSession` as an asyncio
    task (the pipeline is imported lazily, so constructing the controller never
    pulls pipecat). The mic gate is shared with the pipeline's ``MuteProcessor``;
    the FSM state is tracked from ``state_transition`` events. Confirmations
    surfaced by the brain are held here until the UI (or voice) resolves them via
    ``Brain.resume_turn``.
    """

    def __init__(
        self,
        cfg: AppConfig,
        stack: AssembledStack,
        bus: EventBus,
        *,
        bypass_wake_gate: bool = False,
        now: Callable[[], float] = time.monotonic,
    ) -> None:
        self._cfg = cfg
        self._stack = stack
        self._bus = bus
        self._bypass_wake_gate = bypass_wake_gate
        self._now = now
        self._started_at = now()
        self._state = State.IDLE
        self._mic_gate = MicGate(on_change=self._on_mute_change)
        self._session: Any = None  # ConversationSession | None (lazy type)
        self._session_task: Any = None  # asyncio.Task | None
        self._pending: dict[str, ConfirmationInfo] = {}

    # --- observers ------------------------------------------------------

    @property
    def mic_gate(self) -> MicGate:
        return self._mic_gate

    def config(self) -> AppConfig:
        return self._cfg

    def on_state_transition(self, to_state: State) -> None:
        """Update the tracked FSM state (called from the pipeline event wiring)."""
        self._state = to_state

    def register_confirmation(self, event: ConfirmationRequired) -> ConfirmationInfo:
        """Record a brain confirmation and publish it for the UI.

        ``args_resolved`` is redacted before broadcast so secrets typed into
        password fields (``browser_type(into="Password", text="…")``) never
        reach WebSocket clients in cleartext."""
        from dana_agent.actions.audit import redact_arguments  # noqa: PLC0415

        info = ConfirmationInfo(
            id=event.id,
            tool=event.tool,
            args_resolved=redact_arguments(dict(event.arguments)),
            reversibility=str(event.reversibility.value),
            preview=_preview(event.tool, event.arguments),
        )
        self._pending[event.id] = info
        self._bus.publish(EventType.CONFIRMATION_REQUEST, info.to_json())
        return info

    def status(self) -> StatusInfo:
        conv = self._cfg.conversation
        providers: dict[str, str | None]
        if conv.mode == "cascade":
            providers = {
                "vad": self._cfg.vad.provider,
                "wake": self._cfg.wake.provider,
                "stt": conv.stt.provider,
                "llm": conv.llm.provider,
                "tts": conv.tts.provider,
                "actions": self._cfg.actions.provider,
            }
        else:
            providers = {
                "vad": self._cfg.vad.provider,
                "wake": self._cfg.wake.provider,
                "s2s": conv.s2s.provider,
                "actions": self._cfg.actions.provider,
            }
        return StatusInfo(
            state=str(self._state.value),
            session_running=self._session_task is not None,
            muted=self._mic_gate.muted,
            uptime_s=self._now() - self._started_at,
            session_id=self._session.session_id if self._session is not None else None,
            language=self._cfg.language,
            mode=conv.mode,
            providers=providers,
        )

    def is_muted(self) -> bool:
        return self._mic_gate.muted

    def pending_confirmations(self) -> list[ConfirmationInfo]:
        return list(self._pending.values())

    def audit_entries(self, *, since: float | None = None) -> list[dict[str, Any]]:
        entries = self._stack.gate.audit.entries
        return [e.to_record() for e in entries if since is None or e.at >= since]

    # --- controls -------------------------------------------------------

    async def mute(self) -> None:
        self._mic_gate.mute()

    async def unmute(self) -> None:
        self._mic_gate.unmute()

    def _on_mute_change(self, muted: bool) -> None:
        self._bus.publish(EventType.MIC_STATE, {"muted": muted})
        log.info("mic_state", muted=muted)

    async def start_session(self) -> None:
        import asyncio  # noqa: PLC0415

        from dana_agent.conversation import ConversationSession  # noqa: PLC0415

        if self._session_task is not None:
            return
        self._session = ConversationSession(
            self._cfg,
            self._stack,
            bypass_wake_gate=self._bypass_wake_gate,
            mic_gate=self._mic_gate,
        )
        self._session_task = asyncio.create_task(self._session.run())
        log.info("session_started", session_id=self._session.session_id)

    async def stop_session(self) -> None:
        import asyncio  # noqa: PLC0415

        if self._session is not None:
            await self._session.stop()
        if self._session_task is not None:
            self._session_task.cancel()
            try:
                await self._session_task
            except asyncio.CancelledError:
                pass
            except Exception:
                log.debug("session_task_suppressed_on_stop")
            self._session_task = None
        self._session = None
        self._state = State.IDLE
        log.info("session_stopped")

    async def activate(self) -> None:
        if self._session is None:
            raise RuntimeError("no session running — start the session first")
        await self._session.activate()

    async def submit_text(self, text: str) -> None:
        if self._session is None:
            raise RuntimeError("no session running — start the session first")
        await self._session.submit_text(text)

    async def resolve_confirmation(self, confirmation_id: str, *, approved: bool) -> bool:
        info = self._pending.pop(confirmation_id, None)
        if info is None:
            return False
        brain = self._stack.brain
        if brain is not None and self._session is not None:
            async for event in brain.resume_turn(
                thread_id=self._session.session_id, approved=approved
            ):
                if isinstance(event, TurnComplete):
                    self._bus.publish(
                        EventType.ASSISTANT_TURN, {"text": event.text, "interrupted": False}
                    )
        log.info("confirmation_resolved", id=confirmation_id, approved=approved)
        return True

    async def resolve_bot_check(self, *, bypassed: bool = True) -> bool:
        actions = getattr(self._stack, "actions", None)
        if actions is not None and hasattr(actions, "bypass_bot_check"):
            actions.bypass_bot_check()
        else:
            self._bus.publish(
                EventType.BOT_CHECK,
                {"active": False, "url": "", "kind": "bypassed", "bypassed": bypassed},
            )
        log.info("bot_check_resolved", bypassed=bypassed)
        return True

    def apply_config(self, new_config: AppConfig) -> list[FieldChange]:
        changes = diff_and_label(self._cfg, new_config)
        for change in changes:
            if change.status is ChangeStatus.APPLIED:
                set_path(self._cfg, change.path, get_path(new_config, change.path))
                self._hot_side_effect(change.path)
        if changes:
            self._bus.publish(EventType.CONFIG_CHANGED, {"changes": [c.to_json() for c in changes]})
        return changes

    def _hot_side_effect(self, path: str) -> None:
        """Push a hot field into the live stack so ``applied`` is truthful.

        Each branch here corresponds to a path in
        :data:`~dana_agent.control.config_apply.HOT_APPLY_PATHS`; the two must
        stay in lock-step (a field is only labelled ``applied`` if it is re-bound
        here)."""
        if path == "log_level":
            import logging  # noqa: PLC0415

            level = logging.getLevelNamesMapping().get(self._cfg.log_level.upper())
            if level is not None:
                logging.getLogger().setLevel(level)
        elif path == "actions.gate_policy":
            # ActionGate.decide re-reads the policy per action → live effect now.
            self._stack.gate.set_policy(self._cfg.actions.gate_policy)
        elif path == "brain.max_steps":
            # Brain._stream re-reads the budget per turn → live on the next turn.
            if self._stack.brain is not None:
                self._stack.brain.set_max_steps(self._cfg.brain.max_steps)


def _preview(tool: str, arguments: dict[str, Any]) -> str:
    """A short human-readable preview of what the confirmation would run."""
    from dana_agent.actions.audit import redact_arguments  # noqa: PLC0415

    safe = redact_arguments(arguments)
    arg_str = ", ".join(f"{k}={v!r}" for k, v in safe.items())
    return f"{tool}({arg_str})"

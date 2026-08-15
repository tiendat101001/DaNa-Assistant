"""The wake gate: mic frames -> VAD -> KWS -> state machine, forwarding audio
to the conversation session only while ACTIVE.

Signal chain (voice-agent-architecture skill):

    mic -> VAD (cheap gate) -> Wake engine (KWS) -> [on hit] ACTIVE
        -> forward audio to session -> [end phrase | inactivity timeout] -> IDLE

The gate never runs the wake engine on silence (VAD gates it), models the
lifecycle as an explicit state machine, and ends a session on the end phrase
**and** an inactivity timeout — never the phrase alone, since users forget it.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from time import monotonic

from dana_agent.observability import get_logger
from dana_agent.providers.base import VADProvider, WakeWordProvider
from dana_agent.types import AudioFrame, WakeEventType
from dana_agent.wake.state_machine import (
    State,
    StateMachine,
    StateTransition,
    TransitionCause,
)

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class FrameOutcome:
    """Result of feeding one frame: the resulting state, whether the frame
    should be forwarded to the conversation session, and any transition fired."""

    state: State
    forward: bool
    transition: StateTransition | None


class WakeGate:
    def __init__(
        self,
        vad: VADProvider,
        wake: WakeWordProvider,
        *,
        inactivity_timeout_s: float,
        now: Callable[[], float] = monotonic,
        on_transition: Callable[[StateTransition], None] | None = None,
    ) -> None:
        self._vad = vad
        self._wake = wake
        self._timeout = inactivity_timeout_s
        self._now = now
        self._on_transition = on_transition
        self._sm = StateMachine(on_transition=self._emit, now=now)
        self._last_activity = now()

    @property
    def state(self) -> State:
        return self._sm.state

    def _emit(self, transition: StateTransition) -> None:
        log.info(
            "state_transition",
            from_state=transition.from_state,
            to_state=transition.to_state,
            cause=transition.cause,
        )
        if self._on_transition is not None:
            self._on_transition(transition)

    def process_frame(self, frame: AudioFrame) -> FrameOutcome:
        """Feed one captured audio frame; advance the state machine."""
        speech = self._vad.is_speech(frame)

        if self._sm.state is State.IDLE:
            if speech:
                event = self._wake.process(frame)
                if event is not None and event.type is WakeEventType.WAKE:
                    transition = self._sm.fire(TransitionCause.WAKE)
                    self._last_activity = self._now()
                    return FrameOutcome(self._sm.state, forward=False, transition=transition)
            return FrameOutcome(State.IDLE, forward=False, transition=None)

        # ACTIVE ------------------------------------------------------------
        now = self._now()
        if speech:
            self._last_activity = now
        elif now - self._last_activity >= self._timeout:
            transition = self._sm.fire(TransitionCause.INACTIVITY_TIMEOUT)
            return FrameOutcome(self._sm.state, forward=False, transition=transition)

        event = self._wake.process(frame)
        if event is not None and event.type is WakeEventType.END:
            transition = self._sm.fire(TransitionCause.END_PHRASE)
            return FrameOutcome(self._sm.state, forward=False, transition=transition)

        # Still active: this frame belongs to the conversation session.
        return FrameOutcome(State.ACTIVE, forward=True, transition=None)

    def cancel(self) -> StateTransition | None:
        """Force the session to end (e.g. error or explicit stop)."""
        return self._sm.fire(TransitionCause.CANCEL)

    async def run(self, frames: AsyncIterator[AudioFrame]) -> AsyncIterator[AudioFrame]:
        """Drive the gate over an audio stream, yielding only the frames that
        belong to an active conversation session (Phase 3 wires the sink)."""
        async for frame in frames:
            outcome = self.process_frame(frame)
            if outcome.forward:
                yield frame

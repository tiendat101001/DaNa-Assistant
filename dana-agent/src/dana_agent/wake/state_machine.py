"""Explicit IDLE/ACTIVE state machine.

States and transitions are first-class (an enum + a transition table), not
implicit boolean flags scattered across callbacks — the single most important
structural decision for debuggability (voice-agent-architecture skill). Every
edge emits a StateTransition so the UI and logs can observe it.

    IDLE  --wake-->  ACTIVE
    ACTIVE  --end phrase | inactivity timeout | cancel | error-->  IDLE
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from time import monotonic


class State(StrEnum):
    IDLE = "idle"
    ACTIVE = "active"


class TransitionCause(StrEnum):
    WAKE = "wake"
    END_PHRASE = "end_phrase"
    INACTIVITY_TIMEOUT = "inactivity_timeout"
    CANCEL = "cancel"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class StateTransition:
    from_state: State
    to_state: State
    cause: TransitionCause
    at: float


# (current state, cause) -> next state. Any pair not present is a no-op.
_TABLE: dict[tuple[State, TransitionCause], State] = {
    (State.IDLE, TransitionCause.WAKE): State.ACTIVE,
    (State.ACTIVE, TransitionCause.END_PHRASE): State.IDLE,
    (State.ACTIVE, TransitionCause.INACTIVITY_TIMEOUT): State.IDLE,
    (State.ACTIVE, TransitionCause.CANCEL): State.IDLE,
    (State.ACTIVE, TransitionCause.ERROR): State.IDLE,
}


class StateMachine:
    """Minimal transition-table state machine. ``on_transition`` fires on every
    actual state change (never on a no-op)."""

    def __init__(
        self,
        *,
        on_transition: Callable[[StateTransition], None] | None = None,
        now: Callable[[], float] = monotonic,
    ) -> None:
        self._state = State.IDLE
        self._on_transition = on_transition
        self._now = now

    @property
    def state(self) -> State:
        return self._state

    def fire(self, cause: TransitionCause) -> StateTransition | None:
        """Apply ``cause`` from the current state. Returns the StateTransition if
        it changed state, else None (unknown/invalid transition from here)."""
        target = _TABLE.get((self._state, cause))
        if target is None or target == self._state:
            return None
        transition = StateTransition(self._state, target, cause, self._now())
        self._state = target
        if self._on_transition is not None:
            self._on_transition(transition)
        return transition

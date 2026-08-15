"""IDLE/ACTIVE state machine: transitions, no-ops, and transition events."""

from __future__ import annotations

from dana_agent.wake.state_machine import (
    State,
    StateMachine,
    StateTransition,
    TransitionCause,
)


def test_initial_state_is_idle() -> None:
    assert StateMachine().state is State.IDLE


def test_wake_moves_idle_to_active() -> None:
    sm = StateMachine(now=lambda: 1.5)
    tr = sm.fire(TransitionCause.WAKE)
    assert tr is not None
    assert tr.from_state is State.IDLE
    assert tr.to_state is State.ACTIVE
    assert tr.cause is TransitionCause.WAKE
    assert tr.at == 1.5
    assert sm.state is State.ACTIVE


def test_end_phrase_moves_active_to_idle() -> None:
    sm = StateMachine()
    sm.fire(TransitionCause.WAKE)
    assert sm.fire(TransitionCause.END_PHRASE) is not None
    assert sm.state is State.IDLE


def test_timeout_cancel_error_all_end_the_session() -> None:
    for cause in (
        TransitionCause.INACTIVITY_TIMEOUT,
        TransitionCause.CANCEL,
        TransitionCause.ERROR,
    ):
        sm = StateMachine()
        sm.fire(TransitionCause.WAKE)
        assert sm.fire(cause) is not None
        assert sm.state is State.IDLE


def test_invalid_transition_is_noop() -> None:
    sm = StateMachine()
    assert sm.fire(TransitionCause.END_PHRASE) is None  # not ACTIVE
    assert sm.state is State.IDLE


def test_wake_while_active_is_noop() -> None:
    sm = StateMachine()
    sm.fire(TransitionCause.WAKE)
    assert sm.fire(TransitionCause.WAKE) is None
    assert sm.state is State.ACTIVE


def test_on_transition_fires_only_on_real_changes() -> None:
    seen: list[StateTransition] = []
    sm = StateMachine(on_transition=seen.append)
    sm.fire(TransitionCause.WAKE)
    sm.fire(TransitionCause.END_PHRASE)
    sm.fire(TransitionCause.END_PHRASE)  # no-op, must not emit
    assert [t.cause for t in seen] == [
        TransitionCause.WAKE,
        TransitionCause.END_PHRASE,
    ]

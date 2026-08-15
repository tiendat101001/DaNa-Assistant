"""Wake gate integration: VAD + KWS + state machine driven by frame sequences.
Assertions target state transitions and forwarding decisions, not audio bytes
(voice-agent-architecture skill)."""

from __future__ import annotations

from dana_agent.config import VADConfig, WakeConfig
from dana_agent.providers.registry import build_vad, build_wake
from dana_agent.wake import State, StateTransition, WakeGate
from dana_agent.wake.state_machine import TransitionCause
from tests.conftest import async_iter, collect
from tests.fixtures.sequences import end_frame, silence_frame, speech_frame, wake_frame


class FakeClock:
    def __init__(self, t: float = 0.0) -> None:
        self.t = t

    def now(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


def make_gate(
    *,
    timeout: float = 30.0,
    clock: FakeClock | None = None,
    transitions: list[StateTransition] | None = None,
) -> WakeGate:
    clock = clock or FakeClock()
    return WakeGate(
        build_vad(VADConfig()),
        build_wake(WakeConfig()),
        inactivity_timeout_s=timeout,
        now=clock.now,
        on_transition=(transitions.append if transitions is not None else None),
    )


def test_idle_ignores_silence_and_non_wake_speech() -> None:
    gate = make_gate()
    assert gate.process_frame(silence_frame()).forward is False
    outcome = gate.process_frame(speech_frame("hello there"))
    assert outcome.state is State.IDLE
    assert outcome.forward is False


def test_wake_activates_but_does_not_forward_the_wake_frame() -> None:
    gate = make_gate()
    outcome = gate.process_frame(wake_frame())
    assert outcome.transition is not None
    assert outcome.transition.cause is TransitionCause.WAKE
    assert gate.state == State.ACTIVE
    assert outcome.forward is False


def test_active_forwards_conversation_speech() -> None:
    gate = make_gate()
    gate.process_frame(wake_frame())
    outcome = gate.process_frame(speech_frame("what is the weather"))
    assert outcome.state is State.ACTIVE
    assert outcome.forward is True


def test_end_phrase_returns_to_idle() -> None:
    gate = make_gate()
    gate.process_frame(wake_frame())
    outcome = gate.process_frame(end_frame())
    assert outcome.transition is not None
    assert outcome.transition.cause is TransitionCause.END_PHRASE
    assert gate.state == State.IDLE
    assert outcome.forward is False


def test_inactivity_timeout_returns_to_idle() -> None:
    clock = FakeClock()
    gate = make_gate(timeout=30.0, clock=clock)
    gate.process_frame(wake_frame())
    clock.advance(31.0)
    outcome = gate.process_frame(silence_frame())
    assert outcome.transition is not None
    assert outcome.transition.cause is TransitionCause.INACTIVITY_TIMEOUT
    assert gate.state == State.IDLE


def test_speech_resets_the_inactivity_timer() -> None:
    clock = FakeClock()
    gate = make_gate(timeout=30.0, clock=clock)
    gate.process_frame(wake_frame())
    clock.advance(20.0)
    gate.process_frame(speech_frame("still talking"))  # resets timer
    clock.advance(20.0)  # 40s since wake, but only 20s since last speech
    outcome = gate.process_frame(silence_frame())
    assert gate.state == State.ACTIVE  # no timeout: the earlier speech reset it
    assert outcome.transition is None


def test_cancel_ends_the_session() -> None:
    gate = make_gate()
    gate.process_frame(wake_frame())
    tr = gate.cancel()
    assert tr is not None and tr.cause is TransitionCause.CANCEL
    assert gate.state == State.IDLE


async def test_run_forwards_only_active_session_frames() -> None:
    transitions: list[StateTransition] = []
    gate = make_gate(transitions=transitions)
    sequence = [
        silence_frame(),  # IDLE, ignored
        speech_frame("noise"),  # IDLE, not the wake word
        wake_frame(),  # -> ACTIVE
        speech_frame("turn on the lights"),  # forwarded
        speech_frame("please"),  # forwarded
        end_frame(),  # -> IDLE
        speech_frame("ignored now"),  # IDLE again
    ]
    forwarded = await collect(gate.run(async_iter(sequence)))
    assert [f.pcm.decode() for f in forwarded] == ["turn on the lights", "please"]
    assert [t.cause for t in transitions] == [
        TransitionCause.WAKE,
        TransitionCause.END_PHRASE,
    ]

"""MicGate — the hard mic mute shared between control and the pipeline."""

from __future__ import annotations

from dana_agent.conversation.mute import MicGate


def test_starts_unmuted() -> None:
    assert MicGate().muted is False


def test_mute_unmute_and_change_detection() -> None:
    changes: list[bool] = []
    gate = MicGate(on_change=changes.append)

    assert gate.mute() is True  # changed
    assert gate.muted is True
    assert gate.mute() is False  # no-op, already muted
    assert gate.unmute() is True
    assert gate.unmute() is False  # no-op

    # on_change fires once per real transition only.
    assert changes == [True, False]


def test_toggle_returns_new_state() -> None:
    gate = MicGate()
    assert gate.toggle() is True
    assert gate.toggle() is False


def test_initial_muted_state() -> None:
    assert MicGate(muted=True).muted is True

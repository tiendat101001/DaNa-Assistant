"""The structlog→EventBus bridge forwards latency and audit events."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from dana_agent.control.events import EventBus
from dana_agent.control.log_bridge import install_log_bridge, uninstall_log_bridge
from dana_agent.observability.logging import set_event_sink


@pytest.fixture(autouse=True)
def _clear_sink() -> Iterator[None]:
    # Ensure no bridge leaks into other tests.
    yield
    set_event_sink(None)


def test_latency_event_becomes_latency_span() -> None:
    bus = EventBus()
    sub = bus.subscribe()
    _sink(bus)({"event": "latency", "stage": "stt", "turn_id": "t1", "ms": 42})
    env = sub._queue.get_nowait()
    assert env.type == "latency_span"
    assert env.data == {"stage": "stt", "turn_id": "t1", "ms": 42}


def test_install_and_forward_latency() -> None:
    bus = EventBus()
    sub = bus.subscribe()
    install_log_bridge(bus)
    try:
        from dana_agent.observability.logging import _EVENT_SINK

        assert _EVENT_SINK is not None
        _EVENT_SINK({"event": "latency", "stage": "llm_ttft", "turn_id": "t9", "ms": 100})
    finally:
        uninstall_log_bridge()
    env = sub._queue.get_nowait()
    assert env.type == "latency_span"
    assert env.data == {"stage": "llm_ttft", "turn_id": "t9", "ms": 100}


def test_action_audit_becomes_audit_entry() -> None:
    bus = EventBus()
    sub = bus.subscribe()
    _sink(bus)(
        {
            "event": "action_audit",
            "tool": "click",
            "decision": "allow",
            "reversibility": "reversible",
            "executed": True,
            "ok": True,
        }
    )
    env = sub._queue.get_nowait()
    assert env.type == "audit_entry"
    assert env.data["tool"] == "click"
    assert env.data["decision"] == "allow"


def test_unrelated_event_is_ignored() -> None:
    bus = EventBus()
    sub = bus.subscribe()
    _sink(bus)({"event": "some_other_log", "x": 1})
    assert sub._queue.empty()


# --- carry-over B: live pipeline turn events --------------------------------


def test_wake_gate_active_becomes_state_transition() -> None:
    bus = EventBus()
    sub = bus.subscribe()
    _sink(bus)({"event": "wake_gate_active", "method": "assemblyai_stt"})
    env = sub._queue.get_nowait()
    assert env.type == "state_transition"
    assert env.data == {"from": "idle", "to": "active", "cause": "wake"}


def test_wake_gate_idle_becomes_state_transition() -> None:
    bus = EventBus()
    sub = bus.subscribe()
    _sink(bus)({"event": "wake_gate_idle", "cause": "end_phrase"})
    env = sub._queue.get_nowait()
    assert env.type == "state_transition"
    assert env.data == {"from": "active", "to": "idle", "cause": "end_phrase"}


def test_transcript_final_becomes_transcript_final() -> None:
    bus = EventBus()
    sub = bus.subscribe()
    _sink(bus)({"event": "transcript_final", "text": "hello dana", "role": "user", "session": "s1"})
    env = sub._queue.get_nowait()
    assert env.type == "transcript_final"
    assert env.data == {"text": "hello dana", "role": "user"}
    assert env.session_id == "s1"


def test_transcript_final_defaults_role_to_user() -> None:
    bus = EventBus()
    sub = bus.subscribe()
    _sink(bus)({"event": "transcript_final", "text": "hi", "session": "s2"})
    env = sub._queue.get_nowait()
    assert env.data["role"] == "user"


def test_assistant_turn_becomes_assistant_turn() -> None:
    bus = EventBus()
    sub = bus.subscribe()
    _sink(bus)({"event": "assistant_turn", "text": "hi there", "session": "s1"})
    env = sub._queue.get_nowait()
    assert env.type == "assistant_turn"
    assert env.data == {"text": "hi there", "interrupted": False}


def test_confirmation_required_becomes_confirmation_request() -> None:
    bus = EventBus()
    sub = bus.subscribe()
    _sink(bus)(
        {
            "event": "confirmation_required",
            "id": "c1",
            "tool": "click",
            "reversibility": "destructive",
            "args_resolved": {"text": "delete"},
            "preview": "click(text='delete')",
            "session": "s1",
        }
    )
    env = sub._queue.get_nowait()
    assert env.type == "confirmation_request"
    assert env.data["id"] == "c1"
    assert env.data["tool"] == "click"
    assert env.data["reversibility"] == "destructive"
    assert env.data["preview"] == "click(text='delete')"
    assert env.data["args_resolved"] == {"text": "delete"}


def _sink(bus: EventBus):  # type: ignore[no-untyped-def]
    from dana_agent.control.log_bridge import make_log_sink

    return make_log_sink(bus)

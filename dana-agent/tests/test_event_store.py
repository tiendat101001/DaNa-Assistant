"""Unit tests for SessionEventStore abstraction and FileSessionEventStore."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from dana_agent.control.events import EventBus, EventType
from dana_agent.events.store import (
    FileSessionEventStore,
    set_event_store,
)


@pytest.fixture(autouse=True)
def _reset_global_store() -> Iterator[None]:
    set_event_store(None)
    yield
    set_event_store(None)


def test_file_event_store_saves_and_retrieves_events(tmp_path: Path) -> None:
    store = FileSessionEventStore(base_dir=tmp_path)
    session_id = "sess-001"

    store.save_event(session_id, "session_start", {"mode": "cascade"}, seq=1)
    store.save_event(session_id, "transcript_final", {"text": "hello", "role": "user"}, seq=2)
    store.save_event(session_id, "assistant_turn", {"text": "hi user"}, seq=3)

    events = store.get_events(session_id)
    assert len(events) == 3
    assert events[0]["type"] == "session_start"
    assert events[0]["seq"] == 1
    assert events[1]["data"]["text"] == "hello"
    assert events[2]["type"] == "assistant_turn"


def test_file_event_store_list_sessions(tmp_path: Path) -> None:
    store = FileSessionEventStore(base_dir=tmp_path)
    store.save_event("session-a", "state_transition", {"to": "active"})
    store.save_event("session-b", "state_transition", {"to": "idle"})

    sessions = store.list_sessions()
    assert sessions == ["session-a", "session-b"]


def test_file_event_store_limit(tmp_path: Path) -> None:
    store = FileSessionEventStore(base_dir=tmp_path)
    session_id = "sess-limit"

    for i in range(5):
        store.save_event(session_id, "latency_span", {"i": i})

    events = store.get_events(session_id, limit=2)
    assert len(events) == 2
    assert events[0]["data"]["i"] == 3
    assert events[1]["data"]["i"] == 4


def test_event_bus_integration_saves_to_store(tmp_path: Path) -> None:
    store = FileSessionEventStore(base_dir=tmp_path)
    set_event_store(store)

    bus = EventBus()
    bus.publish(
        EventType.TRANSCRIPT_FINAL,
        {"text": "test integration", "role": "user"},
        session_id="session-bus",
    )

    events = store.get_events("session-bus")
    assert len(events) == 1
    assert events[0]["type"] == "transcript_final"
    assert events[0]["data"]["text"] == "test integration"
    assert events[0]["session_id"] == "session-bus"

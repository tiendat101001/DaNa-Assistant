"""Session event storage abstraction.

Provides a pluggable storage interface for session-bound events (e.g. transcripts,
assistant turns, latency metrics, and audit entries).

Default implementation: :class:`FileSessionEventStore` writes events to static
JSON Lines files inside `.dana/events/{session_id}/events.jsonl`.

Future implementations (such as SQLite or PostgreSQL event stores) can implement
:class:`BaseSessionEventStore` without affecting event producers or consumers.
"""

from __future__ import annotations

import json
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from dana_agent.observability import get_logger

log = get_logger(__name__)

# Default base directory for file-based event store
_DEFAULT_EVENTS_DIR = Path.cwd() / ".dana" / "events"


class BaseSessionEventStore(ABC):
    """Abstract base class for session event persistence."""

    @abstractmethod
    def save_event(
        self,
        session_id: str,
        event_type: str,
        data: Mapping[str, Any] | None = None,
        *,
        timestamp: float | None = None,
        seq: int | None = None,
    ) -> None:
        """Persist a single session event."""

    @abstractmethod
    def get_events(
        self,
        session_id: str,
        *,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieve stored events for a given session ID."""

    @abstractmethod
    def list_sessions(self) -> list[str]:
        """Return all known session IDs in the store."""

    def close(self) -> None:  # noqa: B027 — deliberate optional hook, not abstract
        """Optional cleanup resources when shutting down."""


class FileSessionEventStore(BaseSessionEventStore):
    """File-based session event store.

    Stores session events in `.dana/events/{session_id}/events.jsonl` as JSON Lines.
    """

    def __init__(self, base_dir: Path | str | None = None) -> None:
        self.base_dir = Path(base_dir) if base_dir is not None else _DEFAULT_EVENTS_DIR
        self._locks: dict[str, threading.Lock] = {}
        self._global_lock = threading.Lock()

    def _get_lock(self, session_id: str) -> threading.Lock:
        with self._global_lock:
            if session_id not in self._locks:
                self._locks[session_id] = threading.Lock()
            return self._locks[session_id]

    def _session_dir(self, session_id: str) -> Path:
        return self.base_dir / session_id

    def _events_file(self, session_id: str) -> Path:
        return self._session_dir(session_id) / "events.jsonl"

    def save_event(
        self,
        session_id: str,
        event_type: str,
        data: Mapping[str, Any] | None = None,
        *,
        timestamp: float | None = None,
        seq: int | None = None,
    ) -> None:
        if not session_id:
            return

        ts = timestamp if timestamp is not None else time.time()
        record = {
            "type": str(event_type),
            "ts": ts,
            "session_id": session_id,
            "seq": seq,
            "data": dict(data) if data is not None else {},
        }

        session_dir = self._session_dir(session_id)
        events_file = self._events_file(session_id)

        try:
            lock = self._get_lock(session_id)
            with lock:
                session_dir.mkdir(parents=True, exist_ok=True)
                with events_file.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as exc:  # noqa: BLE001
            log.warning("failed_to_save_session_event", session_id=session_id, error=str(exc))

    def get_events(
        self,
        session_id: str,
        *,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        events_file = self._events_file(session_id)
        if not events_file.exists():
            return []

        events: list[dict[str, Any]] = []
        lock = self._get_lock(session_id)
        with lock:
            with events_file.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            events.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue

        if limit is not None and limit > 0:
            events = events[-limit:]

        return events

    def list_sessions(self) -> list[str]:
        if not self.base_dir.exists():
            return []

        sessions: list[str] = []
        for child in self.base_dir.iterdir():
            if child.is_dir() and (child / "events.jsonl").exists():
                sessions.append(child.name)
        return sorted(sessions)


# Global singleton instance management
_EVENT_STORE: BaseSessionEventStore | None = None


def get_event_store() -> BaseSessionEventStore:
    """Return the global event store instance, initializing FileSessionEventStore by default."""
    global _EVENT_STORE
    if _EVENT_STORE is None:
        _EVENT_STORE = FileSessionEventStore()
    return _EVENT_STORE


def set_event_store(store: BaseSessionEventStore | None) -> None:
    """Override or reset the global event store instance."""
    global _EVENT_STORE
    _EVENT_STORE = store

"""In-process pub/sub event bus + the typed WebSocket envelope.

The bus decouples event *producers* (the pipeline, the brain, the FSM, the audit
log, the structlog latency taps) from event *consumers* (WebSocket clients).
Producers call :meth:`EventBus.publish` — a synchronous, non-blocking fan-out —
so a structlog processor or a pipeline callback can emit without awaiting. Each
subscriber owns a bounded queue; on overflow the oldest event is dropped (a live
dashboard cares about *recent* state, never about replaying a backlog).

Envelope shape is the contract in docs/PLAN.md §5::

    {"type": str, "ts": float, "session_id": str | None, "seq": int, "data": {...}}

``seq`` is a monotonic per-bus counter so a client can detect gaps caused by the
drop policy. ``conversation/*`` never imports this module — the bus is injected.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from dana_agent.observability import get_logger

log = get_logger(__name__)

# Bounded per-subscriber queue: enough to smooth bursts, small enough that a slow
# or vanished client can never grow memory without bound.
_DEFAULT_MAXSIZE = 256


class EventType(StrEnum):
    """The WS event types from docs/PLAN.md §5. String-valued so payloads stay
    JSON-native and the UI can switch on ``envelope.type`` directly."""

    STATE_TRANSITION = "state_transition"
    TRANSCRIPT_PARTIAL = "transcript_partial"
    TRANSCRIPT_FINAL = "transcript_final"
    ASSISTANT_TURN = "assistant_turn"
    LATENCY_SPAN = "latency_span"
    CONFIRMATION_REQUEST = "confirmation_request"
    AUDIT_ENTRY = "audit_entry"
    CONFIG_CHANGED = "config_changed"
    MIC_STATE = "mic_state"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class Envelope:
    """One server-push event. Frozen so a published event is never mutated after
    fan-out (all subscribers see identical bytes)."""

    type: str
    ts: float
    session_id: str | None
    seq: int
    data: Mapping[str, Any]

    def to_json(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "ts": self.ts,
            "session_id": self.session_id,
            "seq": self.seq,
            "data": dict(self.data),
        }


class Subscription:
    """A single consumer's view of the bus: an async iterator over envelopes.

    Use as an async context manager so the subscriber is always unregistered,
    even if the WebSocket drops mid-stream::

        async with bus.subscribe() as sub:
            async for envelope in sub:
                await ws.send_json(envelope.to_json())

    The subscription binds to the event loop it was created on. ``publish`` may be
    called from any thread (a structlog processor on a worker thread, a tray
    callback); the enqueue is then marshalled back onto that loop with
    ``call_soon_threadsafe`` so the ``asyncio.Queue`` is only ever touched from
    its owning loop (asyncio queues are not thread-safe).
    """

    def __init__(self, bus: EventBus, maxsize: int) -> None:
        self._bus = bus
        self._queue: asyncio.Queue[Envelope] = asyncio.Queue(maxsize=maxsize)
        self.dropped = 0
        try:
            self._loop: asyncio.AbstractEventLoop | None = asyncio.get_running_loop()
        except RuntimeError:
            self._loop = None

    def _offer(self, envelope: Envelope) -> None:
        """Non-blocking enqueue with a drop-oldest policy (called by the bus).

        Safe to call from any thread: if a different thread owns the loop, the
        enqueue is scheduled on it; otherwise it runs inline."""
        loop = self._loop
        if loop is not None and loop.is_running():
            try:
                running = asyncio.get_running_loop()
            except RuntimeError:
                running = None
            if running is not loop:
                loop.call_soon_threadsafe(self._put, envelope)
                return
        self._put(envelope)

    def _put(self, envelope: Envelope) -> None:
        try:
            self._queue.put_nowait(envelope)
        except asyncio.QueueFull:
            # Drop the oldest to make room — a live view favours recency.
            with contextlib.suppress(asyncio.QueueEmpty):
                self._queue.get_nowait()
                self.dropped += 1
            with contextlib.suppress(asyncio.QueueFull):
                self._queue.put_nowait(envelope)

    async def get(self) -> Envelope:
        return await self._queue.get()

    def __aiter__(self) -> AsyncIterator[Envelope]:
        return self._iter()

    async def _iter(self) -> AsyncIterator[Envelope]:
        while True:
            yield await self._queue.get()

    async def __aenter__(self) -> Subscription:
        return self

    async def __aexit__(self, *exc: object) -> None:
        self._bus.unsubscribe(self)


class EventBus:
    """Fan-out pub/sub over in-process asyncio queues.

    ``publish`` is synchronous and never blocks or awaits, so it is safe to call
    from a structlog processor or a pipeline frame callback. Fan-out is a shallow
    copy of the subscriber set, so subscribing/unsubscribing during a publish is
    safe.
    """

    def __init__(self, *, maxsize: int = _DEFAULT_MAXSIZE) -> None:
        self._subscribers: set[Subscription] = set()
        self._maxsize = maxsize
        self._seq = 0

    @property
    def subscriber_count(self) -> int:
        return len(self._subscribers)

    def subscribe(self) -> Subscription:
        sub = Subscription(self, self._maxsize)
        self._subscribers.add(sub)
        return sub

    def unsubscribe(self, sub: Subscription) -> None:
        self._subscribers.discard(sub)

    def publish(
        self,
        type_: str | EventType,
        data: Mapping[str, Any] | None = None,
        *,
        session_id: str | None = None,
        ts: float | None = None,
    ) -> Envelope:
        """Fan an event out to every subscriber. Returns the envelope (useful in
        tests) even when there are no subscribers."""
        self._seq += 1
        envelope = Envelope(
            type=str(type_),
            ts=ts if ts is not None else time.time(),
            session_id=session_id,
            seq=self._seq,
            data=data or {},
        )
        for sub in tuple(self._subscribers):
            sub._offer(envelope)
        return envelope

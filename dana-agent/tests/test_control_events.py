"""EventBus fan-out, envelope shape, and the drop-oldest backpressure policy."""

from __future__ import annotations

from dana_agent.control.events import Envelope, EventBus, EventType


def test_envelope_shape() -> None:
    bus = EventBus()
    env = bus.publish(EventType.STATE_TRANSITION, {"from": "idle", "to": "active"}, session_id="s1")
    j = env.to_json()
    assert set(j) == {"type", "ts", "session_id", "seq", "data"}
    assert j["type"] == "state_transition"
    assert j["session_id"] == "s1"
    assert j["seq"] == 1
    assert j["data"] == {"from": "idle", "to": "active"}
    assert isinstance(j["ts"], float)


def test_seq_is_monotonic() -> None:
    bus = EventBus()
    seqs = [bus.publish("error", {}).seq for _ in range(5)]
    assert seqs == [1, 2, 3, 4, 5]


async def test_multi_subscriber_fanout() -> None:
    bus = EventBus()
    sub_a = bus.subscribe()
    sub_b = bus.subscribe()
    assert bus.subscriber_count == 2

    bus.publish("mic_state", {"muted": True})

    a = await sub_a.get()
    b = await sub_b.get()
    assert a.type == b.type == "mic_state"
    assert a.data == b.data == {"muted": True}


async def test_unsubscribe_stops_delivery() -> None:
    bus = EventBus()
    sub = bus.subscribe()
    bus.unsubscribe(sub)
    bus.publish("error", {"msg": "x"})
    assert bus.subscriber_count == 0
    assert sub._queue.empty()


async def test_context_manager_unsubscribes() -> None:
    bus = EventBus()
    async with bus.subscribe():
        assert bus.subscriber_count == 1
    assert bus.subscriber_count == 0


async def test_drop_oldest_on_overflow() -> None:
    bus = EventBus(maxsize=3)
    sub = bus.subscribe()
    for i in range(5):  # 5 into a size-3 queue → oldest two dropped
        bus.publish("latency_span", {"i": i})
    received = [await sub.get() for _ in range(3)]
    # The three most-recent survive; the two oldest were dropped.
    assert [e.data["i"] for e in received] == [2, 3, 4]
    assert sub.dropped == 2


def test_publish_without_subscribers_is_noop() -> None:
    bus = EventBus()
    env = bus.publish("error", {"msg": "nobody listening"})
    assert isinstance(env, Envelope)
    assert bus.subscriber_count == 0

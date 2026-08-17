"""Transcript/assistant PII is redacted from the durable log sinks.

docs/PLAN.md §7/§8 and this module's own contract say never persist PII to the
logs. The live control bus still receives the full text (it taps the shared
processor chain upstream of the renderers); only the rendered console/file
outputs redact it.
"""

from __future__ import annotations

import structlog

from dana_agent.observability.logging import (
    _forward_events,
    _redact_pii,
    _redacting_renderer,
    set_event_sink,
)


def test_redact_pii_strips_transcript_text() -> None:
    out = _redact_pii({"event": "transcript_final", "text": "buy me flowers"})
    assert out["text"] == "[redacted]"


def test_redact_pii_covers_all_pii_events() -> None:
    for event in ("brain_turn_start", "transcript_final", "assistant_turn"):
        out = _redact_pii({"event": event, "text": "secret words"})
        assert out["text"] == "[redacted]", event


def test_redact_pii_leaves_non_pii_events_untouched() -> None:
    record = {"event": "latency", "text": "not really pii here", "ms": 3}
    assert _redact_pii(record) is record  # unchanged, same object


def test_redact_pii_does_not_mutate_input() -> None:
    record = {"event": "assistant_turn", "text": "the raw reply"}
    _redact_pii(record)
    assert record["text"] == "the raw reply"  # original preserved for the bus


def test_redacting_renderer_hides_text_in_output() -> None:
    render = _redacting_renderer(structlog.processors.JSONRenderer())
    line = render(None, "info", {"event": "assistant_turn", "text": "the raw reply"})
    assert "the raw reply" not in line
    assert "[redacted]" in line


def test_bus_sink_still_sees_full_text() -> None:
    """The EventBus tap runs before the renderers, so the live dashboard keeps
    the full transcript even though the durable log redacts it."""
    seen: list[str] = []
    set_event_sink(lambda ed: seen.append(str(ed.get("text"))))
    try:
        record = {"event": "transcript_final", "text": "hello dana"}
        _forward_events(None, "info", record)
        assert seen == ["hello dana"]
    finally:
        set_event_sink(None)

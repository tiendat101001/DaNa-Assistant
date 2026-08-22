"""Bridge structlog events into the control :class:`EventBus`.

Several live signals already flow through structlog and shouldn't be re-plumbed
(this IS the project's neutral event sink — ``conversation/*`` only ever calls
``log.info(...)``; it never imports the control plane):

- the per-stage latency taps emitted by the conversation observer
  (``log.info("latency", stage=…, turn_id=…, ms=…)``) → ``latency_span`` events
  that feed the dashboard's waterfall;
- every gate decision/execution (``log.info("action_audit", …)``) →
  ``audit_entry`` events for the live audit view;
- the pipeline's turn taps (carry-over B) — the wake-gate FSM edges
  (``wake_gate_active`` / ``wake_gate_idle``) → ``state_transition``; the final
  user transcript (``transcript_final``); the assistant's reply
  (``assistant_turn``); and a pending destructive action
  (``confirmation_required``) → ``confirmation_request``.

Installing this sink keeps the pipeline decoupled: producers just log, and the
control plane opts in by tapping the structlog stream. Remove it (``set_event_sink(None)``)
to detach. The sink is intentionally tiny and total — it never raises.

Coverage note: the mock/offline pipeline never emits transcripts or wake edges,
so these mappings are unit-tested at the sink level here; end-to-end verification
of the live turn events needs recorded-audio / a live provider run.
"""

from __future__ import annotations

from collections.abc import Callable, MutableMapping
from typing import Any

from dana_agent.control.events import EventBus, EventType
from dana_agent.observability.logging import set_event_sink


def make_log_sink(bus: EventBus) -> Callable[[MutableMapping[str, Any]], None]:
    """Return a structlog event sink that republishes latency/audit events."""

    def sink(event_dict: MutableMapping[str, Any]) -> None:
        event = event_dict.get("event")
        if event == "latency":
            bus.publish(
                EventType.LATENCY_SPAN,
                {
                    "turn_id": event_dict.get("turn_id"),
                    "stage": event_dict.get("stage"),
                    "ms": event_dict.get("ms"),
                },
            )
        elif event == "action_audit":
            bus.publish(
                EventType.AUDIT_ENTRY,
                {
                    "at": event_dict.get("at"),
                    "tool": event_dict.get("tool"),
                    "reversibility": _as_str(event_dict.get("reversibility")),
                    "decision": _as_str(event_dict.get("decision")),
                    "executed": event_dict.get("executed"),
                    "ok": event_dict.get("ok"),
                },
            )
        # --- carry-over B: live pipeline turn events -------------------------
        elif event == "wake_gate_active":
            bus.publish(
                EventType.STATE_TRANSITION,
                # cause: "assemblyai_stt" (heard the phrase) | "ui" (Talk button
                # / typed turn — see WakeGateProcessor.force_active).
                {"from": "idle", "to": "active", "cause": event_dict.get("cause", "wake")},
            )
        elif event == "wake_gate_idle":
            bus.publish(
                EventType.STATE_TRANSITION,
                {"from": "active", "to": "idle", "cause": event_dict.get("cause", "")},
            )
        elif event == "transcript_final":
            bus.publish(
                EventType.TRANSCRIPT_FINAL,
                {"text": event_dict.get("text"), "role": event_dict.get("role", "user")},
                session_id=event_dict.get("session"),
            )
        elif event == "assistant_turn":
            bus.publish(
                EventType.ASSISTANT_TURN,
                {"text": event_dict.get("text"), "interrupted": False},
                session_id=event_dict.get("session"),
            )
        elif event == "confirmation_required":
            bus.publish(
                EventType.CONFIRMATION_REQUEST,
                {
                    "id": event_dict.get("id"),
                    "tool": event_dict.get("tool"),
                    "args_resolved": event_dict.get("args_resolved", {}),
                    "reversibility": _as_str(event_dict.get("reversibility", "")),
                    "preview": event_dict.get("preview", ""),
                },
                session_id=event_dict.get("session"),
            )

    return sink


def _as_str(value: Any) -> Any:
    """StrEnum values log as themselves; normalise to a plain str for the wire."""
    return value.value if hasattr(value, "value") else value


def install_log_bridge(bus: EventBus) -> None:
    set_event_sink(make_log_sink(bus))


def uninstall_log_bridge() -> None:
    set_event_sink(None)

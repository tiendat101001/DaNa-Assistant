"""AgentController wired to a real (offline, all-mock) stack.

Exercises everything that does NOT require starting the pipecat pipeline: mute
publishes a mic_state event, status reflects the config, confirmations publish +
resolve, and config hot-apply mutates the live config and emits config_changed.
"""

from __future__ import annotations

from dana_agent.app import build_stack
from dana_agent.brain.events import ConfirmationRequired
from dana_agent.control.controller import AgentController
from dana_agent.control.events import EventBus
from dana_agent.types import Reversibility
from tests.support import clean_config


async def _controller() -> tuple[AgentController, EventBus, object]:
    # Deterministic all-mock stack, independent of any dev .env.
    cfg = clean_config()
    stack = await build_stack(cfg)
    bus = EventBus()
    return AgentController(cfg, stack, bus), bus, stack


async def test_mute_toggles_gate_and_publishes_mic_state() -> None:
    ctrl, bus, stack = await _controller()
    sub = bus.subscribe()
    try:
        await ctrl.mute()
        assert ctrl.is_muted() is True
        env = await sub.get()
        assert env.type == "mic_state"
        assert env.data == {"muted": True}
        # The controller's mic_gate is the one handed to the pipeline.
        assert ctrl.mic_gate.muted is True
    finally:
        await stack.aclose()  # type: ignore[attr-defined]


async def test_status_reports_providers_and_idle_state() -> None:
    ctrl, _bus, stack = await _controller()
    try:
        s = ctrl.status()
        assert s.state == "idle"
        assert s.session_running is False
        assert s.mode == "cascade"
        assert s.providers["stt"] == "mock"
    finally:
        await stack.aclose()  # type: ignore[attr-defined]


async def test_register_and_resolve_confirmation() -> None:
    ctrl, bus, stack = await _controller()
    sub = bus.subscribe()
    try:
        event = ConfirmationRequired(
            id="c1",
            tool="delete_file",
            arguments={"path": "x"},
            reversibility=Reversibility.DESTRUCTIVE,
        )
        ctrl.register_confirmation(event)
        assert [c.id for c in ctrl.pending_confirmations()] == ["c1"]
        published = await sub.get()
        assert published.type == "confirmation_request"
        assert published.data["tool"] == "delete_file"

        # No live session/brain → resolve pops the pending record and returns True.
        assert await ctrl.resolve_confirmation("c1", approved=False) is True
        assert ctrl.pending_confirmations() == []
        assert await ctrl.resolve_confirmation("missing", approved=True) is False
    finally:
        await stack.aclose()  # type: ignore[attr-defined]


async def test_apply_config_hot_applies_and_publishes() -> None:
    ctrl, bus, stack = await _controller()
    sub = bus.subscribe()
    try:
        new = clean_config(
            log_level="DEBUG",
            conversation={"stt": {"provider": "deepgram"}},
        )
        changes = ctrl.apply_config(new)
        statuses = {c.path: c.status.value for c in changes}
        assert statuses["log_level"] == "applied"
        assert statuses["conversation.stt.provider"] == "restart_required"
        # Applied fields mutate the live config; restart_required do not.
        assert ctrl.config().log_level == "DEBUG"
        assert ctrl.config().conversation.stt.provider == "mock"
        env = await sub.get()
        assert env.type == "config_changed"
    finally:
        await stack.aclose()  # type: ignore[attr-defined]


async def test_apply_config_gate_policy_takes_live_effect() -> None:
    # gate_policy is labelled "applied" only because it re-binds the LIVE gate —
    # verify the running ActionGate actually sees the new policy (no restart).
    ctrl, _bus, stack = await _controller()
    try:
        assert stack.gate.policy == "standard"  # type: ignore[attr-defined]
        changes = ctrl.apply_config(clean_config(actions={"gate_policy": "conservative"}))
        statuses = {c.path: c.status.value for c in changes}
        assert statuses["actions.gate_policy"] == "applied"
        assert ctrl.config().actions.gate_policy == "conservative"
        # The live gate — the one the running pipeline uses — is re-bound in place.
        assert stack.gate.policy == "conservative"  # type: ignore[attr-defined]
    finally:
        await stack.aclose()  # type: ignore[attr-defined]


async def test_audit_entries_reads_from_gate() -> None:
    ctrl, _bus, stack = await _controller()
    try:
        # A fresh stack has an empty audit trail; the accessor returns a list.
        assert ctrl.audit_entries() == []
    finally:
        await stack.aclose()  # type: ignore[attr-defined]

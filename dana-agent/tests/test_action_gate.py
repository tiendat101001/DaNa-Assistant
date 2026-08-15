"""Action gate: reversibility -> decision per policy, plus the audit trail.
This is where safety lives, so it is tested hardest (safe-computer-use skill)."""

from __future__ import annotations

import pytest

from dana_agent.actions.gate import ActionGate, GateDecision
from dana_agent.config import ActionsConfig
from dana_agent.providers.registry import build_actions
from dana_agent.types import Action, Reversibility


def make_gate(policy: str = "standard") -> ActionGate:
    return ActionGate(build_actions(ActionsConfig()), policy=policy)


def test_read_only_is_allowed() -> None:
    decision, rev = make_gate().decide(Action("read_page", {}))
    assert decision is GateDecision.ALLOW
    assert rev is Reversibility.READ_ONLY


def test_destructive_requires_confirmation_under_standard() -> None:
    decision, rev = make_gate("standard").decide(Action("delete_file", {}))
    assert decision is GateDecision.CONFIRM
    assert rev is Reversibility.DESTRUCTIVE


def test_reversible_allowed_under_standard_but_confirmed_under_conservative() -> None:
    action = Action("fill_form", {})
    assert make_gate("standard").decide(action)[0] is GateDecision.ALLOW
    assert make_gate("conservative").decide(action)[0] is GateDecision.CONFIRM


def test_wide_still_confirms_destructive() -> None:
    assert make_gate("wide").decide(Action("send_email", {}))[0] is GateDecision.CONFIRM


def test_unknown_policy_rejected() -> None:
    with pytest.raises(ValueError, match="unknown gate policy"):
        make_gate("yolo")


def test_decide_records_a_non_executing_audit_entry() -> None:
    gate = make_gate()
    decision, _ = gate.decide(Action("read_page", {"url": "x"}))
    assert decision is GateDecision.ALLOW
    entries = gate.audit.entries
    assert len(entries) == 1
    assert entries[0].tool == "read_page"
    assert entries[0].executed is False


async def test_execute_records_execution_entry() -> None:
    gate = make_gate()
    action = Action("read_page", {})
    _, rev = gate.decide(action)
    result = await gate.execute(action, rev)
    assert result.ok is True
    entries = gate.audit.entries
    assert len(entries) == 2
    assert entries[1].executed is True
    assert entries[1].ok is True

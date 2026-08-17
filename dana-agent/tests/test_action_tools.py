"""Parameterized gated tools end to end: validated arguments flow through
gate.decide -> (destructive) interrupt -> gate.execute, driven by a scripted
model and a fake page (no real browser). Plus the raw gate-level allowlist
checks. This is the safety-critical layer, so it is tested hardest."""

from __future__ import annotations

from dana_agent.actions.gate import ActionGate, GateDecision
from dana_agent.brain import (
    BROWSER_TOOLS,
    Brain,
    ConfirmationRequired,
    ToolCompleted,
    TurnComplete,
)
from dana_agent.config import ActionsConfig
from dana_agent.providers.actions.browser import BrowserActions
from dana_agent.types import Action, Reversibility
from tests.conftest import collect
from tests.fakes.chat import ScriptedChatModel
from tests.fixtures.browser import FakePage, browser_with_fake_page

SYSTEM = "You are DaNa."


def make_gate(domains: list[str], page: FakePage | None = None) -> tuple[ActionGate, FakePage]:
    provider = BrowserActions(ActionsConfig(provider="browser", allowed_domains=domains))
    # Start on an allowlisted page so content/interaction tools have a valid
    # current-page precondition (the redirect guard refuses off-allowlist pages).
    fake = browser_with_fake_page(provider, page or FakePage(url="https://example.com"))
    return ActionGate(provider, policy="standard"), fake


def make_brain(gate: ActionGate, model: ScriptedChatModel) -> Brain:
    return Brain(model, gate, system_prompt=SYSTEM, tools=list(BROWSER_TOOLS))


# --- gate-level validation (no model) ------------------------------------


def test_gate_allows_allowlisted_navigate() -> None:
    gate, _ = make_gate(["example.com"])
    decision, rev = gate.decide(Action("browser_navigate", {"url": "https://example.com"}))
    assert decision is GateDecision.ALLOW and rev is Reversibility.READ_ONLY


def test_gate_refuses_off_allowlist_navigate() -> None:
    gate, _ = make_gate(["example.com"])
    decision, _ = gate.decide(Action("browser_navigate", {"url": "https://evil.com"}))
    assert decision is GateDecision.REFUSE


def test_gate_confirms_committing_click() -> None:
    gate, _ = make_gate(["example.com"])
    decision, rev = gate.decide(Action("browser_click", {"text": "Submit payment"}))
    assert decision is GateDecision.CONFIRM and rev is Reversibility.DESTRUCTIVE


# --- through the brain: parameterized args flow to the provider ----------


async def test_read_only_tool_runs_with_validated_args() -> None:
    gate, page = make_gate(["example.com"])
    model = ScriptedChatModel(
        turns=[
            {"tool": "browser_navigate", "args": {"url": "https://example.com/hi"}},
            {"text": "done"},
        ]
    )
    events = await collect(make_brain(gate, model).run_turn("go", thread_id="t1"))
    completed = next(e for e in events if isinstance(e, ToolCompleted))
    assert completed.ok is True and completed.refused is False
    assert isinstance(events[-1], TurnComplete)
    # the validated url argument reached the provider/page
    assert page.goto_urls == ["https://example.com/hi"]


async def test_destructive_tool_pauses_for_confirmation() -> None:
    gate, page = make_gate(["example.com"])
    model = ScriptedChatModel(
        turns=[{"tool": "browser_click", "args": {"text": "Place order"}}, {"text": "done"}]
    )
    events = await collect(make_brain(gate, model).run_turn("buy", thread_id="t1"))
    assert isinstance(events[-1], ConfirmationRequired)
    assert events[-1].reversibility is Reversibility.DESTRUCTIVE
    assert events[-1].arguments == {"text": "Place order"}
    # nothing was clicked while paused
    assert page.clicks == []


async def test_off_allowlist_tool_is_refused() -> None:
    gate, page = make_gate(["example.com"])
    model = ScriptedChatModel(
        turns=[
            {"tool": "browser_navigate", "args": {"url": "https://evil.com"}},
            {"text": "done"},
        ]
    )
    events = await collect(make_brain(gate, model).run_turn("go", thread_id="t1"))
    completed = next(e for e in events if isinstance(e, ToolCompleted))
    assert completed.refused is True and completed.ok is False
    assert page.goto_urls == []


async def test_confirmed_destructive_click_then_executes() -> None:
    gate, page = make_gate(["example.com"])
    model = ScriptedChatModel(
        turns=[{"tool": "browser_click", "args": {"text": "Submit"}}, {"text": "ok"}]
    )
    brain = make_brain(gate, model)
    await collect(brain.run_turn("submit", thread_id="t1"))
    resumed = await collect(brain.resume_turn(thread_id="t1", approved=True))
    completed = next(e for e in resumed if isinstance(e, ToolCompleted))
    assert completed.ok is True and completed.refused is False
    assert page.clicks == [{"text": "Submit"}]

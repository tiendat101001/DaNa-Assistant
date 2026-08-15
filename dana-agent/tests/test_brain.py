"""Brain (LangGraph) agent loop: token streaming, tool routing, the confirmation
interrupt, memory, and the step limit — driven by a scripted fake chat model and
mock actions (no real model, no network)."""

from __future__ import annotations

from dana_agent.actions.gate import ActionGate
from dana_agent.brain import (
    Brain,
    ConfirmationRequired,
    StepLimitReached,
    TokenDelta,
    ToolCompleted,
    ToolInvoked,
    ToolSpec,
    TurnComplete,
    build_chat_model,
)
from dana_agent.config import ActionsConfig, LLMConfig
from dana_agent.providers.registry import build_actions
from tests.conftest import collect
from tests.fakes.chat import ScriptedChatModel

SYSTEM = "You are DaNa."


def make_gate(policy: str = "standard") -> ActionGate:
    return ActionGate(build_actions(ActionsConfig()), policy=policy)


async def test_plain_turn_streams_tokens_then_completes() -> None:
    brain = Brain(build_chat_model(LLMConfig()), make_gate(), system_prompt=SYSTEM)
    events = await collect(brain.run_turn("hello there", thread_id="t1"))
    assert any(isinstance(e, TokenDelta) for e in events)
    assert isinstance(events[-1], TurnComplete)
    assert "hello there" in events[-1].text


async def test_read_only_tool_runs_then_completes() -> None:
    model = ScriptedChatModel(turns=[{"tool": "read_page"}, {"text": "all done here"}])
    brain = Brain(
        model, make_gate(), system_prompt=SYSTEM, tools=[ToolSpec("read_page", "read the page")]
    )
    events = await collect(brain.run_turn("read it", thread_id="t1"))
    assert any(isinstance(e, ToolInvoked) for e in events)
    completed = next(e for e in events if isinstance(e, ToolCompleted))
    assert completed.ok is True and completed.refused is False
    assert isinstance(events[-1], TurnComplete)


async def test_destructive_tool_pauses_for_confirmation() -> None:
    model = ScriptedChatModel(turns=[{"tool": "delete_file"}, {"text": "done"}])
    brain = Brain(
        model, make_gate(), system_prompt=SYSTEM, tools=[ToolSpec("delete_file", "delete a file")]
    )
    events = await collect(brain.run_turn("delete it", thread_id="t1"))
    assert isinstance(events[-1], ConfirmationRequired)
    assert events[-1].reversibility.value == "destructive"
    assert events[-1].tool == "delete_file"


async def test_confirmation_approved_executes_and_finishes() -> None:
    model = ScriptedChatModel(turns=[{"tool": "delete_file"}, {"text": "deleted"}])
    gate = make_gate()
    brain = Brain(
        model, gate, system_prompt=SYSTEM, tools=[ToolSpec("delete_file", "delete a file")]
    )
    await collect(brain.run_turn("delete it", thread_id="t1"))
    resumed = await collect(brain.resume_turn(thread_id="t1", approved=True))
    completed = next(e for e in resumed if isinstance(e, ToolCompleted))
    assert completed.ok is True and completed.refused is False
    assert isinstance(resumed[-1], TurnComplete)
    assert any(entry.executed for entry in gate.audit.entries)


async def test_confirmation_denied_refuses_and_finishes() -> None:
    model = ScriptedChatModel(turns=[{"tool": "delete_file"}, {"text": "ok"}])
    brain = Brain(
        model, make_gate(), system_prompt=SYSTEM, tools=[ToolSpec("delete_file", "delete a file")]
    )
    await collect(brain.run_turn("delete it", thread_id="t1"))
    resumed = await collect(brain.resume_turn(thread_id="t1", approved=False))
    completed = next(e for e in resumed if isinstance(e, ToolCompleted))
    assert completed.refused is True and completed.ok is False
    assert isinstance(resumed[-1], TurnComplete)


async def test_memory_persists_across_turns() -> None:
    brain = Brain(build_chat_model(LLMConfig()), make_gate(), system_prompt=SYSTEM)
    await collect(brain.run_turn("first message", thread_id="t1"))
    await collect(brain.run_turn("second message", thread_id="t1"))
    assert await brain.user_messages("t1") == ["first message", "second message"]


async def test_separate_threads_do_not_share_memory() -> None:
    brain = Brain(build_chat_model(LLMConfig()), make_gate(), system_prompt=SYSTEM)
    await collect(brain.run_turn("hello from A", thread_id="a"))
    await collect(brain.run_turn("hello from B", thread_id="b"))
    assert await brain.user_messages("a") == ["hello from A"]
    assert await brain.user_messages("b") == ["hello from B"]


async def test_step_limit_halts_a_runaway_agent() -> None:
    # A model that only ever calls the tool -> the loop never terminates.
    model = ScriptedChatModel(turns=[{"tool": "read_page"}])
    brain = Brain(
        model,
        make_gate(),
        system_prompt=SYSTEM,
        tools=[ToolSpec("read_page", "read the page")],
        max_steps=4,
    )
    events = await collect(brain.run_turn("go", thread_id="t1"))
    assert isinstance(events[-1], StepLimitReached)

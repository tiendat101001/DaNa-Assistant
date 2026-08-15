"""Brain agent loop: routing, tool execution, confirmation interrupts, memory,
and the step limit — driven by the mock LLM and mock actions (no real model)."""

from __future__ import annotations

from collections.abc import AsyncIterator

from dana_agent.actions.gate import ActionGate
from dana_agent.brain import (
    Brain,
    ConfirmationRequired,
    InMemoryCheckpointer,
    StepLimitReached,
    TokenDelta,
    ToolCompleted,
    ToolInvoked,
    TurnComplete,
)
from dana_agent.config import ActionsConfig, LLMConfig
from dana_agent.providers.base import Capabilities, LLMProvider
from dana_agent.providers.registry import build_actions, build_llm
from dana_agent.types import LLMChunk, Message, Role, Tool, ToolCall
from tests.conftest import collect

SYSTEM = "You are DaNa."


class AlwaysToolLLM(LLMProvider):
    """Pathological model that never stops calling a tool — used to prove the
    step limit halts a runaway agent."""

    async def stream(
        self, messages: list[Message], tools: list[Tool] | None = None
    ) -> AsyncIterator[LLMChunk]:
        yield LLMChunk(tool_calls=(ToolCall(id="c", name="read_page", arguments={}),))
        yield LLMChunk(done=True)

    @property
    def capabilities(self) -> Capabilities:
        return Capabilities(streaming=True, languages=frozenset({"en", "vi"}), tool_calling=True)


def make_brain(
    *,
    policy: str = "standard",
    tools: list[Tool] | None = None,
    checkpointer: InMemoryCheckpointer | None = None,
    max_steps: int = 8,
) -> Brain:
    gate = ActionGate(build_actions(ActionsConfig()), policy=policy)
    return Brain(
        build_llm(LLMConfig()),
        gate,
        system_prompt=SYSTEM,
        tools=tools or [],
        max_steps=max_steps,
        checkpointer=checkpointer or InMemoryCheckpointer(),
    )


async def test_plain_turn_streams_tokens_then_completes() -> None:
    brain = make_brain()
    events = await collect(brain.run_turn("hello there", thread_id="t1"))
    assert any(isinstance(e, TokenDelta) for e in events)
    assert isinstance(events[-1], TurnComplete)
    assert "hello there" in events[-1].text


async def test_read_only_tool_runs_then_turn_completes() -> None:
    tools = [Tool(name="read_page", description="read the current page")]
    brain = make_brain(tools=tools)
    events = await collect(brain.run_turn("please read_page now", thread_id="t1"))
    kinds = [type(e).__name__ for e in events]
    assert "ToolInvoked" in kinds
    assert "ToolCompleted" in kinds
    assert isinstance(events[-1], TurnComplete)
    completed = next(e for e in events if isinstance(e, ToolCompleted))
    assert completed.ok is True and completed.refused is False


async def test_destructive_tool_pauses_for_confirmation() -> None:
    tools = [Tool(name="delete_file", description="delete a file")]
    brain = make_brain(tools=tools)
    events = await collect(brain.run_turn("please delete_file x", thread_id="t1"))
    # No ToolInvoked/Completed yet — the turn paused awaiting approval.
    assert not any(isinstance(e, (ToolInvoked, ToolCompleted)) for e in events)
    assert isinstance(events[-1], ConfirmationRequired)


async def test_confirmation_approved_executes_and_finishes() -> None:
    tools = [Tool(name="delete_file", description="delete a file")]
    cp = InMemoryCheckpointer()
    brain = make_brain(tools=tools, checkpointer=cp)
    pause = await collect(brain.run_turn("please delete_file x", thread_id="t1"))
    assert isinstance(pause[-1], ConfirmationRequired)

    resumed = await collect(brain.resume_turn(thread_id="t1", approved=True))
    completed = next(e for e in resumed if isinstance(e, ToolCompleted))
    assert completed.ok is True and completed.refused is False
    assert isinstance(resumed[-1], TurnComplete)


async def test_confirmation_denied_refuses_and_finishes() -> None:
    tools = [Tool(name="delete_file", description="delete a file")]
    cp = InMemoryCheckpointer()
    brain = make_brain(tools=tools, checkpointer=cp)
    await collect(brain.run_turn("please delete_file x", thread_id="t1"))

    resumed = await collect(brain.resume_turn(thread_id="t1", approved=False))
    completed = next(e for e in resumed if isinstance(e, ToolCompleted))
    assert completed.refused is True and completed.ok is False
    assert isinstance(resumed[-1], TurnComplete)


async def test_memory_persists_across_turns() -> None:
    cp = InMemoryCheckpointer()
    brain = make_brain(checkpointer=cp)
    await collect(brain.run_turn("first message", thread_id="t1"))
    await collect(brain.run_turn("second message", thread_id="t1"))
    state = cp.load("t1")
    assert state is not None
    user_msgs = [m.content for m in state.messages if m.role is Role.USER]
    assert user_msgs == ["first message", "second message"]


async def test_separate_threads_do_not_share_memory() -> None:
    cp = InMemoryCheckpointer()
    brain = make_brain(checkpointer=cp)
    await collect(brain.run_turn("hello from A", thread_id="a"))
    await collect(brain.run_turn("hello from B", thread_id="b"))
    state_a = cp.load("a")
    state_b = cp.load("b")
    assert state_a is not None and state_b is not None
    assert [m.content for m in state_a.messages if m.role is Role.USER] == ["hello from A"]
    assert [m.content for m in state_b.messages if m.role is Role.USER] == ["hello from B"]


async def test_step_limit_halts_a_runaway_agent() -> None:
    gate = ActionGate(build_actions(ActionsConfig()), policy="standard")
    brain = Brain(
        AlwaysToolLLM(),
        gate,
        system_prompt=SYSTEM,
        tools=[Tool(name="read_page", description="read")],
        max_steps=3,
    )
    events = await collect(brain.run_turn("go", thread_id="t1"))
    assert isinstance(events[-1], StepLimitReached)
    assert events[-1].steps == 3
    # It ran the read-only tool once per step, up to the limit.
    assert sum(isinstance(e, ToolCompleted) for e in events) == 3

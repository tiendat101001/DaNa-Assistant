"""History helpers (trim/summarize) and their end-to-end effect on the brain."""

from __future__ import annotations

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from dana_agent.actions.gate import ActionGate
from dana_agent.brain import Brain, HistoryPolicy, build_chat_model
from dana_agent.brain.history import (
    build_system,
    messages_to_prune,
    select_conversation,
)
from dana_agent.config import ActionsConfig, LLMConfig
from dana_agent.providers.registry import build_actions
from tests.conftest import collect

SYSTEM = "You are DaNa."


def _convo(n: int) -> list[BaseMessage]:
    msgs: list[BaseMessage] = []
    for i in range(n):
        msgs.append(HumanMessage(content=f"h{i}"))
        msgs.append(AIMessage(content=f"a{i}"))
    return msgs


def test_trim_keeps_only_last_n() -> None:
    messages = _convo(5)  # 10 messages
    kept = select_conversation(messages, HistoryPolicy(strategy="trim", keep_last=4))
    assert len(kept) == 4
    assert [m.content for m in kept] == ["h3", "a3", "h4", "a4"]


def test_none_and_summarize_do_not_trim_at_read() -> None:
    messages = _convo(5)
    assert select_conversation(messages, HistoryPolicy(strategy="none")) is messages
    assert select_conversation(messages, HistoryPolicy(strategy="summarize")) is messages


def test_build_system_injects_summary() -> None:
    assert build_system(SYSTEM, "").content == SYSTEM
    withsum = build_system(SYSTEM, "prior facts").content
    assert SYSTEM in withsum and "prior facts" in withsum


def test_prune_keeps_last_and_avoids_orphan_tool_result() -> None:
    # ...older..., AIMessage(tool call), ToolMessage(result), HumanMessage
    messages = [
        HumanMessage(content="old1"),
        AIMessage(content="old2"),
        AIMessage(content="calls tool"),
        ToolMessage(content="tool result", tool_call_id="1"),
        HumanMessage(content="recent"),
    ]
    old = messages_to_prune(messages, keep_last=2)
    # keep_last=2 would cut at index 3 (a ToolMessage) -> boundary shifts past it
    assert [m.content for m in old] == ["old1", "old2", "calls tool", "tool result"]


def make_brain(history: HistoryPolicy) -> Brain:
    gate = ActionGate(build_actions(ActionsConfig()), policy="standard")
    return Brain(build_chat_model(LLMConfig()), gate, system_prompt=SYSTEM, history=history)


async def test_summarize_prunes_and_records_summary() -> None:
    brain = make_brain(HistoryPolicy(strategy="summarize", trigger_messages=4, keep_last=2))
    for i in range(4):
        await collect(brain.run_turn(f"message {i}", thread_id="t1"))
    snapshot = await brain._graph.aget_state({"configurable": {"thread_id": "t1"}})
    # Older turns were pruned (fewer than the 4 user messages sent) ...
    users = [m.content for m in snapshot.values["messages"] if isinstance(m, HumanMessage)]
    assert len(users) < 4
    assert "message 3" in users  # most recent kept
    # ... and a running summary was recorded.
    assert snapshot.values.get("summary")


async def test_trim_leaves_stored_history_intact() -> None:
    brain = make_brain(HistoryPolicy(strategy="trim", keep_last=2))
    for i in range(4):
        await collect(brain.run_turn(f"message {i}", thread_id="t1"))
    # trim only bounds what the model sees; stored history keeps every user turn.
    assert await brain.user_messages("t1") == [f"message {i}" for i in range(4)]

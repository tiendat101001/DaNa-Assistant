"""The LangGraph state graph: a prebuilt agent/tools loop with history control.

    START -> [summarize] -> agent -> (tools? -> tools -> agent | END)

Uses LangGraph's own nodes and routing (``ToolNode`` + ``tools_condition``)
rather than a hand-rolled loop. The agent node prepends the system prompt (plus
any running summary), applies the history policy to bound context, and calls the
chat model; ``tools_condition`` routes to the tool node when the model emits tool
calls, then back to the agent. A ``summarize`` node is added only for the
summarize strategy. A checkpointer gives per-thread memory and makes a
confirmation-paused turn resumable.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage, RemoveMessage
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from dana_agent.brain.history import (
    HistoryPolicy,
    build_system,
    messages_to_prune,
    select_conversation,
    summary_prompt,
)

AGENT = "agent"
TOOLS = "tools"
SUMMARIZE = "summarize"


class BrainState(MessagesState):  # type: ignore[misc]  # MessagesState is Any under strict
    summary: str


def _to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            p if isinstance(p, str) else str(p.get("text", ""))
            for p in content
            if isinstance(p, str | dict)
        )
    return str(content)


def build_graph(
    model: Any,  # a chat model, or a bind_tools() runnable wrapping it
    tools: list[Any],
    *,
    system_prompt: str,
    checkpointer: Any,
    policy: HistoryPolicy,
    summarizer: Any,
) -> Any:
    async def agent(state: BrainState) -> dict[str, list[BaseMessage]]:
        conversation = select_conversation(state["messages"], policy)
        system = build_system(system_prompt, state.get("summary", ""))
        response = await model.ainvoke([system, *conversation])
        return {"messages": [response]}

    async def summarize(state: BrainState) -> dict[str, Any]:
        messages = state["messages"]
        if len(messages) <= policy.trigger_messages:
            return {}
        old = messages_to_prune(messages, policy.keep_last)
        if not old:
            return {}
        prompt = summary_prompt(state.get("summary", ""), old)
        result = await summarizer.ainvoke([HumanMessage(content=prompt)])
        return {
            "summary": _to_text(result.content),
            "messages": [RemoveMessage(id=m.id) for m in old if m.id],
        }

    builder: StateGraph = StateGraph(BrainState)
    builder.add_node(AGENT, agent)

    if policy.strategy == "summarize":
        builder.add_node(SUMMARIZE, summarize)
        builder.add_edge(START, SUMMARIZE)
        builder.add_edge(SUMMARIZE, AGENT)
    else:
        builder.add_edge(START, AGENT)

    if tools:
        builder.add_node(TOOLS, ToolNode(tools))
        builder.add_conditional_edges(AGENT, tools_condition)
        builder.add_edge(TOOLS, AGENT)
    # No tools: agent is terminal (tools_condition would route only to END).

    return builder.compile(checkpointer=checkpointer)

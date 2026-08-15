"""The LangGraph state graph: a prebuilt agent/tools loop.

    START -> agent -> (tools? -> tools -> agent | END)

Uses LangGraph's own nodes and routing (``ToolNode`` + ``tools_condition``)
rather than a hand-rolled loop. The agent node prepends the system prompt and
calls the chat model; ``tools_condition`` routes to the tool node when the model
emits tool calls, then back to the agent. A checkpointer gives per-thread memory
and makes a confirmation-paused turn resumable.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import BaseMessage, SystemMessage
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

AGENT = "agent"
TOOLS = "tools"


def build_graph(
    model: Any,  # a chat model, or a bind_tools() runnable wrapping it
    tools: list[Any],
    *,
    system_prompt: str,
    checkpointer: Any,
) -> Any:
    async def agent(state: MessagesState) -> dict[str, list[BaseMessage]]:
        messages = state["messages"]
        if not any(isinstance(m, SystemMessage) for m in messages):
            messages = [SystemMessage(content=system_prompt), *messages]
        response = await model.ainvoke(messages)
        return {"messages": [response]}

    builder: StateGraph = StateGraph(MessagesState)
    builder.add_node(AGENT, agent)
    builder.add_edge(START, AGENT)

    if tools:
        builder.add_node(TOOLS, ToolNode(tools))
        builder.add_conditional_edges(AGENT, tools_condition)
        builder.add_edge(TOOLS, AGENT)
    # No tools: agent is terminal (tools_condition would route only to END).

    return builder.compile(checkpointer=checkpointer)

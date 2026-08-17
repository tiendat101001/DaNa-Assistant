"""The brain: a LangGraph agent that streams DaNa ``BrainEvent``s.

Wraps the compiled agent/tools graph and translates LangGraph's event stream
into DaNa's neutral event API so the voice pipeline never sees LangChain types:

- ``on_chat_model_stream`` (agent node) -> ``TokenDelta`` (feeds TTS immediately)
- ``on_tool_start`` / ``on_tool_end``    -> ``ToolInvoked`` / ``ToolCompleted``
- a tool's ``interrupt`` (surfaced as a pending interrupt after the stream)
                                          -> ``ConfirmationRequired`` (turn paused)
- otherwise                              -> ``TurnComplete``
- recursion limit hit                    -> ``StepLimitReached``

Resume a paused turn with ``resume_turn(thread_id, approved=...)``.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import BaseTool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphRecursionError
from langgraph.types import Command

from dana_agent.actions.gate import ActionGate
from dana_agent.brain.events import (
    BrainEvent,
    ConfirmationRequired,
    StepLimitReached,
    TokenDelta,
    ToolCompleted,
    ToolInvoked,
    TurnComplete,
)
from dana_agent.brain.graph import AGENT, build_graph
from dana_agent.brain.history import HistoryPolicy
from dana_agent.brain.tools import ToolSpec, build_gated_tool
from dana_agent.observability import get_logger
from dana_agent.types import Action, Reversibility

log = get_logger(__name__)


def _to_text(content: Any) -> str:
    """Flatten message content (str, or a list of parts as some providers emit)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for piece in content:
            if isinstance(piece, str):
                parts.append(piece)
            elif isinstance(piece, dict) and isinstance(piece.get("text"), str):
                parts.append(piece["text"])
        return "".join(parts)
    return str(content)


class Brain:
    def __init__(
        self,
        model: BaseChatModel,
        gate: ActionGate,
        *,
        system_prompt: str,
        tools: Sequence[ToolSpec | BaseTool] | None = None,
        max_steps: int = 8,
        checkpointer: Any = None,
        history: HistoryPolicy | None = None,
    ) -> None:
        self._gate = gate
        self._max_steps = max_steps
        # Accept either ToolSpecs (wrapped by the gate here) or already-gated
        # BaseTools (built by build_action_tools at the composition root).
        self._tools: list[BaseTool] = [
            t if isinstance(t, BaseTool) else build_gated_tool(t, gate) for t in (tools or [])
        ]
        bound = model.bind_tools(self._tools) if self._tools else model
        self._graph = build_graph(
            bound,
            self._tools,
            system_prompt=system_prompt,
            checkpointer=checkpointer or InMemorySaver(),
            policy=history or HistoryPolicy(),
            summarizer=model,  # unbound model: summaries don't call tools
        )

    def set_max_steps(self, max_steps: int) -> None:
        """Change the per-turn step budget on the live brain (hot-applied from
        PUT /config). ``_stream`` reads ``self._max_steps`` at the start of each
        turn, so the new limit applies to the next turn — no restart required."""
        if max_steps <= 0:
            raise ValueError(f"max_steps must be positive, got {max_steps}")
        self._max_steps = max_steps

    async def run_turn(self, user_text: str, *, thread_id: str) -> AsyncIterator[BrainEvent]:
        """Start a turn for ``user_text`` in conversation ``thread_id``."""
        async for event in self._stream({"messages": [HumanMessage(content=user_text)]}, thread_id):
            yield event

    async def resume_turn(self, *, thread_id: str, approved: bool) -> AsyncIterator[BrainEvent]:
        """Resume a turn paused on a confirmation with the human's decision."""
        async for event in self._stream(Command(resume=approved), thread_id):
            yield event

    async def user_messages(self, thread_id: str) -> list[str]:
        """The user turns recorded for a conversation (checkpointer memory)."""
        snapshot = await self._graph.aget_state({"configurable": {"thread_id": thread_id}})
        messages = snapshot.values.get("messages", []) if snapshot.values else []
        return [_to_text(m.content) for m in messages if isinstance(m, HumanMessage)]

    async def _stream(self, graph_input: Any, thread_id: str) -> AsyncIterator[BrainEvent]:
        config = {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": self._max_steps,
        }
        try:
            async for ev in self._graph.astream_events(graph_input, config=config, version="v2"):
                translated = self._translate(ev)
                if translated is not None:
                    yield translated
        except GraphRecursionError:
            log.warning("step_limit_reached", steps=self._max_steps)
            yield StepLimitReached(self._max_steps)
            return

        # Terminal event: paused for confirmation, or turn complete.
        snapshot = await self._graph.aget_state(config)
        interrupts = getattr(snapshot, "interrupts", ())
        if interrupts:
            payload = interrupts[0].value
            yield ConfirmationRequired(
                id=str(interrupts[0].id),
                tool=payload.get("tool", ""),
                arguments=payload.get("arguments", {}),
                reversibility=Reversibility(payload.get("reversibility", "reversible")),
            )
            return
        yield TurnComplete(self._final_text(snapshot))

    def _translate(self, ev: Any) -> BrainEvent | None:
        kind = ev["event"]
        if kind == "on_chat_model_stream":
            if ev.get("metadata", {}).get("langgraph_node") != AGENT:
                return None
            text = _to_text(ev["data"]["chunk"].content)
            return TokenDelta(text) if text else None
        if kind == "on_tool_start":
            name = ev.get("name", "")
            args = ev["data"].get("input") or {}
            reversibility = self._gate.classify(Action(tool=name, arguments=args))
            return ToolInvoked(
                call_id=str(ev.get("run_id", "")),
                tool=name,
                arguments=args,
                reversibility=reversibility,
            )
        if kind == "on_tool_end":
            output = ev["data"].get("output")
            content = _to_text(getattr(output, "content", output))
            refused = content.startswith("refused")
            return ToolCompleted(
                call_id=str(ev.get("run_id", "")),
                tool=ev.get("name", ""),
                ok=not refused,
                output=content,
                refused=refused,
            )
        return None

    @staticmethod
    def _final_text(snapshot: Any) -> str:
        messages = snapshot.values.get("messages", []) if snapshot.values else []
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and not msg.tool_calls:
                return _to_text(msg.content).strip()
        return ""

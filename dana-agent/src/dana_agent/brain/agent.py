"""The agent/tools loop.

    ┌────────────┐
    │   agent    │  stream the model: text and/or tool calls
    └─────┬──────┘
   tools? │ else
    ┌─────▼──────┐
    │   tools    │  gate + execute (or pause for confirmation) ──► back to agent
    └────────────┘

Why not LangGraph directly: LangGraph's prebuilt nodes assume a LangChain chat
model, but DaNa's model is an ``LLMProvider`` streaming abstraction (a Pipecat
service in Phase 3). This loop follows the same idiomatic shape — typed state, an
agent/tools conditional edge, checkpointing, human-in-the-loop interrupts, and a
step limit — over DaNa's own abstractions. Revisit backing it with LangGraph once
the Pipecat integration is settled (tracked in the backlog).
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator

from dana_agent.actions.gate import ActionGate, GateDecision
from dana_agent.brain.events import (
    BrainEvent,
    ConfirmationRequired,
    StepLimitReached,
    TokenDelta,
    ToolCompleted,
    ToolInvoked,
    TurnComplete,
)
from dana_agent.brain.state import (
    AgentState,
    Checkpointer,
    InMemoryCheckpointer,
    PendingConfirmation,
)
from dana_agent.observability import get_logger
from dana_agent.providers.base import LLMProvider
from dana_agent.types import Action, ActionResult, Message, Role, Tool, ToolCall

log = get_logger(__name__)


class Brain:
    def __init__(
        self,
        llm: LLMProvider,
        gate: ActionGate,
        *,
        system_prompt: str,
        tools: list[Tool] | None = None,
        max_steps: int = 8,
        checkpointer: Checkpointer | None = None,
    ) -> None:
        self._llm = llm
        self._gate = gate
        self._system_prompt = system_prompt
        self._tools = tools or []
        self._max_steps = max_steps
        self._checkpointer = checkpointer or InMemoryCheckpointer()

    # -- public API --------------------------------------------------------

    async def run_turn(self, user_text: str, *, thread_id: str) -> AsyncIterator[BrainEvent]:
        """Start a turn for ``user_text`` in conversation ``thread_id``."""
        state = self._checkpointer.load(thread_id) or self._new_state()
        state.messages.append(Message(Role.USER, user_text))
        async for event in self._drive(state, thread_id):
            yield event

    async def resume_turn(self, *, thread_id: str, approved: bool) -> AsyncIterator[BrainEvent]:
        """Resume a turn paused on a confirmation. Approve executes the pending
        action; deny records a refusal. Either way the loop then continues."""
        state = self._checkpointer.load(thread_id)
        if state is None or state.pending is None:
            raise RuntimeError(f"no pending confirmation for thread {thread_id!r}")
        pending = state.pending
        state.pending = None

        if approved:
            result = await self._gate.execute(pending.action, pending.reversibility)
            yield self._completed(pending.call, result)
            state.messages.append(Message(Role.TOOL, result.output))
        else:
            yield ToolCompleted(
                call_id=pending.call.id,
                tool=pending.call.name,
                ok=False,
                output="refused by user",
                refused=True,
            )
            state.messages.append(Message(Role.TOOL, "refused by user"))

        async for event in self._drive(state, thread_id):
            yield event

    # -- loop --------------------------------------------------------------

    async def _drive(self, state: AgentState, thread_id: str) -> AsyncIterator[BrainEvent]:
        for _step in range(self._max_steps):
            text_parts: list[str] = []
            tool_calls: list[ToolCall] = []
            async for chunk in self._llm.stream(self._prompt(state), self._tools):
                if chunk.text:
                    text_parts.append(chunk.text)
                    yield TokenDelta(chunk.text)
                tool_calls.extend(chunk.tool_calls)

            assistant_text = "".join(text_parts)

            if not tool_calls:
                state.messages.append(Message(Role.ASSISTANT, assistant_text))
                self._checkpointer.save(thread_id, state)
                yield TurnComplete(assistant_text.strip())
                return

            state.messages.append(Message(Role.ASSISTANT, assistant_text))
            paused = False
            async for event in self._run_tools(state, thread_id, tool_calls):
                yield event
                if isinstance(event, ConfirmationRequired):
                    paused = True
            if paused:
                return  # wait for resume_turn

        log.warning("step_limit_reached", steps=self._max_steps)
        self._checkpointer.save(thread_id, state)
        yield StepLimitReached(self._max_steps)

    async def _run_tools(
        self, state: AgentState, thread_id: str, tool_calls: list[ToolCall]
    ) -> AsyncIterator[BrainEvent]:
        for call in tool_calls:
            action = Action(tool=call.name, arguments=call.arguments)
            decision, reversibility = self._gate.decide(action)

            if decision is GateDecision.REFUSE:
                yield ToolCompleted(
                    call_id=call.id,
                    tool=call.name,
                    ok=False,
                    output="refused: not allowed",
                    refused=True,
                )
                state.messages.append(Message(Role.TOOL, "refused: not allowed"))
                continue

            if decision is GateDecision.CONFIRM:
                pending = PendingConfirmation(
                    id=uuid.uuid4().hex,
                    call=call,
                    action=action,
                    reversibility=reversibility,
                )
                state.pending = pending
                self._checkpointer.save(thread_id, state)
                yield ConfirmationRequired(
                    id=pending.id,
                    tool=call.name,
                    arguments=call.arguments,
                    reversibility=reversibility,
                )
                return  # first confirmation pauses the turn

            yield ToolInvoked(
                call_id=call.id,
                tool=call.name,
                arguments=call.arguments,
                reversibility=reversibility,
            )
            result = await self._gate.execute(action, reversibility)
            yield self._completed(call, result)
            state.messages.append(Message(Role.TOOL, result.output))

    # -- helpers -----------------------------------------------------------

    def _prompt(self, state: AgentState) -> list[Message]:
        return [Message(Role.SYSTEM, self._system_prompt), *state.messages]

    def _new_state(self) -> AgentState:
        return AgentState()

    @staticmethod
    def _completed(call: ToolCall, result: ActionResult) -> ToolCompleted:
        return ToolCompleted(
            call_id=call.id,
            tool=call.name,
            ok=result.ok,
            output=result.output or (result.error or ""),
        )

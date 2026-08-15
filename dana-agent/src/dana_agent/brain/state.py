"""Brain state + checkpointing (memory).

The state is the single object flowing through the graph; keep it minimal and
typed (langgraph-agent-patterns skill). A checkpointer persists it per thread
(conversation session) so memory survives across turns and a paused turn can be
resumed. In-memory here; a durable SQLite checkpointer arrives with Phase 4's
real integration.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Protocol

from dana_agent.types import Action, Message, Reversibility, ToolCall


@dataclass(frozen=True, slots=True)
class PendingConfirmation:
    """A destructive tool call paused awaiting a human decision."""

    id: str
    call: ToolCall
    action: Action
    reversibility: Reversibility


@dataclass(slots=True)
class AgentState:
    messages: list[Message] = field(default_factory=list)
    pending: PendingConfirmation | None = None

    def copy(self) -> AgentState:
        """Shallow-immutable snapshot: copies the message list so a stored
        checkpoint is not mutated by an in-flight turn."""
        return replace(self, messages=list(self.messages))


class Checkpointer(Protocol):
    """Persists brain state keyed by a thread (conversation) id."""

    def load(self, thread_id: str) -> AgentState | None: ...

    def save(self, thread_id: str, state: AgentState) -> None: ...


class InMemoryCheckpointer:
    """Dev/test checkpointer. Stores a copy so callers can't mutate the store
    by reference."""

    def __init__(self) -> None:
        self._store: dict[str, AgentState] = {}

    def load(self, thread_id: str) -> AgentState | None:
        stored = self._store.get(thread_id)
        return stored.copy() if stored is not None else None

    def save(self, thread_id: str, state: AgentState) -> None:
        self._store[thread_id] = state.copy()

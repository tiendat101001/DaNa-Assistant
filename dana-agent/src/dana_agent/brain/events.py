"""Events streamed from a brain turn.

Streaming, not blocking: token deltas feed TTS immediately and tool/state events
let a UI show what the agent is doing (langgraph-agent-patterns skill). A turn
ends with exactly one terminal event: TurnComplete, ConfirmationRequired (paused
for approval), or StepLimitReached.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

from dana_agent.types import Reversibility


@dataclass(frozen=True, slots=True)
class TokenDelta:
    text: str


@dataclass(frozen=True, slots=True)
class ToolInvoked:
    call_id: str
    tool: str
    arguments: dict[str, object]
    reversibility: Reversibility


@dataclass(frozen=True, slots=True)
class ToolCompleted:
    call_id: str
    tool: str
    ok: bool
    output: str
    refused: bool = False


@dataclass(frozen=True, slots=True)
class ConfirmationRequired:
    """Terminal (paused) event: a destructive tool needs approval. Resume with
    ``Brain.resume_turn(thread_id, approved=...)``."""

    id: str
    tool: str
    arguments: dict[str, object]
    reversibility: Reversibility


@dataclass(frozen=True, slots=True)
class TurnComplete:
    text: str


@dataclass(frozen=True, slots=True)
class StepLimitReached:
    steps: int


BrainEvent: TypeAlias = (
    TokenDelta
    | ToolInvoked
    | ToolCompleted
    | ConfirmationRequired
    | TurnComplete
    | StepLimitReached
)

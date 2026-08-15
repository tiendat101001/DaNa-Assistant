"""The agent brain: a LangGraph-style typed state graph over the LLM provider.

Models the assistant's reasoning as an explicit agent/tools loop with a
conditional edge (langgraph-agent-patterns skill): the agent node streams the
model; if it emits tool calls they are gated and executed, then control returns
to the agent; otherwise the turn ends. Destructive tools pause the turn for a
human decision (the confirmation gate), resumable because state is checkpointed.

Implemented over DaNa's own provider abstractions (``LLMProvider`` +
``ActionGate``) rather than importing LangGraph directly, because the model here
is a streaming provider (a Pipecat service in Phase 3), not a LangChain chat
model — see the brain module docstring in ``agent.py``.
"""

from dana_agent.brain.agent import Brain
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

__all__ = [
    "AgentState",
    "Brain",
    "BrainEvent",
    "Checkpointer",
    "ConfirmationRequired",
    "InMemoryCheckpointer",
    "PendingConfirmation",
    "StepLimitReached",
    "TokenDelta",
    "ToolCompleted",
    "ToolInvoked",
    "TurnComplete",
]

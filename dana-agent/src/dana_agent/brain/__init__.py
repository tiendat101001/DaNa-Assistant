"""The agent brain: a LangGraph agent/tools graph over a chat model built with
``init_chat_model`` (langgraph-agent-patterns skill).

Uses LangGraph's own nodes (``ToolNode`` + ``tools_condition``), token streaming
(``astream_events`` -> ``on_chat_model_stream``), checkpointer-backed memory, and
native ``interrupt`` for the destructive-action confirmation gate — rather than a
hand-rolled loop. The chat model is provider-swappable purely from ``LLMConfig``
via ``init_chat_model``; an offline ``MockChatModel`` backs the ``mock`` provider.
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
from dana_agent.brain.model import BRAIN_PROVIDERS, MockChatModel, build_chat_model
from dana_agent.brain.tools import ToolSpec, build_gated_tool, build_gated_tools

__all__ = [
    "BRAIN_PROVIDERS",
    "Brain",
    "BrainEvent",
    "ConfirmationRequired",
    "MockChatModel",
    "StepLimitReached",
    "TokenDelta",
    "ToolCompleted",
    "ToolInvoked",
    "ToolSpec",
    "TurnComplete",
    "build_chat_model",
    "build_gated_tool",
    "build_gated_tools",
]

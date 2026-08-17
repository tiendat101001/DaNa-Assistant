"""Phase 3 — Pipecat conversation core.

Assembles the streaming voice pipeline (STT → Brain → TTS) using Pipecat's
frame-based architecture so every stage starts before its upstream completes,
hitting the ≤ 700 ms latency hard gate (docs/PLAN.md §7).

Public surface:

- ``ConversationSession`` — manages the pipeline lifecycle for one session.
"""

from dana_agent.conversation.session import ConversationSession

__all__ = ["ConversationSession"]

"""Wake gate: the custom component in front of the conversation core — VAD
gate + KWS (wake engine) + an explicit IDLE/ACTIVE state machine
(voice-agent-architecture skill). Distinct from ``providers.wake`` (the
swappable wake-engine backends)."""

from dana_agent.wake.gate import FrameOutcome, WakeGate
from dana_agent.wake.state_machine import (
    State,
    StateMachine,
    StateTransition,
    TransitionCause,
)

__all__ = [
    "FrameOutcome",
    "State",
    "StateMachine",
    "StateTransition",
    "TransitionCause",
    "WakeGate",
]

"""Domain data types exchanged at provider boundaries.

These are the neutral currency of the provider abstraction: interfaces speak
in these types so backends are interchangeable and no vendor SDK type ever
leaks through an ABC (provider-plugin-architecture skill).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

# --- audio ---------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AudioFrame:
    """A small chunk of captured PCM audio flowing toward the wake gate / STT."""

    pcm: bytes
    sample_rate: int
    channels: int = 1


@dataclass(frozen=True, slots=True)
class AudioChunk:
    """A chunk of synthesized PCM audio flowing from TTS/S2S toward playback."""

    pcm: bytes
    sample_rate: int


# --- speech-to-text ------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Transcript:
    """A (possibly partial) STT hypothesis. Stream emits partials then a final."""

    text: str
    is_final: bool
    language: str = "en"
    confidence: float | None = None


# --- wake word -----------------------------------------------------------


class WakeEventType(StrEnum):
    WAKE = "wake"  # wake phrase detected -> IDLE transitions to ACTIVE
    END = "end"  # end phrase detected -> ACTIVE transitions to IDLE


@dataclass(frozen=True, slots=True)
class WakeEvent:
    type: WakeEventType
    phrase: str
    confidence: float


# --- llm / brain ---------------------------------------------------------


class Role(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass(frozen=True, slots=True)
class Message:
    role: Role
    content: str


@dataclass(frozen=True, slots=True)
class Tool:
    """A tool the LLM may call. ``description`` reads as instructions to the
    model — it is the model's only guide to when/how to use the tool."""

    name: str
    description: str
    parameters: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, object]


@dataclass(frozen=True, slots=True)
class LLMChunk:
    """One streamed increment from an LLM: text delta and/or tool calls."""

    text: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    done: bool = False


# --- actions -------------------------------------------------------------


class Reversibility(StrEnum):
    """How bad a mistake would be — gates actions by consequence, not phrasing
    (safe-computer-use skill)."""

    READ_ONLY = "read_only"  # run freely
    REVERSIBLE = "reversible"  # run; log; keep undo info
    DESTRUCTIVE = "destructive"  # require explicit confirmation
    OUT_OF_SCOPE = "out_of_scope"  # refuse (not allowlisted)


@dataclass(frozen=True, slots=True)
class Action:
    """A structured, typed tool call proposed by the LLM. Never raw shell."""

    tool: str
    arguments: dict[str, object]


@dataclass(frozen=True, slots=True)
class ActionResult:
    ok: bool
    output: str = ""
    error: str | None = None

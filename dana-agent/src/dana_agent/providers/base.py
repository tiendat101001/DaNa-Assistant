"""Provider interfaces — one ABC per capability, all the same shape.

Design rule (provider-plugin-architecture skill): interfaces are built around
what call sites need, kept minimal, async + streaming by default, and speak in
neutral domain types (dana_agent.types) — never a vendor SDK's types.

Bridge note (ADR-0001): the cascade providers (STT/LLM/TTS/S2S) are, in later
phases, thin factories over Pipecat service objects — we do not re-implement
streaming clients. Their ABCs here are the neutral streaming contract the rest
of the app and the offline test suite depend on; the mock implementations honor
it with no external dependencies.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Literal

from pydantic import BaseModel, ConfigDict

from dana_agent.types import (
    Action,
    ActionResult,
    AudioChunk,
    AudioFrame,
    LLMChunk,
    Message,
    Reversibility,
    Tool,
    Transcript,
    WakeEvent,
)

LatencyClass = Literal["realtime", "fast", "batch"]


class Capabilities(BaseModel):
    """What a provider can do — declared as data so a chosen stack can be
    validated at startup and fail fast, instead of crashing mid-conversation."""

    model_config = ConfigDict(frozen=True)

    streaming: bool
    languages: frozenset[str]
    self_hosted: bool = False
    tool_calling: bool = False
    latency_class: LatencyClass = "fast"


class Provider(ABC):
    """Common base: every provider declares its capabilities."""

    @property
    @abstractmethod
    def capabilities(self) -> Capabilities:
        """Declare what this provider supports so combinations can be validated."""

    async def aclose(self) -> None:
        """Release any resources (clients, models, devices). Default no-op so
        stateless providers need not override."""
        return None


class VADProvider(Provider):
    """Voice activity detection — the cheap gate that decides "is someone
    speaking" so heavier models (KWS, STT) never run on silence. A real VAD
    model (Silero) is far better than a bare dB threshold."""

    @abstractmethod
    def is_speech(self, frame: AudioFrame) -> bool:
        """Return True if the frame contains speech."""


class WakeWordProvider(Provider):
    """Always-on, on-device keyword spotter. Fully custom (not a Pipecat
    service): sits in front of the pipeline as the IDLE/ACTIVE gate."""

    @abstractmethod
    def process(self, frame: AudioFrame) -> WakeEvent | None:
        """Feed one audio frame; return a WakeEvent on a wake/end hit, else None."""


class STTProvider(Provider):
    """Streaming speech-to-text."""

    @abstractmethod
    def stream(self, audio: AsyncIterator[AudioFrame]) -> AsyncIterator[Transcript]:
        """Consume an audio stream; yield partial transcripts then a final one."""


class LLMProvider(Provider):
    """Streaming large language model (the brain's underlying model)."""

    @abstractmethod
    def stream(
        self, messages: list[Message], tools: list[Tool] | None = None
    ) -> AsyncIterator[LLMChunk]:
        """Yield response chunks (text deltas and/or tool calls) as they arrive."""


class TTSProvider(Provider):
    """Streaming text-to-speech."""

    @abstractmethod
    def stream(self, text: AsyncIterator[str]) -> AsyncIterator[AudioChunk]:
        """Consume a text stream (sentence-by-sentence); yield audio chunks so
        playback can begin on the first sentence, not the whole reply."""


class S2SProvider(Provider):
    """Single speech-to-speech model — an alternative to the STT+LLM+TTS
    cascade. The conversation abstraction accepts either (docs/PLAN.md §6)."""

    @abstractmethod
    def stream(self, audio: AsyncIterator[AudioFrame]) -> AsyncIterator[AudioChunk]:
        """Consume input audio; yield response audio."""


class ActionProvider(Provider):
    """Executes structured, typed tool calls against a browser/OS. The model
    proposes; this layer validates and disposes (safe-computer-use skill)."""

    @abstractmethod
    def classify(self, action: Action) -> Reversibility:
        """Classify an action by consequence so the gate can decide the gate."""

    @abstractmethod
    async def execute(self, action: Action) -> ActionResult:
        """Execute an already-authorized action; return a structured result."""

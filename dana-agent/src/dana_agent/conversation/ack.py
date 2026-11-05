"""Instant acknowledgement: a short cached phrase played at end of speech.

The phrase is synthesized once per language at startup (:class:`AckCache`) with
the active TTS provider. :class:`AckProcessor` then pushes the cached PCM the
moment an ACTIVE turn's VAD stop arrives, ahead of the real reply. It is
opt-in (``latency.ack_enabled``), never fires for typed turns (no VAD frames)
nor for utterances that are too short / too quiet to be speech, and a cache
miss means silently no ack.
"""

from __future__ import annotations

import array
import math
from collections.abc import AsyncIterator
from dataclasses import dataclass

from pipecat.frames.frames import (
    Frame,
    InputAudioRawFrame,
    TTSAudioRawFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from dana_agent.observability import get_logger
from dana_agent.providers.base import TTSProvider

log = get_logger(__name__)


@dataclass
class AckAudioRawFrame(TTSAudioRawFrame):  # type: ignore[misc]
    """Audio of the acknowledgement; distinguishable from the real reply so the
    latency clock does not mistake it for the first reply audio."""


class AckCache:
    """language -> list of (pcm, sample_rate) chunks."""

    def __init__(self) -> None:
        self._chunks: dict[str, list[tuple[bytes, int]]] = {}

    def get(self, language: str) -> list[tuple[bytes, int]] | None:
        return self._chunks.get(language) or None

    def put(self, language: str, chunks: list[tuple[bytes, int]]) -> None:
        if chunks:
            self._chunks[language] = chunks

    async def fill(self, tts: TTSProvider, phrases: dict[str, str]) -> None:
        for lang, phrase in phrases.items():
            if not phrase.strip():
                continue

            async def text(p: str = phrase) -> AsyncIterator[str]:
                yield p

            try:
                out = [(c.pcm, c.sample_rate) async for c in tts.stream(text())]
            except Exception as exc:  # noqa: BLE001 - a missing ack is harmless
                log.warning("ack_synth_failed", language=lang, exc=str(exc))
                continue
            self.put(lang, out)
            log.info("ack_cached", language=lang, chunks=len(out))


def _rms(pcm: bytes) -> float:
    samples = array.array("h")
    samples.frombytes(pcm[: len(pcm) // 2 * 2])
    sub = samples[::8]
    if not sub:
        return 0.0
    return math.sqrt(sum(x * x for x in sub) / len(sub))


class AckProcessor(FrameProcessor):  # type: ignore[misc]
    """Pushes the cached ack on VAD stop. Place it after the wake gate."""

    def __init__(
        self,
        cache: AckCache,
        *,
        language: str = "en",
        min_speech_ms: int = 250,
        min_rms: float = 150.0,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        self._cache = cache
        self._language = language
        self._min_bytes = min_speech_ms * 32
        self._min_rms = min_rms
        self._in_speech = False
        self._bytes = 0
        self._peak_rms = 0.0

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)

        if isinstance(frame, VADUserStartedSpeakingFrame):
            self._in_speech = True
            self._bytes = 0
            self._peak_rms = 0.0
        elif isinstance(frame, InputAudioRawFrame):
            if self._in_speech:
                self._bytes += len(frame.audio)
                self._peak_rms = max(self._peak_rms, _rms(frame.audio))
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            speech = self._in_speech and self._bytes >= self._min_bytes
            loud = self._peak_rms >= self._min_rms
            self._in_speech = False
            # Forward the VAD stop first so STT starts before the ack is queued.
            await self.push_frame(frame, direction)
            chunks = self._cache.get(self._language) or self._cache.get("en")
            if speech and loud and chunks:
                for pcm, rate in chunks:
                    await self.push_frame(
                        AckAudioRawFrame(audio=pcm, sample_rate=rate, num_channels=1)
                    )
            return
        await self.push_frame(frame, direction)

"""Deterministic mock TTS. Emits one silent PCM AudioChunk per input text chunk
so downstream playback/latency logic can run with no audio backend."""

from __future__ import annotations

from collections.abc import AsyncIterator

from dana_agent.config import TTSConfig
from dana_agent.providers.base import Capabilities, TTSProvider
from dana_agent.providers.registry import register
from dana_agent.types import AudioChunk

_SAMPLE_RATE = 16_000


@register("tts", "mock")
class MockTTS(TTSProvider):
    def __init__(self, cfg: TTSConfig) -> None:
        self._cfg = cfg

    async def stream(self, text: AsyncIterator[str]) -> AsyncIterator[AudioChunk]:
        async for chunk in text:
            if not chunk:
                continue
            # 20 ms of silence per input chunk; length stands in for content.
            samples = max(1, len(chunk)) * (_SAMPLE_RATE // 50)
            yield AudioChunk(pcm=b"\x00\x00" * samples, sample_rate=_SAMPLE_RATE)

    @property
    def capabilities(self) -> Capabilities:
        return Capabilities(streaming=True, languages=frozenset({"en", "vi"}), latency_class="fast")

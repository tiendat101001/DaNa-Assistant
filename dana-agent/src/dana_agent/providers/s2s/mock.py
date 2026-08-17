"""Deterministic mock speech-to-speech provider. Consumes input audio frames
and emits one silent AudioChunk per frame, proving the cascade-or-S2S
abstraction end to end without a real realtime model."""

from __future__ import annotations

from collections.abc import AsyncIterator

from dana_agent.config import S2SConfig
from dana_agent.providers.base import Capabilities, S2SProvider
from dana_agent.providers.registry import register
from dana_agent.types import AudioChunk, AudioFrame

_SAMPLE_RATE = 16_000


@register("s2s", "mock")
class MockS2S(S2SProvider):
    def __init__(self, cfg: S2SConfig) -> None:
        self._cfg = cfg

    async def stream(self, audio: AsyncIterator[AudioFrame]) -> AsyncIterator[AudioChunk]:
        async for _frame in audio:
            yield AudioChunk(pcm=b"\x00\x00" * (_SAMPLE_RATE // 50), sample_rate=_SAMPLE_RATE)

    @property
    def capabilities(self) -> Capabilities:
        return Capabilities(
            streaming=True,
            languages=frozenset({"en", "vi"}),
            self_hosted=True,
            latency_class="realtime",
        )

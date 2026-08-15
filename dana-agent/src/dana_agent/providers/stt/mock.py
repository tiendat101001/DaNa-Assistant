"""Deterministic mock STT. Decodes each frame's PCM as utf-8 text (fixtures put
the intended transcript there), streaming a partial then a final Transcript."""

from __future__ import annotations

from collections.abc import AsyncIterator

from dana_agent.config import STTConfig
from dana_agent.providers.base import Capabilities, STTProvider
from dana_agent.providers.registry import register
from dana_agent.types import AudioFrame, Transcript


@register("stt", "mock")
class MockSTT(STTProvider):
    def __init__(self, cfg: STTConfig) -> None:
        self._cfg = cfg

    async def stream(self, audio: AsyncIterator[AudioFrame]) -> AsyncIterator[Transcript]:
        words: list[str] = []
        async for frame in audio:
            token = frame.pcm.decode("utf-8", errors="ignore").strip()
            if not token:
                continue
            words.append(token)
            yield Transcript(" ".join(words), is_final=False)
        yield Transcript(" ".join(words), is_final=True, confidence=0.99)

    @property
    def capabilities(self) -> Capabilities:
        return Capabilities(streaming=True, languages=frozenset({"en", "vi"}), latency_class="fast")

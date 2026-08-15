"""Deterministic mock VAD: any non-zero PCM byte counts as speech, all-zero
(or empty) frames as silence. Lets sentinel/silence fixtures drive the wake
gate offline. The real Silero VAD arrives in Phase 3 (TASK-003) under this ABC.
"""

from __future__ import annotations

from dana_agent.config import VADConfig
from dana_agent.providers.base import Capabilities, VADProvider
from dana_agent.providers.registry import register
from dana_agent.types import AudioFrame


@register("vad", "mock")
class MockVAD(VADProvider):
    def __init__(self, cfg: VADConfig) -> None:
        self._cfg = cfg

    def is_speech(self, frame: AudioFrame) -> bool:
        return any(frame.pcm)

    @property
    def capabilities(self) -> Capabilities:
        return Capabilities(streaming=True, languages=frozenset({"en", "vi"}))

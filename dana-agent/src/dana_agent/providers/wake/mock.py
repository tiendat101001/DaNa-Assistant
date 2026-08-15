"""Deterministic mock wake-word engine — the default for tests and local dev.

Fires on sentinel frames so recorded-audio fixtures can drive the state machine
without a real KWS model: a frame whose PCM equals the (utf-8) wake phrase emits
a WAKE event; one equal to the end phrase emits END. Real engines (Porcupine,
openWakeWord) arrive in Phase 2 under the same ABC.
"""

from __future__ import annotations

from dana_agent.config import WakeConfig
from dana_agent.providers.base import Capabilities, WakeWordProvider
from dana_agent.providers.registry import register
from dana_agent.types import AudioFrame, WakeEvent, WakeEventType


@register("wake", "mock")
class MockWakeWord(WakeWordProvider):
    def __init__(self, cfg: WakeConfig) -> None:
        self._cfg = cfg
        self._wake = cfg.phrase.encode("utf-8")
        self._end = cfg.end_phrase.encode("utf-8")

    def process(self, frame: AudioFrame) -> WakeEvent | None:
        if frame.pcm == self._wake:
            return WakeEvent(WakeEventType.WAKE, self._cfg.phrase, confidence=1.0)
        if frame.pcm == self._end:
            return WakeEvent(WakeEventType.END, self._cfg.end_phrase, confidence=1.0)
        return None

    @property
    def capabilities(self) -> Capabilities:
        return Capabilities(streaming=True, languages=frozenset({"en", "vi"}))

"""Real Silero VAD — wraps pipecat's bundled SileroVADAnalyzer (ONNX model).

``voice_confidence(buffer)`` is synchronous so it satisfies the ``VADProvider``
ABC without needing an event loop.  The pipecat transport also instantiates its
own ``SileroVADAnalyzer`` for VAD-frame emission; these are independent
instances, which is fine — the ONNX model is stateless between calls.
"""

from __future__ import annotations

from pipecat.audio.vad.silero import SileroVADAnalyzer

from dana_agent.config import VADConfig
from dana_agent.providers.base import Capabilities, VADProvider
from dana_agent.providers.registry import register
from dana_agent.types import AudioFrame


@register("vad", "silero")
class SileroVADProvider(VADProvider):
    """VAD via pipecat's bundled Silero ONNX model (on-device, no API key)."""

    def __init__(self, cfg: VADConfig) -> None:
        self._cfg = cfg
        self._analyzer = SileroVADAnalyzer()

    def is_speech(self, frame: AudioFrame) -> bool:
        confidence: float = self._analyzer.voice_confidence(frame.pcm)
        return confidence >= self._cfg.threshold

    @property
    def capabilities(self) -> Capabilities:
        return Capabilities(
            streaming=True,
            languages=frozenset({"en", "vi"}),
            self_hosted=True,
        )

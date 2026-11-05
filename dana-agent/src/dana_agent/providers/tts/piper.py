"""Piper TTS — fast CPU neural TTS (VITS via ONNX), English + Vietnamese.

Piper is the practical low-latency CPU voice: a short sentence synthesizes in
well under 200 ms on a laptop CPU, with no network and no key. Both ``piper-tts``
API generations are supported, detected with ``hasattr``:

* new (>= 1.3): ``PiperVoice.load(model, config_path=...)`` and
  ``voice.synthesize(text)`` yielding per-sentence chunks carrying
  ``audio_int16_bytes`` / ``sample_rate``;
* old (1.2): ``PiperVoice.load(model, config_path=...)`` and
  ``voice.synthesize_stream_raw(text)`` yielding int16 ``bytes`` per sentence,
  with the rate in ``voice.config.sample_rate``.

Synthesis is CPU-bound and runs in a worker thread (``asyncio.to_thread``) so
the event loop (VAD, barge-in, playback) never stalls. Each Piper sentence is
re-cut into ~150 ms ``AudioChunk``s. Voices are ``<voice>.onnx`` +
``<voice>.onnx.json``; see ``resolve_local_model`` for lookup order.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Any

from dana_agent.config import TTSConfig
from dana_agent.observability import get_logger
from dana_agent.providers.base import Capabilities, TTSProvider
from dana_agent.providers.registry import register
from dana_agent.providers.tts._common import resolve_local_model, slice_pcm
from dana_agent.types import AudioChunk

log = get_logger(__name__)

_DONE = object()


def _next_or_done(it: Iterator[Any]) -> Any:
    return next(it, _DONE)


@register("tts", "piper")
class PiperTTS(TTSProvider):
    """Local CPU Piper voice. Requires ``pip install piper-tts``."""

    def __init__(self, cfg: TTSConfig, *, models_dir: str | None = None) -> None:
        self._cfg = cfg
        self._models_dir = models_dir
        self._voice: Any = None
        self._sample_rate: int = 22_050
        self._load_lock = asyncio.Lock()

    @property
    def capabilities(self) -> Capabilities:
        return Capabilities(
            streaming=True,
            languages=frozenset({"en", "vi"}),
            self_hosted=True,
            latency_class="realtime",
        )

    # -- loading -----------------------------------------------------------

    def _load_sync(self) -> Any:
        try:
            from piper import PiperVoice  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "piper-tts is required for the piper TTS provider: pip install piper-tts"
            ) from exc
        model = resolve_local_model(self._cfg, ".onnx", models_dir=self._models_dir)
        cfg_json = Path(f"{model}.json")
        if not cfg_json.is_file():
            alt = model.with_suffix(".json")
            cfg_json = alt if alt.is_file() else cfg_json
        voice = PiperVoice.load(str(model), config_path=str(cfg_json))
        rate = getattr(getattr(voice, "config", None), "sample_rate", None)
        if isinstance(rate, int) and rate > 0:
            self._sample_rate = rate
        log.info("piper_voice_loaded", model=str(model), sample_rate=self._sample_rate)
        return voice

    async def _ensure_voice(self) -> Any:
        if self._voice is None:
            async with self._load_lock:
                if self._voice is None:
                    self._voice = await asyncio.to_thread(self._load_sync)
        return self._voice

    async def warmup(self) -> None:
        """Load the voice and synthesize one short word (primes ONNX/espeak)."""
        voice = await self._ensure_voice()
        await asyncio.to_thread(lambda: sum(len(p) for p in self._synth_sync(voice, "ok")))

    # -- synthesis ---------------------------------------------------------

    def _synth_sync(self, voice: Any, text: str) -> Iterator[bytes]:
        """Yield int16 PCM bytes, one item per Piper sentence (blocking)."""
        if hasattr(voice, "synthesize"):
            for chunk in voice.synthesize(text):
                rate = getattr(chunk, "sample_rate", None)
                if isinstance(rate, int) and rate > 0:
                    self._sample_rate = rate
                yield bytes(chunk.audio_int16_bytes)
        elif hasattr(voice, "synthesize_stream_raw"):
            for raw in voice.synthesize_stream_raw(text):
                yield bytes(raw)
        else:
            raise RuntimeError("Unsupported piper-tts version: no synthesize API on PiperVoice")

    async def stream(self, text: AsyncIterator[str]) -> AsyncIterator[AudioChunk]:
        voice = await self._ensure_voice()
        async for sentence in text:
            sentence = sentence.strip()
            if not sentence:
                continue
            it = self._synth_sync(voice, sentence)
            while True:
                # One blocking next() per worker hop keeps the loop free and
                # lets us yield the first Piper sentence before the rest exist.
                pcm = await asyncio.to_thread(_next_or_done, it)
                if pcm is _DONE:
                    break
                for piece in slice_pcm(pcm, self._sample_rate):
                    yield AudioChunk(pcm=piece, sample_rate=self._sample_rate)

    async def aclose(self) -> None:
        self._voice = None

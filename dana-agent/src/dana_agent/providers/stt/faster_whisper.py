"""CPU fallback STT via faster-whisper (CTranslate2, int8). Utterance-level."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import AsyncIterator
from typing import Any

import numpy as np

from dana_agent.config import STTConfig
from dana_agent.observability import get_logger
from dana_agent.providers.base import Capabilities, STTProvider
from dana_agent.providers.registry import register
from dana_agent.runtime.audio import resample
from dana_agent.types import AudioFrame, Transcript

log = get_logger(__name__)


@register("stt", "faster_whisper")
class FasterWhisperSTT(STTProvider):
    def __init__(self, cfg: STTConfig) -> None:
        self._cfg = cfg
        self._model: Any = None
        self._lock = threading.Lock()

    @property
    def capabilities(self) -> Capabilities:
        return Capabilities(
            streaming=True,
            languages=frozenset({"en", "vi"}),
            self_hosted=True,
            latency_class="fast",
        )

    def _load(self) -> Any:
        with self._lock:
            if self._model is None:
                import importlib

                WhisperModel = importlib.import_module("faster_whisper").WhisperModel

                name = self._cfg.model_path or self._cfg.model
                if name in ("", "mock-stt"):
                    name = "tiny"
                self._model = WhisperModel(
                    name,
                    device="cpu",
                    compute_type="int8",
                    cpu_threads=self._cfg.num_threads,
                )
            return self._model

    async def warmup(self) -> None:
        model = await asyncio.to_thread(self._load)
        await asyncio.to_thread(self._run, model, np.zeros(16_000, dtype=np.float32))

    def _run(self, model: Any, wav: np.ndarray) -> str:
        segments, _info = model.transcribe(
            wav,
            language=self._cfg.language_code,
            beam_size=1,
            vad_filter=False,
            condition_on_previous_text=False,
        )
        return " ".join(s.text.strip() for s in segments).strip()

    async def stream(self, audio: AsyncIterator[AudioFrame]) -> AsyncIterator[Transcript]:
        chunks: list[np.ndarray] = []
        async for f in audio:
            x = (
                np.frombuffer(f.pcm[: len(f.pcm) // 2 * 2], dtype="<i2").astype(np.float32)
                / 32768.0
            )
            if f.channels > 1:
                x = x[: x.size // f.channels * f.channels].reshape(-1, f.channels).mean(axis=1)
            chunks.append(resample(x, f.sample_rate))
        lang = self._cfg.language_code
        if not chunks:
            yield Transcript("", is_final=True, language=lang)
            return
        model = await asyncio.to_thread(self._load)
        text = await asyncio.to_thread(self._run, model, np.concatenate(chunks))
        yield Transcript(text, is_final=True, language=lang)

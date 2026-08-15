"""AssemblyAI streaming STT — real-time transcription via WebSocket.

Bridges AssemblyAI's callback-based ``RealtimeTranscriber`` to the
``STTProvider.stream`` async-iterator contract using an asyncio.Queue.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator
from typing import Any

from dana_agent.config import STTConfig
from dana_agent.providers.base import Capabilities, STTProvider
from dana_agent.providers.registry import register
from dana_agent.types import AudioFrame, Transcript

_SAMPLE_RATE = 16_000


@register("stt", "assemblyai")
class AssemblyAISTT(STTProvider):
    """Real-time STT via AssemblyAI WebSocket API.

    Requires the ``assemblyai`` package (``pip install assemblyai``).
    API key sourced from ``DANA_CONVERSATION__STT__API_KEY`` (never hardcoded).
    Supports English and Vietnamese through the AssemblyAI ``best`` model.
    """

    def __init__(self, cfg: STTConfig) -> None:
        self._cfg = cfg

    @property
    def capabilities(self) -> Capabilities:
        return Capabilities(
            streaming=True,
            languages=frozenset({"en", "vi"}),
            latency_class="realtime",
        )

    async def stream(self, audio: AsyncIterator[AudioFrame]) -> AsyncIterator[Transcript]:
        try:
            import assemblyai as aai  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "assemblyai is required for the assemblyai STT provider: "
                "install dana-agent[cloud] or pip install assemblyai"
            ) from exc

        api_key = self._cfg.api_key.get_secret_value() if self._cfg.api_key else None
        if api_key:
            aai.settings.api_key = api_key

        queue: asyncio.Queue[Transcript | BaseException | None] = asyncio.Queue()
        loop = asyncio.get_event_loop()

        def _on_data(transcript: Any) -> None:
            if not transcript.text:
                return
            # RealtimeFinalTranscript is a subclass of RealtimeTranscript
            is_final = type(transcript).__name__ == "RealtimeFinalTranscript"
            t = Transcript(
                text=transcript.text,
                is_final=is_final,
                confidence=getattr(transcript, "confidence", None),
            )
            loop.call_soon_threadsafe(queue.put_nowait, t)

        def _on_error(error: BaseException) -> None:
            loop.call_soon_threadsafe(queue.put_nowait, error)

        transcriber = aai.RealtimeTranscriber(
            on_data=_on_data,
            on_error=_on_error,
            sample_rate=_SAMPLE_RATE,
        )

        await asyncio.to_thread(transcriber.connect)

        async def _feed() -> None:
            try:
                async for frame in audio:
                    # Resample if the frame's sample rate differs from 16 kHz;
                    # for now we trust callers to send compatible audio.
                    await asyncio.to_thread(transcriber.stream, frame.pcm)
            finally:
                await asyncio.to_thread(transcriber.close)
                loop.call_soon_threadsafe(queue.put_nowait, None)  # sentinel

        feeder = asyncio.create_task(_feed())
        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                if isinstance(item, BaseException):
                    raise item
                yield item
        finally:
            feeder.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await feeder

"""AssemblyAI streaming STT — real-time transcription via WebSocket (SDK v1+).

Bridges AssemblyAI's asyncio ``AsyncRealTimeTranscriber`` (v1.0 SDK lives in
``assemblyai.streaming.v3``) to the ``STTProvider.stream`` async-iterator
contract using an asyncio.Queue.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator

from dana_agent.config import STTConfig
from dana_agent.providers.base import Capabilities, STTProvider
from dana_agent.providers.registry import register
from dana_agent.types import AudioFrame, Transcript

_SAMPLE_RATE = 16_000


@register("stt", "assemblyai")
class AssemblyAISTT(STTProvider):
    """Real-time STT via AssemblyAI WebSocket API (SDK ≥ 1.0).

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
            from assemblyai.streaming.v3 import (  # noqa: PLC0415
                AsyncRealTimeTranscriber,
                RealTimeEvents,
            )
            from assemblyai.streaming.v3.models import RealTimeParameters  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "assemblyai>=1.0 is required for the assemblyai STT provider: "
                "install dana-agent[cloud] or pip install 'assemblyai>=1.0'"
            ) from exc

        api_key = self._cfg.api_key.get_secret_value() if self._cfg.api_key else None
        if not api_key:
            raise ValueError(
                "DANA_CONVERSATION__STT__API_KEY must be set for the assemblyai STT provider"
            )

        queue: asyncio.Queue[Transcript | BaseException | None] = asyncio.Queue()

        def _on_turn(_client: object, event: object) -> None:
            text = getattr(event, "transcript", "")
            if not text:
                return
            is_final = bool(getattr(event, "end_of_turn", False))
            confidence = getattr(event, "end_of_turn_confidence", None)
            t = Transcript(
                text=text,
                is_final=is_final,
                confidence=confidence,
            )
            queue.put_nowait(t)

        def _on_error(_client: object, event: object) -> None:
            msg = getattr(event, "message", str(event))
            queue.put_nowait(RuntimeError(f"AssemblyAI error: {msg}"))

        def _on_termination(_client: object, _event: object) -> None:
            queue.put_nowait(None)  # sentinel: session closed

        transcriber = AsyncRealTimeTranscriber(api_key=api_key)
        transcriber.on(RealTimeEvents.Turn, _on_turn)
        transcriber.on(RealTimeEvents.Error, _on_error)
        transcriber.on(RealTimeEvents.Termination, _on_termination)

        params = RealTimeParameters(
            sample_rate=_SAMPLE_RATE,
            language_codes=[self._cfg.language_code],
        )
        await transcriber.connect(params)

        async def _feed() -> None:
            try:
                async for frame in audio:
                    await transcriber.stream(frame.pcm)
            finally:
                await transcriber.disconnect(terminate=True)
                # If Termination event never fires (e.g. network drop), unblock
                # the consumer with a fallback sentinel after a short grace period.
                await asyncio.sleep(2.0)
                queue.put_nowait(None)

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

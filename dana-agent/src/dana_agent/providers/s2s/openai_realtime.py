"""Reference speech-to-speech provider — OpenAI Realtime API.

This is the "real-ish" S2S provider that proves the cascade-or-S2S abstraction
end to end: it is selected purely by config (``conversation.mode = "s2s"`` +
``conversation.s2s.provider = "openai_realtime"``), constructed by the same
registry/factory as every other provider, and validated at startup like any
other stack.

It follows the project's import-guard rule (see ``providers/tts/gemini.py``): the
module top-level imports only stdlib + dana types, so importing it offline never
fails. The OpenAI SDK is imported lazily inside :meth:`stream`, which opens a
Realtime WebSocket session — so the streaming path is **live-only** (it needs a
real API key and network and is exercised under the ``live`` pytest marker, never
in the offline suite). Construction and capability validation are fully offline.

Because it is a cloud provider it does **not** advertise
``Capabilities.self_hosted`` — so ``enforce_local_only`` correctly refuses to
build it under ``privacy.local_only``.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from dana_agent.config import S2SConfig
from dana_agent.providers.base import Capabilities, S2SProvider
from dana_agent.providers.registry import register
from dana_agent.types import AudioChunk, AudioFrame

# OpenAI Realtime speaks 24 kHz mono PCM16 in and out.
_SAMPLE_RATE = 24_000
_DEFAULT_MODEL = "gpt-4o-realtime-preview"


@register("s2s", "openai_realtime")
class OpenAIRealtimeS2S(S2SProvider):
    """Single-model speech-to-speech via the OpenAI Realtime API.

    Requires the ``openai`` package (``pip install openai`` / the ``cloud``
    extra) and ``OPENAI_API_KEY`` (or ``DANA_CONVERSATION__S2S__API_KEY``).
    """

    def __init__(self, cfg: S2SConfig) -> None:
        self._cfg = cfg
        self._client: Any = None  # lazy — never constructed offline

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                from openai import AsyncOpenAI  # noqa: PLC0415
            except ImportError as exc:
                raise ImportError(
                    "openai is required for the openai_realtime S2S provider: "
                    "install dana-agent[cloud] or pip install openai"
                ) from exc
            api_key = self._cfg.api_key.get_secret_value() if self._cfg.api_key else None
            self._client = AsyncOpenAI(api_key=api_key, base_url=self._cfg.base_url or None)
        return self._client

    async def stream(self, audio: AsyncIterator[AudioFrame]) -> AsyncIterator[AudioChunk]:
        """Stream input audio to the Realtime session and yield response audio.

        Live-only: opens a Realtime WebSocket, appends each input frame to the
        input buffer, and yields ``response.audio.delta`` chunks as PCM16. The
        offline suite never reaches this (no key/network); it is covered by the
        ``live`` provider tests.
        """
        client = self._get_client()
        model = self._cfg.model if self._cfg.model != "mock-s2s" else _DEFAULT_MODEL
        async with client.beta.realtime.connect(model=model) as conn:  # pragma: no cover - live
            await conn.session.update(
                session={
                    "modalities": ["audio", "text"],
                    "voice": self._cfg.voice or "alloy",
                    "input_audio_format": "pcm16",
                    "output_audio_format": "pcm16",
                }
            )
            async for frame in audio:
                await conn.input_audio_buffer.append(audio=frame.pcm)
                async for event in conn:
                    if event.type == "response.audio.delta":
                        yield AudioChunk(pcm=event.delta, sample_rate=_SAMPLE_RATE)
                    elif event.type == "response.done":
                        break

    @property
    def capabilities(self) -> Capabilities:
        return Capabilities(
            streaming=True,
            languages=frozenset({"en", "vi"}),
            self_hosted=False,  # cloud → refused under local_only
            latency_class="realtime",
        )

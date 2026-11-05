"""OpenAI TTS — low-latency cloud speech (``gpt-4o-mini-tts`` by default).

Uses ``client.audio.speech.with_streaming_response.create(response_format="pcm")``
which returns raw 24 kHz mono int16 PCM, iterated in 4800-byte (100 ms) pieces
and yielded as they arrive, so first audio is the server's first byte. Because
network chunks need not be sample-aligned, a dangling odd byte is carried into
the next chunk. ``base_url`` makes this work for any OpenAI-compatible TTS
server (e.g. a local Kokoro/Piper gateway). Requires the ``openai`` package.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from dana_agent.config import TTSConfig
from dana_agent.observability import get_logger
from dana_agent.providers.base import Capabilities, TTSProvider
from dana_agent.providers.registry import register
from dana_agent.types import AudioChunk

log = get_logger(__name__)

_SAMPLE_RATE = 24_000
_CHUNK_BYTES = 4_800  # 100 ms of 24 kHz int16
_DEFAULT_MODEL = "gpt-4o-mini-tts"
_DEFAULT_VOICE = "alloy"


@register("tts", "openai")
class OpenAITTS(TTSProvider):
    def __init__(self, cfg: TTSConfig) -> None:
        self._cfg = cfg
        self._client: Any = None

    @property
    def capabilities(self) -> Capabilities:
        return Capabilities(
            streaming=True,
            languages=frozenset({"en", "vi"}),
            latency_class="fast",
        )

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                import openai  # noqa: PLC0415
            except ImportError as exc:
                raise ImportError(
                    "openai is required for the openai TTS provider: "
                    "install dana-agent[cloud] or pip install openai"
                ) from exc
            api_key = self._cfg.api_key.get_secret_value() if self._cfg.api_key else None
            self._client = openai.AsyncOpenAI(api_key=api_key, base_url=self._cfg.base_url)
        return self._client

    async def warmup(self) -> None:
        """Create the client so the first sentence pays no import/TLS-setup cost."""
        self._get_client()

    async def stream(self, text: AsyncIterator[str]) -> AsyncIterator[AudioChunk]:
        client = self._get_client()
        model = self._cfg.model if self._cfg.model and "mock" not in self._cfg.model else None
        voice = self._cfg.voice if self._cfg.voice and self._cfg.voice != "default" else None
        async for sentence in text:
            sentence = sentence.strip()
            if not sentence:
                continue
            carry = b""
            async with client.audio.speech.with_streaming_response.create(
                model=model or _DEFAULT_MODEL,
                voice=voice or _DEFAULT_VOICE,
                input=sentence,
                response_format="pcm",
            ) as response:
                async for data in response.iter_bytes(chunk_size=_CHUNK_BYTES):
                    buf = carry + data
                    if len(buf) % 2:
                        buf, carry = buf[:-1], buf[-1:]
                    else:
                        carry = b""
                    if buf:
                        yield AudioChunk(pcm=buf, sample_rate=_SAMPLE_RATE)

    async def aclose(self) -> None:
        client, self._client = self._client, None
        if client is not None and hasattr(client, "close"):
            await client.close()

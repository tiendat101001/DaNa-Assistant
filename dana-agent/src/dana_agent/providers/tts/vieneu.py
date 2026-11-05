"""VieNeu TTS — text-to-speech via self-hosted VieNeu-TTS engine.

Supports real-time audio streaming from a VieNeu server (e.g. running on port 7860
or 8001 via ``uv run vieneu-web`` or ``uv run vieneu-stream``).

Streams 48 kHz / 24 kHz 16-bit PCM AudioChunks over HTTP (/stream).
"""

from __future__ import annotations

import struct
import sys
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from dana_agent.config import TTSConfig
from dana_agent.providers.base import Capabilities, TTSProvider
from dana_agent.providers.registry import register
from dana_agent.types import AudioChunk

_DEFAULT_BASE_URL = "http://localhost:7860"
_DEFAULT_SAMPLE_RATE = 24_000
_EXT_SRC_PATH = str(Path(__file__).resolve().parents[4] / "dana-extensions" / "VieNeu-TTS" / "src")



@register("tts", "vieneu")
class VieNeuTTS(TTSProvider):
    """Streaming TTS via self-hosted VieNeu-TTS server.

    Connects to `{base_url}/stream?text={sentence}&voice_id={voice}` via HTTP streaming.
    Supports both Vietnamese and English (`vi`, `en`).
    """

    def __init__(self, cfg: TTSConfig) -> None:
        self._cfg = cfg
        self._base_url = (cfg.base_url or _DEFAULT_BASE_URL).rstrip("/")
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                import httpx  # noqa: PLC0415
            except ImportError as exc:
                raise ImportError(
                    "httpx is required for the vieneu TTS provider: "
                    "pip install httpx"
                ) from exc
            self._client = httpx.AsyncClient(timeout=30.0)
        return self._client

    @property
    def capabilities(self) -> Capabilities:
        return Capabilities(
            streaming=True,
            languages=frozenset({"en", "vi"}),
            self_hosted=True,
            latency_class="fast",
        )

    async def stream(self, text: AsyncIterator[str]) -> AsyncIterator[AudioChunk]:
        client = self._get_client()
        voice_id = self._cfg.voice if self._cfg.voice and self._cfg.voice != "default" else None

        async for sentence in text:
            sentence = sentence.strip()
            if not sentence:
                continue

            params = {"text": sentence}
            if voice_id:
                params["voice_id"] = voice_id

            url = f"{self._base_url}/stream"
            chunks_emitted = 0

            try:
                async with client.stream("GET", url, params=params) as response:
                    if response.status_code == 200:
                        async for chunk in _stream_audio_response(response):
                            yield chunk
                            chunks_emitted += 1
                    else:
                        json_payload = {"text": sentence}
                        if voice_id:
                            json_payload["voice_id"] = voice_id
                        async with client.stream("POST", url, json=json_payload) as post_resp:
                            post_resp.raise_for_status()
                            async for chunk in _stream_audio_response(post_resp):
                                yield chunk
                                chunks_emitted += 1
            except Exception:
                if chunks_emitted > 0:
                    # Stream finished after sending chunks (e.g. socket closed at EOF)
                    continue
                # If remote server returns error or is unreachable on 0 chunks,
                # attempt local fallback
                async for chunk in self._fallback_local_stream(sentence, voice_id):

                    yield chunk

    async def _fallback_local_stream(
        self, text: str, voice_id: str | None
    ) -> AsyncIterator[AudioChunk]:
        """Fallback to in-process vieneu SDK if installed locally or in dana-extensions."""
        try:
            if _EXT_SRC_PATH not in sys.path:
                sys.path.insert(0, _EXT_SRC_PATH)

            import numpy as np  # noqa: PLC0415
            from vieneu import Vieneu  # noqa: PLC0415

            engine = Vieneu(mode="standard", backbone_device="cpu")

            sr = getattr(engine, "sample_rate", _DEFAULT_SAMPLE_RATE)
            for f32_chunk in engine.infer_stream(text, voice=voice_id):
                if f32_chunk is None or len(f32_chunk) == 0:
                    continue
                pcm16 = (np.clip(f32_chunk, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
                yield AudioChunk(pcm=pcm16, sample_rate=sr)
        except Exception as exc:
            msg = (
                f"Failed to stream from VieNeu server at {self._base_url} "
                f"and local vieneu package fallback failed: {exc}"
            )
            raise RuntimeError(msg) from exc

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None


async def _stream_audio_response(response: Any) -> AsyncIterator[AudioChunk]:
    """Parse streaming HTTP response (WAV header + PCM16 data) into AudioChunks."""
    header_parsed = False
    sample_rate = _DEFAULT_SAMPLE_RATE

    if hasattr(response, "headers") and response.headers is not None:
        x_sr = response.headers.get("x-sample-rate") or response.headers.get("X-Sample-Rate")
        if x_sr and str(x_sr).isdigit():
            sample_rate = int(x_sr)

    buffer = bytearray()


    try:
        async for raw_bytes in response.aiter_bytes():
            if not raw_bytes:
                continue

            if not header_parsed:
                buffer.extend(raw_bytes)
                if len(buffer) >= 44:
                    # Check for RIFF header
                    if buffer[:4] == b"RIFF" and buffer[8:12] == b"WAVE":
                        # Bytes 24..28 contain sample rate (little-endian uint32)
                        sr = struct.unpack("<I", buffer[24:28])[0]
                        if sr in (16000, 22050, 24000, 32000, 44100, 48000):
                            sample_rate = sr
                        pcm_data = bytes(buffer[44:])
                    else:
                        pcm_data = bytes(buffer)
                    header_parsed = True
                    buffer.clear()
                    if pcm_data:
                        yield AudioChunk(pcm=pcm_data, sample_rate=sample_rate)
            else:
                yield AudioChunk(pcm=raw_bytes, sample_rate=sample_rate)

        if buffer and not header_parsed:
            yield AudioChunk(pcm=bytes(buffer), sample_rate=sample_rate)

    except Exception:
        # End of stream / closed connection after reading audio is expected on HTTP EOF
        if header_parsed:
            return
        raise

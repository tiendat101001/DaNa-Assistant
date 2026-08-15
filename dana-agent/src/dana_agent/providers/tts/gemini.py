"""Gemini TTS — text-to-speech via Google Gemini's audio generation model.

Wraps ``google-genai`` (the unified Google AI SDK) around the
``gemini-2.5-flash-preview-tts`` model. Gemini TTS synthesises a full sentence
in one API round-trip and returns Linear16 PCM at 24 kHz; we yield one
AudioChunk per input text segment so the rest of the pipeline can begin
playback while the brain is still generating later sentences.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

from dana_agent.config import TTSConfig
from dana_agent.providers.base import Capabilities, TTSProvider
from dana_agent.providers.registry import register
from dana_agent.types import AudioChunk

# Gemini TTS always outputs 24 kHz Linear16 PCM.
_SAMPLE_RATE = 24_000
_DEFAULT_VOICE = "Aoede"


@register("tts", "gemini")
class GeminiTTS(TTSProvider):
    """Sentence-by-sentence TTS via the Gemini audio generation model.

    Requires the ``google-genai`` package (``pip install google-genai``).
    API key sourced from ``DANA_CONVERSATION__TTS__API_KEY``.
    Voice controlled by ``DANA_CONVERSATION__TTS__VOICE`` (Gemini prebuilt voice
    name; default ``Aoede``).  Both ``en`` and ``vi`` are supported.
    """

    def __init__(self, cfg: TTSConfig) -> None:
        self._cfg = cfg
        self._client: Any = None  # lazy-initialised on first use

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                from google import genai  # noqa: PLC0415
            except ImportError as exc:
                raise ImportError(
                    "google-genai is required for the gemini TTS provider: "
                    "install dana-agent[cloud] or pip install google-genai"
                ) from exc
            api_key = self._cfg.api_key.get_secret_value() if self._cfg.api_key else None
            self._client = genai.Client(api_key=api_key)
        return self._client

    @property
    def capabilities(self) -> Capabilities:
        return Capabilities(
            streaming=True,
            languages=frozenset({"en", "vi"}),
            latency_class="fast",
        )

    async def stream(self, text: AsyncIterator[str]) -> AsyncIterator[AudioChunk]:
        from google.genai import types  # noqa: PLC0415

        client = self._get_client()
        model = self._cfg.model or "gemini-2.5-flash-preview-tts"
        voice = self._cfg.voice or _DEFAULT_VOICE

        speech_cfg = types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice)
            )
        )
        gen_cfg = types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=speech_cfg,
        )

        async for sentence in text:
            sentence = sentence.strip()
            if not sentence:
                continue
            # Gemini TTS is a synchronous call; run in a thread so we don't
            # block the event loop while the model synthesises audio.
            response = await asyncio.to_thread(
                client.models.generate_content,
                model=model,
                contents=sentence,
                config=gen_cfg,
            )
            pcm = _extract_audio(response)
            if pcm:
                yield AudioChunk(pcm=pcm, sample_rate=_SAMPLE_RATE)

    async def aclose(self) -> None:
        self._client = None


def _extract_audio(response: object) -> bytes | None:
    """Pull raw PCM bytes out of a Gemini generate_content response."""
    try:
        candidates = getattr(response, "candidates", None)
        if not candidates:
            return None
        parts = candidates[0].content.parts
        if not parts:
            return None
        inline = parts[0].inline_data
        if inline is None:
            return None
        data = inline.data
        if data is None:
            return None
        # Older SDK versions return base64-encoded strings; newer return bytes.
        if isinstance(data, (bytes, bytearray)):
            return bytes(data)
        import base64  # noqa: PLC0415

        return base64.b64decode(data)
    except (AttributeError, IndexError):
        return None

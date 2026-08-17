"""Gemini TTS — text-to-speech via Google Gemini's audio generation model.

Wraps ``google-genai`` (the unified Google AI SDK) around the
``gemini-2.5-flash-preview-tts`` model via the async streaming API
(``client.aio.models.generate_content_stream``).

The server streams audio back as a sequence of ``GenerateContentResponse``
chunks, each carrying a portion of the PCM data in
``candidates[0].content.parts[0].inline_data.data``.  We yield each non-empty
chunk immediately so the output transport can start playing before the model
has finished synthesising the full sentence.

Sentence-level pipelining is handled upstream by ``ProviderTTSProcessor``,
which flushes sentences to this provider at punctuation boundaries — TTS for
the first sentence starts while the LLM is still generating the rest of the
reply.
"""

from __future__ import annotations

import base64
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
    """Streaming TTS via the Gemini audio generation model.

    Requires the ``google-genai`` package (``pip install google-genai``).
    API key sourced from ``GEMINI_API_KEY`` (or ``GOOGLE_API_KEY``, or
    ``DANA_CONVERSATION__TTS__API_KEY`` for an explicit override).
    Voice controlled by ``DANA_CONVERSATION__TTS__VOICE`` (Gemini prebuilt
    voice name; default ``Aoede``).  Both ``en`` and ``vi`` are supported.
    """

    # Explicit language label for the TTS prompt so Gemini uses the right accent.
    _LANG_LABEL: dict[str, str] = {
        "en": "English",
        "vi": "Vietnamese",
    }

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

        gen_cfg = types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice)
                )
            ),
        )

        lang_code = self._cfg.language_code
        lang_label = self._LANG_LABEL.get(lang_code, lang_code)
        # "Say in <Language>: <text>" — explicit language keeps the right accent
        # and prevents Gemini from generating a text reply instead of audio.
        tts_prefix = f"Say in {lang_label}"

        async for sentence in text:
            sentence = sentence.strip()
            if not sentence:
                continue
            response_stream = await client.aio.models.generate_content_stream(
                model=model,
                contents=f"{tts_prefix}: {sentence}",
                config=gen_cfg,
            )
            async for chunk in response_stream:
                pcm = _extract_audio_chunk(chunk)
                if pcm:
                    yield AudioChunk(pcm=pcm, sample_rate=_SAMPLE_RATE)

    async def aclose(self) -> None:
        self._client = None


def _extract_audio_chunk(response: object) -> bytes | None:
    """Pull raw PCM bytes out of one streaming GenerateContentResponse chunk."""
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
        if isinstance(data, (bytes, bytearray)):
            return bytes(data)
        return base64.b64decode(data)
    except (AttributeError, IndexError):
        return None

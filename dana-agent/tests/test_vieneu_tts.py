"""Unit tests for the VieNeu TTS provider in dana-agent."""

from __future__ import annotations

import io
import wave

import pytest

from dana_agent.config import TTSConfig
from dana_agent.providers.base import Capabilities, TTSProvider
from dana_agent.providers.registry import build_tts, get_class, known
from dana_agent.providers.tts.vieneu import VieNeuTTS
from tests.conftest import async_iter, collect


def test_vieneu_provider_registered() -> None:
    assert "vieneu" in known("tts")
    cls = get_class("tts", "vieneu")
    assert cls is VieNeuTTS


def test_vieneu_capabilities() -> None:
    provider = build_tts(TTSConfig(provider="vieneu", base_url="http://localhost:7860"))
    assert isinstance(provider, TTSProvider)
    caps = provider.capabilities
    assert isinstance(caps, Capabilities)
    assert caps.streaming is True
    assert caps.self_hosted is True
    assert "vi" in caps.languages
    assert "en" in caps.languages


@pytest.mark.asyncio
async def test_vieneu_stream_parses_wav_header_and_pcm() -> None:
    # Build a mock WAV header with 48000 Hz sample rate + 20ms of audio
    sample_rate = 48000
    pcm16_data = b"\x01\x00\x02\x00" * 480  # 480 samples = 10ms at 48kHz mono 16-bit
    h = io.BytesIO()
    with wave.open(h, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm16_data)
    wav_bytes = h.getvalue()
    assert wav_bytes[:4] == b"RIFF"

    cfg = TTSConfig(provider="vieneu", base_url="http://localhost:7860", voice="Adam")
    provider = VieNeuTTS(cfg)

    class MockStreamResponse:
        status_code = 200

        async def aiter_bytes(self):
            # Yield in 2 chunks: first chunk includes WAV header (44 bytes),
            # second contains remaining PCM
            yield wav_bytes[:60]
            yield wav_bytes[60:]

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

    class MockClient:
        def stream(self, method, url, params=None, json=None):
            return MockStreamResponse()

        async def aclose(self):
            pass

    provider._client = MockClient()

    chunks = await collect(provider.stream(async_iter(["Xin chào cả nhà."])))
    assert len(chunks) > 0
    total_pcm = b"".join(c.pcm for c in chunks)
    assert total_pcm == pcm16_data
    assert chunks[0].sample_rate == 48000

    await provider.aclose()


@pytest.mark.asyncio
async def test_vieneu_stream_skips_empty_sentences() -> None:
    provider = VieNeuTTS(TTSConfig(provider="vieneu", base_url="http://localhost:7860"))

    class MockClient:
        def stream(self, method, url, params=None, json=None):
            raise RuntimeError("Should not be called for empty strings")

        async def aclose(self):
            pass

    provider._client = MockClient()
    chunks = await collect(provider.stream(async_iter(["", "   ", "\n"])))
    assert len(chunks) == 0
    await provider.aclose()

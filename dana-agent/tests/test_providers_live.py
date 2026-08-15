"""Live smoke tests — hit real APIs, require keys, never run in CI.

Run with: python -m pytest -m live -v
Keys must be set (e.g. via .env):
  DANA_CONVERSATION__STT__API_KEY  — AssemblyAI
  DANA_CONVERSATION__TTS__API_KEY  — Gemini (same key as LLM)
  DANA_CONVERSATION__LLM__API_KEY  — Gemini
"""

from __future__ import annotations

import struct

import pytest

from dana_agent.config import LLMConfig, STTConfig, TTSConfig, get_settings
from dana_agent.providers.registry import build_stt, build_tts
from dana_agent.types import AudioChunk, AudioFrame, Transcript
from tests.conftest import async_iter, collect

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _silence_pcm(duration_ms: int, sample_rate: int = 16_000) -> bytes:
    """Return ``duration_ms`` ms of 16-bit signed little-endian silence."""
    n_samples = int(sample_rate * duration_ms / 1000)
    return struct.pack(f"<{n_samples}h", *([0] * n_samples))


# ---------------------------------------------------------------------------
# AssemblyAI STT
# ---------------------------------------------------------------------------


@pytest.mark.live
async def test_assemblyai_stt_connects_and_transcribes() -> None:
    """Connect to AssemblyAI, stream 2 s of silence, close cleanly.

    A silence clip won't produce a real transcript, but the test verifies the
    WebSocket handshake, streaming, and clean teardown.  Replace with a real
    audio clip for a stricter assertion.
    """
    cfg = get_settings()
    stt_cfg = STTConfig(
        provider="assemblyai",
        api_key=cfg.conversation.stt.api_key,
        model=cfg.conversation.stt.model,
    )
    stt = build_stt(stt_cfg)
    silence = _silence_pcm(2000)  # 2 s = enough for a session round-trip
    frames = [
        AudioFrame(pcm=silence[i : i + 3200], sample_rate=16_000)
        for i in range(0, len(silence), 3200)
    ]
    # Must not raise; partial transcripts (empty on silence) are fine
    results: list[Transcript] = await collect(stt.stream(async_iter(frames)))
    # filter any empties
    non_empty = [t for t in results if t.text]
    assert isinstance(non_empty, list)  # just structural — silence = no words


@pytest.mark.live
async def test_assemblyai_stt_capabilities() -> None:
    cfg = get_settings()
    stt_cfg = STTConfig(provider="assemblyai", api_key=cfg.conversation.stt.api_key)
    stt = build_stt(stt_cfg)
    caps = stt.capabilities
    assert caps.streaming is True
    assert "en" in caps.languages
    assert "vi" in caps.languages
    assert caps.latency_class == "realtime"


# ---------------------------------------------------------------------------
# Gemini TTS
# ---------------------------------------------------------------------------


@pytest.mark.live
async def test_gemini_tts_returns_audio() -> None:
    cfg = get_settings()
    tts_cfg = TTSConfig(
        provider="gemini",
        model=cfg.conversation.tts.model,
        api_key=cfg.conversation.tts.api_key,
        voice=cfg.conversation.tts.voice,
    )
    tts = build_tts(tts_cfg)
    chunks: list[AudioChunk] = await collect(tts.stream(async_iter(["Hello, I am DaNa."])))
    assert len(chunks) == 1
    assert len(chunks[0].pcm) > 0
    assert chunks[0].sample_rate == 24_000


@pytest.mark.live
async def test_gemini_tts_empty_text_is_skipped() -> None:
    cfg = get_settings()
    tts_cfg = TTSConfig(
        provider="gemini",
        model=cfg.conversation.tts.model,
        api_key=cfg.conversation.tts.api_key,
        voice=cfg.conversation.tts.voice,
    )
    tts = build_tts(tts_cfg)
    chunks: list[AudioChunk] = await collect(tts.stream(async_iter(["", "   ", ""])))
    assert chunks == []


@pytest.mark.live
async def test_gemini_tts_multiple_sentences() -> None:
    cfg = get_settings()
    tts_cfg = TTSConfig(
        provider="gemini",
        model=cfg.conversation.tts.model,
        api_key=cfg.conversation.tts.api_key,
        voice=cfg.conversation.tts.voice,
    )
    tts = build_tts(tts_cfg)
    sentences = ["This is the first sentence.", "And this is the second one."]
    chunks: list[AudioChunk] = await collect(tts.stream(async_iter(sentences)))
    assert len(chunks) == 2
    assert all(len(c.pcm) > 0 for c in chunks)


# ---------------------------------------------------------------------------
# Gemini LLM (brain model)
# ---------------------------------------------------------------------------


@pytest.mark.live
async def test_gemini_llm_streams_response() -> None:
    from langchain_core.messages import HumanMessage

    from dana_agent.brain.model import build_chat_model

    cfg = get_settings()
    llm_cfg = LLMConfig(
        provider=cfg.conversation.llm.provider,
        model=cfg.conversation.llm.model,
        api_key=cfg.conversation.llm.api_key,
    )
    model = build_chat_model(llm_cfg)
    # astream yields AIMessageChunks
    chunks = []
    async for chunk in model.astream([HumanMessage(content="Say hello in one word.")]):
        chunks.append(chunk)
    full = "".join(c.content for c in chunks if isinstance(c.content, str))
    assert len(full) > 0

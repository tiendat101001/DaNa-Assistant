"""Instant-ack tests (offline)."""

from __future__ import annotations

import struct
from collections.abc import AsyncIterator
from typing import Any

from pipecat.frames.frames import (
    InputAudioRawFrame,
    TTSAudioRawFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from structlog.testing import capture_logs

from dana_agent.config import LatencyConfig
from dana_agent.conversation.ack import AckAudioRawFrame, AckCache, AckProcessor
from dana_agent.conversation.latency import TurnClock
from dana_agent.types import AudioChunk

D = FrameDirection.DOWNSTREAM


def loud(ms: int = 400, amp: int = 3000) -> InputAudioRawFrame:
    n = 16 * ms
    return InputAudioRawFrame(
        audio=struct.pack(f"<{n}h", *([amp] * n)), num_channels=1, sample_rate=16_000
    )


def quiet(ms: int = 400) -> InputAudioRawFrame:
    return loud(ms, amp=2)


class FakeTTS:
    def __init__(self, fail: set[str] | None = None) -> None:
        self.texts: list[str] = []
        self.fail = fail or set()

    async def stream(self, text: AsyncIterator[str]) -> AsyncIterator[AudioChunk]:
        async for t in text:
            self.texts.append(t)
            if t in self.fail:
                raise RuntimeError("tts down")
            yield AudioChunk(pcm=b"\x01\x00" * 480, sample_rate=24_000)


async def run(proc: Any, frames: list[Any]) -> list[Any]:
    out: list[Any] = []

    async def cap(frame: Any, direction: Any = D) -> None:
        out.append(frame)

    proc.push_frame = cap
    for f in frames:
        await proc.process_frame(f, D)
    return out


async def _cache() -> AckCache:
    c = AckCache()
    await c.fill(FakeTTS(), {"en": "Mm-hm.", "vi": "Dạ."})  # type: ignore[arg-type]
    return c


def test_config_defaults_off() -> None:
    lat = LatencyConfig()
    assert lat.ack_enabled is False
    assert lat.ack_phrases == {"en": "Mm-hm.", "vi": "Dạ."}


async def test_cache_fill_per_language_and_failure_isolated() -> None:
    tts = FakeTTS(fail={"Dạ."})
    c = AckCache()
    await c.fill(tts, {"en": "Mm-hm.", "vi": "Dạ."})  # type: ignore[arg-type]
    assert tts.texts == ["Mm-hm.", "Dạ."]
    assert c.get("en") and c.get("vi") is None


async def test_ack_pushed_after_vad_stop_before_reply() -> None:
    proc = AckProcessor(await _cache(), language="vi")
    seq = [VADUserStartedSpeakingFrame(), loud(), VADUserStoppedSpeakingFrame()]
    out = await run(proc, seq)
    kinds = [type(f).__name__ for f in out]
    assert kinds[-2:] == ["VADUserStoppedSpeakingFrame", "AckAudioRawFrame"]
    # a later real reply frame comes after the ack in the stream
    out += await run(proc, [TTSAudioRawFrame(audio=b"\0\0", sample_rate=24_000, num_channels=1)])
    assert [type(f).__name__ for f in out][-2:] == ["AckAudioRawFrame", "TTSAudioRawFrame"]


async def test_ack_language_falls_back_to_en() -> None:
    c = AckCache()
    c.put("en", [(b"\x01\x00" * 10, 24_000)])
    out = await run(
        AckProcessor(c, language="vi"),
        [VADUserStartedSpeakingFrame(), loud(), VADUserStoppedSpeakingFrame()],
    )
    assert any(isinstance(f, AckAudioRawFrame) for f in out)


async def test_no_ack_on_cache_miss() -> None:
    out = await run(
        AckProcessor(AckCache()),
        [VADUserStartedSpeakingFrame(), loud(), VADUserStoppedSpeakingFrame()],
    )
    assert not any(isinstance(f, AckAudioRawFrame) for f in out)
    assert any(isinstance(f, VADUserStoppedSpeakingFrame) for f in out)


async def test_no_ack_for_empty_or_quiet_or_typed() -> None:
    c = await _cache()
    # too short
    out = await run(
        AckProcessor(c), [VADUserStartedSpeakingFrame(), loud(50), VADUserStoppedSpeakingFrame()]
    )
    assert not any(isinstance(f, AckAudioRawFrame) for f in out)
    # silence (energy-gated empty)
    out = await run(
        AckProcessor(c), [VADUserStartedSpeakingFrame(), quiet(), VADUserStoppedSpeakingFrame()]
    )
    assert not any(isinstance(f, AckAudioRawFrame) for f in out)
    # typed turn: no VAD frames at all
    out = await run(AckProcessor(c), [loud()])
    assert not any(isinstance(f, AckAudioRawFrame) for f in out)
    # stop without a start
    out = await run(AckProcessor(c), [VADUserStoppedSpeakingFrame()])
    assert not any(isinstance(f, AckAudioRawFrame) for f in out)


def test_clock_ack_does_not_change_headline() -> None:
    t = [0.0]
    clock = TurnClock(600, clock=lambda: t[0], vad_delay_ms=200)
    with capture_logs() as logs:
        clock.mark_end_of_speech()
        t[0] = 0.02
        clock.mark_ack_first_audio()
        clock.mark_ack_first_audio()
        t[0] = 0.10
        clock.mark_stt_final()
        t[0] = 0.30
        clock.mark_tts_first_audio()
    lat = {e["stage"]: e["ms"] for e in logs if e["event"] == "latency"}
    assert lat["ack_first_audio"] == 220
    assert lat["voice_to_first_audio"] == 500
    assert len([e for e in logs if e.get("stage") == "ack_first_audio"]) == 1

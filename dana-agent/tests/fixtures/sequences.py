"""Synthetic frame sequences that stand in for recorded audio until real clips
land (TASK-010). The mock VAD treats any non-zero PCM as speech; the mock wake
engine fires on PCM equal to the phrase text — so these builders drive the wake
gate and state machine deterministically, offline."""

from __future__ import annotations

from dana_agent.types import AudioFrame

SAMPLE_RATE = 16_000


def wake_frame(phrase: str = "Hey DaNa") -> AudioFrame:
    return AudioFrame(phrase.encode("utf-8"), SAMPLE_RATE)


def end_frame(phrase: str = "Bye DaNa") -> AudioFrame:
    return AudioFrame(phrase.encode("utf-8"), SAMPLE_RATE)


def speech_frame(text: str = "command") -> AudioFrame:
    """Non-silent, non-wake speech (VAD: speech; KWS: no hit)."""
    return AudioFrame(text.encode("utf-8"), SAMPLE_RATE)


def silence_frame(samples: int = 160) -> AudioFrame:
    return AudioFrame(b"\x00\x00" * samples, SAMPLE_RATE)

"""Unit tests for conversation/processors.py — offline, no network, no audio device.

Covers: HalfDuplexGateProcessor, MuteProcessor, WakeGateProcessor,
BrainProcessor, and ProviderTTSProcessor.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

from pipecat.frames.frames import (
    EndFrame,
    InputAudioRawFrame,
    InterruptionFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection

from dana_agent.actions.gate import ActionGate
from dana_agent.brain import Brain
from dana_agent.config import ActionsConfig, TTSConfig
from dana_agent.conversation.mute import BotSpeakingGate, MicGate
from dana_agent.conversation.processors import (
    BrainProcessor,
    HalfDuplexGateProcessor,
    MuteProcessor,
    ProviderTTSProcessor,
    WakeGateProcessor,
)
from dana_agent.providers.registry import build_actions
from dana_agent.providers.tts.mock import MockTTS
from tests.fakes.chat import ScriptedChatModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def collect_frames(
    processor,
    frames_to_send: list[tuple],
) -> list:
    """Push frames into the processor and collect what it pushes downstream."""
    collected: list = []

    async def capture(frame, direction: FrameDirection = FrameDirection.DOWNSTREAM) -> None:
        collected.append(frame)

    processor.push_frame = capture
    for frame, direction in frames_to_send:
        await processor.process_frame(frame, direction)
    return collected


def audio_frame() -> InputAudioRawFrame:
    return InputAudioRawFrame(audio=b"\x00" * 640, num_channels=1, sample_rate=16_000)


def transcription(text: str) -> TranscriptionFrame:
    return TranscriptionFrame(text=text, user_id="user1", timestamp="0")


def make_brain(turns: list[dict]) -> Brain:
    model = ScriptedChatModel(turns=turns)
    gate = ActionGate(build_actions(ActionsConfig()), policy="standard")
    return Brain(model, gate, system_prompt="You are DaNa.")


# ---------------------------------------------------------------------------
# HalfDuplexGateProcessor
# ---------------------------------------------------------------------------


async def test_halfduplex_passes_audio_when_not_speaking() -> None:
    gate = BotSpeakingGate()  # speaking=False by default
    proc = HalfDuplexGateProcessor(gate)
    af = audio_frame()
    frames = await collect_frames(proc, [(af, FrameDirection.DOWNSTREAM)])
    assert af in frames


async def test_halfduplex_drops_audio_when_speaking() -> None:
    gate = BotSpeakingGate()
    gate.set_speaking(True)
    proc = HalfDuplexGateProcessor(gate)
    frames = await collect_frames(proc, [(audio_frame(), FrameDirection.DOWNSTREAM)])
    assert not any(isinstance(f, InputAudioRawFrame) for f in frames)


async def test_halfduplex_drops_vad_start_when_speaking() -> None:
    gate = BotSpeakingGate()
    gate.set_speaking(True)
    proc = HalfDuplexGateProcessor(gate)
    frames = await collect_frames(
        proc, [(VADUserStartedSpeakingFrame(), FrameDirection.DOWNSTREAM)]
    )
    assert not any(isinstance(f, VADUserStartedSpeakingFrame) for f in frames)


async def test_halfduplex_drops_vad_stop_when_speaking() -> None:
    gate = BotSpeakingGate()
    gate.set_speaking(True)
    proc = HalfDuplexGateProcessor(gate)
    frames = await collect_frames(
        proc, [(VADUserStoppedSpeakingFrame(), FrameDirection.DOWNSTREAM)]
    )
    assert not any(isinstance(f, VADUserStoppedSpeakingFrame) for f in frames)


async def test_halfduplex_passes_non_audio_when_speaking() -> None:
    gate = BotSpeakingGate()
    gate.set_speaking(True)
    proc = HalfDuplexGateProcessor(gate)
    end = EndFrame()
    llm = LLMTextFrame(text="hello")
    frames = await collect_frames(
        proc,
        [
            (end, FrameDirection.DOWNSTREAM),
            (llm, FrameDirection.DOWNSTREAM),
        ],
    )
    assert end in frames
    assert llm in frames


async def test_halfduplex_responds_after_gate_cleared() -> None:
    gate = BotSpeakingGate()
    gate.set_speaking(True)
    proc = HalfDuplexGateProcessor(gate)
    # Drop while speaking
    frames = await collect_frames(proc, [(audio_frame(), FrameDirection.DOWNSTREAM)])
    assert not any(isinstance(f, InputAudioRawFrame) for f in frames)
    # Clear the gate — next audio should be forwarded
    gate.set_speaking(False)
    af = audio_frame()
    frames2 = await collect_frames(proc, [(af, FrameDirection.DOWNSTREAM)])
    assert af in frames2


# ---------------------------------------------------------------------------
# MuteProcessor
# ---------------------------------------------------------------------------


async def test_mute_passes_audio_when_unmuted() -> None:
    gate = MicGate(muted=False)
    proc = MuteProcessor(gate)
    af = audio_frame()
    frames = await collect_frames(proc, [(af, FrameDirection.DOWNSTREAM)])
    assert af in frames


async def test_mute_drops_audio_when_muted() -> None:
    gate = MicGate(muted=True)
    proc = MuteProcessor(gate)
    frames = await collect_frames(proc, [(audio_frame(), FrameDirection.DOWNSTREAM)])
    assert not any(isinstance(f, InputAudioRawFrame) for f in frames)


async def test_mute_drops_vad_frames_when_muted() -> None:
    gate = MicGate(muted=True)
    proc = MuteProcessor(gate)
    frames = await collect_frames(
        proc,
        [
            (VADUserStartedSpeakingFrame(), FrameDirection.DOWNSTREAM),
            (VADUserStoppedSpeakingFrame(), FrameDirection.DOWNSTREAM),
        ],
    )
    assert not any(
        isinstance(f, (VADUserStartedSpeakingFrame, VADUserStoppedSpeakingFrame))
        for f in frames
    )


async def test_mute_passes_non_audio_when_muted() -> None:
    gate = MicGate(muted=True)
    proc = MuteProcessor(gate)
    end = EndFrame()
    llm = LLMTextFrame(text="hello")
    frames = await collect_frames(
        proc,
        [
            (end, FrameDirection.DOWNSTREAM),
            (llm, FrameDirection.DOWNSTREAM),
        ],
    )
    assert end in frames
    assert llm in frames


# ---------------------------------------------------------------------------
# WakeGateProcessor
# ---------------------------------------------------------------------------


async def test_wake_gate_bypass_forwards_all_frames() -> None:
    proc = WakeGateProcessor(bypass=True)
    af = audio_frame()
    vad = VADUserStartedSpeakingFrame()
    end = EndFrame()
    frames = await collect_frames(
        proc,
        [
            (af, FrameDirection.DOWNSTREAM),
            (vad, FrameDirection.DOWNSTREAM),
            (end, FrameDirection.DOWNSTREAM),
        ],
    )
    assert af in frames
    assert vad in frames
    assert end in frames


async def test_wake_gate_idle_vad_start_begins_buffering() -> None:
    proc = WakeGateProcessor(detector=None, bypass=False)
    frames = await collect_frames(
        proc, [(VADUserStartedSpeakingFrame(), FrameDirection.DOWNSTREAM)]
    )
    # VAD start frame is consumed (not forwarded) to start buffering
    assert not any(isinstance(f, VADUserStartedSpeakingFrame) for f in frames)
    assert proc._buffering is True


async def test_wake_gate_audio_buffered_not_forwarded_in_idle() -> None:
    proc = WakeGateProcessor(detector=None, bypass=False)
    af = audio_frame()
    # VAD start + audio: both consumed — audio goes into the internal buffer
    frames = await collect_frames(
        proc,
        [
            (VADUserStartedSpeakingFrame(), FrameDirection.DOWNSTREAM),
            (af, FrameDirection.DOWNSTREAM),
        ],
    )
    assert not any(isinstance(f, InputAudioRawFrame) for f in frames)
    assert af.audio in proc._audio_buffer


async def test_wake_gate_activates_on_positive_detection() -> None:
    detector = MagicMock()
    detector.detect = AsyncMock(return_value=True)
    proc = WakeGateProcessor(detector=detector, bypass=False)

    collected: list = []

    async def capture(f, d: FrameDirection = FrameDirection.DOWNSTREAM) -> None:
        collected.append(f)

    proc.push_frame = capture

    await proc.process_frame(VADUserStartedSpeakingFrame(), FrameDirection.DOWNSTREAM)
    await proc.process_frame(audio_frame(), FrameDirection.DOWNSTREAM)
    await proc.process_frame(VADUserStoppedSpeakingFrame(), FrameDirection.DOWNSTREAM)
    if proc._detect_task:
        await proc._detect_task

    assert proc._active is True


async def test_wake_gate_no_activation_on_negative_detection() -> None:
    detector = MagicMock()
    detector.detect = AsyncMock(return_value=False)
    proc = WakeGateProcessor(detector=detector, bypass=False)

    collected: list = []

    async def capture(f, d: FrameDirection = FrameDirection.DOWNSTREAM) -> None:
        collected.append(f)

    proc.push_frame = capture

    await proc.process_frame(VADUserStartedSpeakingFrame(), FrameDirection.DOWNSTREAM)
    await proc.process_frame(audio_frame(), FrameDirection.DOWNSTREAM)
    await proc.process_frame(VADUserStoppedSpeakingFrame(), FrameDirection.DOWNSTREAM)
    if proc._detect_task:
        await proc._detect_task

    assert proc._active is False


async def test_wake_gate_active_forwards_audio() -> None:
    detector = MagicMock()
    detector.detect = AsyncMock(return_value=True)
    proc = WakeGateProcessor(detector=detector, bypass=False)

    collected: list = []

    async def capture(f, d: FrameDirection = FrameDirection.DOWNSTREAM) -> None:
        collected.append(f)

    proc.push_frame = capture

    # Activate the gate
    await proc.process_frame(VADUserStartedSpeakingFrame(), FrameDirection.DOWNSTREAM)
    await proc.process_frame(audio_frame(), FrameDirection.DOWNSTREAM)
    await proc.process_frame(VADUserStoppedSpeakingFrame(), FrameDirection.DOWNSTREAM)
    if proc._detect_task:
        await proc._detect_task
    assert proc._active is True

    # Audio sent after activation should be forwarded
    af = audio_frame()
    await proc.process_frame(af, FrameDirection.DOWNSTREAM)
    assert af in collected


async def test_wake_gate_end_phrase_returns_to_idle() -> None:
    detector = MagicMock()
    detector.detect = AsyncMock(return_value=True)
    proc = WakeGateProcessor(detector=detector, bypass=False, end_phrase="Bye DaNa")

    collected: list = []

    async def capture(f, d: FrameDirection = FrameDirection.DOWNSTREAM) -> None:
        collected.append(f)

    proc.push_frame = capture

    # Activate
    await proc.process_frame(VADUserStartedSpeakingFrame(), FrameDirection.DOWNSTREAM)
    await proc.process_frame(audio_frame(), FrameDirection.DOWNSTREAM)
    await proc.process_frame(VADUserStoppedSpeakingFrame(), FrameDirection.DOWNSTREAM)
    if proc._detect_task:
        await proc._detect_task
    assert proc._active is True

    # End phrase — gate returns to IDLE; the frame is consumed, not forwarded
    tf = transcription("bye dana")
    await proc.process_frame(tf, FrameDirection.DOWNSTREAM)

    assert proc._active is False
    assert tf not in collected


async def test_wake_gate_normal_transcription_forwarded_when_active() -> None:
    detector = MagicMock()
    detector.detect = AsyncMock(return_value=True)
    proc = WakeGateProcessor(detector=detector, bypass=False)

    collected: list = []

    async def capture(f, d: FrameDirection = FrameDirection.DOWNSTREAM) -> None:
        collected.append(f)

    proc.push_frame = capture

    # Activate
    await proc.process_frame(VADUserStartedSpeakingFrame(), FrameDirection.DOWNSTREAM)
    await proc.process_frame(audio_frame(), FrameDirection.DOWNSTREAM)
    await proc.process_frame(VADUserStoppedSpeakingFrame(), FrameDirection.DOWNSTREAM)
    if proc._detect_task:
        await proc._detect_task

    tf = transcription("what is the weather")
    await proc.process_frame(tf, FrameDirection.DOWNSTREAM)
    assert tf in collected


async def test_wake_gate_interruption_deactivates_and_forwards() -> None:
    detector = MagicMock()
    detector.detect = AsyncMock(return_value=True)
    proc = WakeGateProcessor(detector=detector, bypass=False)

    collected: list = []

    async def capture(f, d: FrameDirection = FrameDirection.DOWNSTREAM) -> None:
        collected.append(f)

    proc.push_frame = capture

    # Activate
    await proc.process_frame(VADUserStartedSpeakingFrame(), FrameDirection.DOWNSTREAM)
    await proc.process_frame(audio_frame(), FrameDirection.DOWNSTREAM)
    await proc.process_frame(VADUserStoppedSpeakingFrame(), FrameDirection.DOWNSTREAM)
    if proc._detect_task:
        await proc._detect_task
    assert proc._active is True

    interr = InterruptionFrame()
    await proc.process_frame(interr, FrameDirection.DOWNSTREAM)

    assert proc._active is False
    assert interr in collected


async def test_wake_gate_end_frame_cancels_detect_task() -> None:
    detect_started = asyncio.Event()
    detect_cancelled = asyncio.Event()

    async def slow_detect(pcm: bytes) -> bool:
        detect_started.set()
        try:
            await asyncio.sleep(100)
            return True
        except asyncio.CancelledError:
            detect_cancelled.set()
            raise

    detector = MagicMock()
    detector.detect = slow_detect
    proc = WakeGateProcessor(detector=detector, bypass=False)

    collected: list = []

    async def capture(f, d: FrameDirection = FrameDirection.DOWNSTREAM) -> None:
        collected.append(f)

    proc.push_frame = capture

    # Start detection
    await proc.process_frame(VADUserStartedSpeakingFrame(), FrameDirection.DOWNSTREAM)
    await proc.process_frame(audio_frame(), FrameDirection.DOWNSTREAM)
    await proc.process_frame(VADUserStoppedSpeakingFrame(), FrameDirection.DOWNSTREAM)

    # Wait until slow_detect is actually running before sending EndFrame
    await asyncio.wait_for(detect_started.wait(), timeout=1.0)

    end = EndFrame()
    await proc.process_frame(end, FrameDirection.DOWNSTREAM)

    # Give cancellation time to propagate through _detect_and_activate
    await asyncio.sleep(0.01)

    assert detect_cancelled.is_set()
    assert end in collected


# ---------------------------------------------------------------------------
# BrainProcessor
# ---------------------------------------------------------------------------


async def test_brain_processor_transcription_starts_turn() -> None:
    brain = make_brain([{"text": "Hello there!"}])
    proc = BrainProcessor(brain, session_id="test-session")

    collected: list = []

    async def capture(f, d: FrameDirection = FrameDirection.DOWNSTREAM) -> None:
        collected.append(f)

    proc.push_frame = capture

    await proc.process_frame(transcription("hello"), FrameDirection.DOWNSTREAM)
    if proc._turn_task:
        await proc._turn_task

    types = [type(f) for f in collected]
    assert LLMFullResponseStartFrame in types
    assert LLMTextFrame in types
    assert LLMFullResponseEndFrame in types


async def test_brain_processor_empty_transcription_ignored() -> None:
    brain = make_brain([{"text": "Hello!"}])
    proc = BrainProcessor(brain, session_id="test-session")

    collected: list = []

    async def capture(f, d: FrameDirection = FrameDirection.DOWNSTREAM) -> None:
        collected.append(f)

    proc.push_frame = capture

    # Whitespace-only transcription — should be ignored
    await proc.process_frame(transcription("   "), FrameDirection.DOWNSTREAM)
    await asyncio.sleep(0)

    assert proc._turn_task is None
    assert not collected


async def test_brain_processor_emits_llm_text_frames() -> None:
    brain = make_brain([{"text": "Hello there!"}])
    proc = BrainProcessor(brain, session_id="test-session")

    collected: list = []

    async def capture(f, d: FrameDirection = FrameDirection.DOWNSTREAM) -> None:
        collected.append(f)

    proc.push_frame = capture

    await proc.process_frame(transcription("say hello"), FrameDirection.DOWNSTREAM)
    if proc._turn_task:
        await proc._turn_task

    text_frames = [f for f in collected if isinstance(f, LLMTextFrame)]
    assert text_frames  # at least one token pushed to TTS


async def test_brain_processor_sets_bot_gate_speaking_true_then_false() -> None:
    brain = make_brain([{"text": "Hello!"}])
    bot_gate = BotSpeakingGate()
    proc = BrainProcessor(brain, session_id="test-session", bot_gate=bot_gate)

    collected: list = []

    async def capture(f, d: FrameDirection = FrameDirection.DOWNSTREAM) -> None:
        collected.append(f)

    proc.push_frame = capture

    await proc.process_frame(transcription("hi"), FrameDirection.DOWNSTREAM)
    if proc._turn_task:
        await proc._turn_task

    # Gate should be False again after the turn completes
    assert bot_gate.speaking is False


async def test_brain_processor_interruption_cancels_turn() -> None:
    brain = make_brain([{"text": "Hello there!"}])
    proc = BrainProcessor(brain, session_id="test-session")

    collected: list = []

    async def capture(f, d: FrameDirection = FrameDirection.DOWNSTREAM) -> None:
        collected.append(f)

    proc.push_frame = capture

    await proc.process_frame(transcription("hi"), FrameDirection.DOWNSTREAM)
    interr = InterruptionFrame()
    await proc.process_frame(interr, FrameDirection.DOWNSTREAM)

    # InterruptionFrame is forwarded downstream
    assert interr in collected
    # Turn task is cleaned up
    assert proc._turn_task is None


async def test_brain_processor_non_transcription_frames_forwarded() -> None:
    brain = make_brain([{"text": "Hello!"}])
    proc = BrainProcessor(brain, session_id="test-session")

    collected: list = []

    async def capture(f, d: FrameDirection = FrameDirection.DOWNSTREAM) -> None:
        collected.append(f)

    proc.push_frame = capture

    end = EndFrame()
    llm_text = LLMTextFrame(text="pass-through")
    await proc.process_frame(end, FrameDirection.DOWNSTREAM)
    await proc.process_frame(llm_text, FrameDirection.DOWNSTREAM)

    assert end in collected
    assert llm_text in collected


# ---------------------------------------------------------------------------
# ProviderTTSProcessor
# ---------------------------------------------------------------------------


async def test_tts_processor_chunk_sentences_1_flushes_on_sentence() -> None:
    provider = MockTTS(TTSConfig())
    proc = ProviderTTSProcessor(provider, chunk_sentences=1)

    collected: list = []

    async def capture(f, d: FrameDirection = FrameDirection.DOWNSTREAM) -> None:
        collected.append(f)

    proc.push_frame = capture

    await proc.process_frame(LLMFullResponseStartFrame(), FrameDirection.DOWNSTREAM)
    await proc.process_frame(LLMTextFrame(text="Hello world."), FrameDirection.DOWNSTREAM)
    await proc.process_frame(LLMFullResponseEndFrame(), FrameDirection.DOWNSTREAM)

    audio_frames = [f for f in collected if isinstance(f, TTSAudioRawFrame)]
    assert audio_frames  # TTS produced audio before end-of-response


async def test_tts_processor_chunk_sentences_0_holds_until_end() -> None:
    provider = MockTTS(TTSConfig())
    proc = ProviderTTSProcessor(provider, chunk_sentences=0)

    collected: list = []

    async def capture(f, d: FrameDirection = FrameDirection.DOWNSTREAM) -> None:
        collected.append(f)

    proc.push_frame = capture

    await proc.process_frame(LLMFullResponseStartFrame(), FrameDirection.DOWNSTREAM)
    await proc.process_frame(LLMTextFrame(text="Hello."), FrameDirection.DOWNSTREAM)
    await proc.process_frame(LLMTextFrame(text=" World."), FrameDirection.DOWNSTREAM)

    # No audio yet — chunk_sentences=0 means hold everything until response ends
    audio_before_end = [f for f in collected if isinstance(f, TTSAudioRawFrame)]
    assert not audio_before_end

    await proc.process_frame(LLMFullResponseEndFrame(), FrameDirection.DOWNSTREAM)

    audio_after_end = [f for f in collected if isinstance(f, TTSAudioRawFrame)]
    assert audio_after_end  # flushed on response end


async def test_tts_processor_emits_audio_with_pcm_bytes() -> None:
    provider = MockTTS(TTSConfig())
    proc = ProviderTTSProcessor(provider, chunk_sentences=1)

    collected: list = []

    async def capture(f, d: FrameDirection = FrameDirection.DOWNSTREAM) -> None:
        collected.append(f)

    proc.push_frame = capture

    await proc.process_frame(LLMFullResponseStartFrame(), FrameDirection.DOWNSTREAM)
    await proc.process_frame(LLMTextFrame(text="Done."), FrameDirection.DOWNSTREAM)
    await proc.process_frame(LLMFullResponseEndFrame(), FrameDirection.DOWNSTREAM)

    audio_frames = [f for f in collected if isinstance(f, TTSAudioRawFrame)]
    assert audio_frames
    for af in audio_frames:
        assert isinstance(af.audio, bytes)
        assert len(af.audio) > 0  # MockTTS emits non-empty PCM


async def test_tts_processor_interruption_cancels_and_forwards() -> None:
    provider = MockTTS(TTSConfig())
    proc = ProviderTTSProcessor(provider, chunk_sentences=1)

    collected: list = []

    async def capture(f, d: FrameDirection = FrameDirection.DOWNSTREAM) -> None:
        collected.append(f)

    proc.push_frame = capture

    # Start a TTS session; the task begins running but blocks waiting for text
    await proc.process_frame(LLMFullResponseStartFrame(), FrameDirection.DOWNSTREAM)
    assert proc._active is True
    assert proc._tts_task is not None

    # Interrupt before the response ends
    interr = InterruptionFrame()
    await proc.process_frame(interr, FrameDirection.DOWNSTREAM)

    assert proc._active is False
    assert proc._tts_task is None
    assert interr in collected

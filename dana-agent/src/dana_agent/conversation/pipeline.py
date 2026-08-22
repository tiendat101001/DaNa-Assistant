"""Pipecat pipeline factory for the cascade conversation core (Phase 3).

Assembles the production pipeline::

    LocalAudioInputTransport (mic, 16 kHz, SileroVAD)
        → WakeGateProcessor        (IDLE/ACTIVE gate, wake word)
        → STT service              (AssemblyAI streaming WebSocket)
        → BrainProcessor           (LangGraph → LLMTextFrames)
        → TTS service or processor (ElevenLabs WS | ProviderTTSProcessor)
        → LocalAudioOutputTransport (speakers, 24 kHz)

Speech-to-speech mode assembles a different pipeline (placeholder; wired when
an S2S provider is selected via config).

Usage::

    transport, pipeline, wake_gate = build_cascade_pipeline(cfg, stack)
    worker = PipelineWorker(pipeline, params=PipelineParams(allow_interruptions=True))
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams

from dana_agent.conversation.mute import BotSpeakingGate
from dana_agent.conversation.processors import (
    BrainProcessor,
    HalfDuplexGateProcessor,
    MuteProcessor,
    ProviderTTSProcessor,
    S2SProcessor,
    WakeGateProcessor,
)
from dana_agent.observability import get_logger
from dana_agent.providers.wake.assemblyai import AssemblyAIWakeDetector

if TYPE_CHECKING:
    from dana_agent.app import AssembledStack
    from dana_agent.config import AppConfig
    from dana_agent.conversation.mute import MicGate

log = get_logger(__name__)

# Linear16 PCM at 16 kHz is what AssemblyAI expects.
_STT_SAMPLE_RATE = 16_000
# Google/ElevenLabs TTS output is 24 kHz Linear16.
_TTS_SAMPLE_RATE = 24_000


def _build_stt_service(cfg: AppConfig, stack: AssembledStack) -> FrameProcessor:
    """Return a pipecat STT FrameProcessor for the configured provider."""
    provider = cfg.conversation.stt.provider

    if provider == "assemblyai":
        from pipecat.services.assemblyai.stt import AssemblyAISTTService  # noqa: PLC0415

        api_key = (
            cfg.conversation.stt.api_key.get_secret_value()
            if cfg.conversation.stt.api_key
            else None
        )
        if not api_key:
            raise ValueError(
                "DANA_CONVERSATION__STT__API_KEY is required for the assemblyai STT provider"
            )
        return AssemblyAISTTService(api_key=api_key, sample_rate=_STT_SAMPLE_RATE)

    if provider == "deepgram":
        from pipecat.services.deepgram.stt import DeepgramSTTService  # noqa: PLC0415

        api_key = (
            cfg.conversation.stt.api_key.get_secret_value()
            if cfg.conversation.stt.api_key
            else None
        )
        if not api_key:
            raise ValueError(
                "DANA_CONVERSATION__STT__API_KEY is required for the deepgram STT provider"
            )
        return DeepgramSTTService(api_key=api_key)

    if provider == "mock":
        # In cascade mode with a mock STT, wrap our existing mock provider in a
        # thin FrameProcessor so tests stay fully offline.
        return _MockSTTBridge(stack.stt)

    raise ValueError(
        f"STT provider '{provider}' is not supported in the Pipecat pipeline. "
        "Supported: assemblyai, deepgram, mock."
    )


def _build_tts_service(cfg: AppConfig, stack: AssembledStack) -> FrameProcessor:
    """Return a pipecat TTS FrameProcessor for the configured provider."""
    provider = cfg.conversation.tts.provider

    if provider == "elevenlabs":
        from pipecat.services.elevenlabs.tts import ElevenLabsTTSService  # noqa: PLC0415

        api_key = (
            cfg.conversation.tts.api_key.get_secret_value()
            if cfg.conversation.tts.api_key
            else None
        )
        if not api_key:
            raise ValueError(
                "DANA_CONVERSATION__TTS__API_KEY is required for the elevenlabs TTS provider"
            )
        voice_id = cfg.conversation.tts.voice or "Rachel"
        return ElevenLabsTTSService(
            api_key=api_key,
            voice_id=voice_id,
            sample_rate=_TTS_SAMPLE_RATE,
        )

    # For all other providers (gemini, mock, …) use the provider-ABC bridge.
    if stack.tts is None:
        raise ValueError("TTS provider is not set on the stack")
    return ProviderTTSProcessor(stack.tts, chunk_sentences=cfg.conversation.tts.chunk_sentences)


def _build_wake_gate(cfg: AppConfig, *, bypass: bool) -> WakeGateProcessor:
    """Build a WakeGateProcessor from config.

    If ``DANA_WAKE__API_KEY`` is set, constructs a real ``AssemblyAIWakeDetector``
    so the gate uses Silero VAD + a single AssemblyAI streaming call to detect
    the wake phrase.  Without an API key the gate is inert (stays IDLE forever)
    unless ``bypass=True``.
    """
    detector: AssemblyAIWakeDetector | None = None
    if not bypass and cfg.wake.api_key:
        api_key = cfg.wake.api_key.get_secret_value()
        detector = AssemblyAIWakeDetector(
            api_key=api_key,
            phrase=cfg.wake.phrase,
            sample_rate=_STT_SAMPLE_RATE,
        )
        log.info("wake_detector_ready", phrase=cfg.wake.phrase)
    elif not bypass:
        log.warning(
            "wake_detector_unconfigured",
            hint="Set DANA_WAKE__API_KEY to enable wake-phrase detection, "
            "or use --bypass-wake for development.",
        )

    return WakeGateProcessor(
        detector=detector,
        wake_phrase=cfg.wake.phrase,
        end_phrase=cfg.wake.end_phrase,
        bypass=bypass,
    )


def build_cascade_pipeline(
    cfg: AppConfig,
    stack: AssembledStack,
    *,
    session_id: str | None = None,
    bypass_wake_gate: bool = False,
    mic_gate: MicGate | None = None,
) -> tuple[LocalAudioTransport, Pipeline, WakeGateProcessor]:
    """Assemble and return the Pipecat cascade pipeline.

    Returns ``(transport, pipeline, wake_gate)`` — the gate is returned so the
    session can expose external control (UI activate / typed turns) through
    ``force_active`` / ``inject_transcript``.

    ``bypass_wake_gate=True`` forwards every audio frame to STT immediately,
    skipping wake-word detection — useful for development without a physical
    microphone or wake word engine.

    ``mic_gate`` (optional) inserts a :class:`MuteProcessor` right after the mic
    transport so the control plane / tray can hard-mute capture at the source.
    """
    # ----- transport --------------------------------------------------------
    transport = LocalAudioTransport(
        LocalAudioTransportParams(
            audio_in_enabled=True,
            audio_in_sample_rate=_STT_SAMPLE_RATE,
            audio_in_channels=1,
            audio_out_enabled=True,
            audio_out_sample_rate=_TTS_SAMPLE_RATE,
            audio_out_channels=1,
            vad_analyzer=SileroVADAnalyzer(),
        )
    )

    # ----- processors -------------------------------------------------------
    wake_gate = _build_wake_gate(cfg, bypass=bypass_wake_gate)
    stt = _build_stt_service(cfg, stack)

    # When no_interrupt is enabled, BotSpeakingGate is shared between
    # BrainProcessor (writer) and HalfDuplexGateProcessor (reader) so the
    # pipeline head can drop mic frames while TTS is playing.
    bot_gate: BotSpeakingGate | None = None
    if cfg.conversation.no_interrupt:
        bot_gate = BotSpeakingGate()
        log.info("half_duplex_mode_enabled")

    brain = BrainProcessor(stack.brain, session_id=session_id, bot_gate=bot_gate)  # type: ignore[arg-type]
    tts = _build_tts_service(cfg, stack)

    # Hard mic mute sits at the very head so muted capture never reaches the
    # wake gate or STT. Omitted entirely when no gate is injected (headless run).
    # HalfDuplexGateProcessor follows immediately after so bot-speaking audio
    # is dropped before it reaches the wake gate or STT.
    stages: list[FrameProcessor] = [transport.input()]
    if mic_gate is not None:
        stages.append(MuteProcessor(mic_gate))
    if bot_gate is not None:
        stages.append(HalfDuplexGateProcessor(bot_gate))
    stages += [wake_gate, stt, brain, tts, transport.output()]

    pipeline = Pipeline(stages)

    log.info(
        "pipeline_assembled",
        stt=cfg.conversation.stt.provider,
        tts=cfg.conversation.tts.provider,
        bypass_wake=bypass_wake_gate,
        session_id=session_id,
    )
    return transport, pipeline, wake_gate


def build_s2s_pipeline(
    cfg: AppConfig,
    stack: AssembledStack,
    *,
    session_id: str | None = None,
    mic_gate: MicGate | None = None,
) -> tuple[LocalAudioTransport, Pipeline]:
    """Assemble the speech-to-speech pipeline — the alternative to the cascade.

    A single S2S model owns the whole turn, so there is no wake gate / STT /
    brain / TTS chain: captured audio flows straight into the provider and its
    response audio flows to the speakers::

        transport.input() → [MuteProcessor] → S2SProcessor → transport.output()

    Proves the cascade-or-S2S abstraction: selected purely by
    ``conversation.mode = "s2s"``; the same ``AssembledStack`` carries either an
    ``s2s`` provider or the cascade providers. The mock S2S makes this
    constructible offline; a real provider (``openai_realtime``) runs live.
    """
    if stack.s2s is None:
        raise ValueError("S2S provider is not set on the stack (mode must be 's2s')")

    transport = LocalAudioTransport(
        LocalAudioTransportParams(
            audio_in_enabled=True,
            audio_in_sample_rate=_STT_SAMPLE_RATE,
            audio_in_channels=1,
            audio_out_enabled=True,
            audio_out_sample_rate=_TTS_SAMPLE_RATE,
            audio_out_channels=1,
            vad_analyzer=SileroVADAnalyzer(),
        )
    )

    stages: list[FrameProcessor] = [transport.input()]
    if mic_gate is not None:
        stages.append(MuteProcessor(mic_gate))
    stages += [S2SProcessor(stack.s2s, sample_rate=_STT_SAMPLE_RATE), transport.output()]

    pipeline = Pipeline(stages)
    log.info("s2s_pipeline_assembled", s2s=cfg.conversation.s2s.provider, session_id=session_id)
    return transport, pipeline


# ---------------------------------------------------------------------------
# MockSTTBridge  (offline tests only)
# ---------------------------------------------------------------------------


class _MockSTTBridge(FrameProcessor):  # type: ignore[misc]
    """Wraps our mock STTProvider in a pipecat FrameProcessor so the pipeline
    compiles with no network during offline tests. The mock provider never
    produces real transcriptions — it just passes frames through."""

    def __init__(self, stt_provider: object, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self._provider = stt_provider

    async def process_frame(self, frame: FrameProcessor, direction: object) -> None:
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)

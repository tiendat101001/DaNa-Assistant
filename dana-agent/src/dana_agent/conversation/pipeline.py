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

    transport, pipeline = build_cascade_pipeline(cfg, stack)
    worker = PipelineWorker(pipeline, params=PipelineParams(allow_interruptions=True))
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams

from dana_agent.conversation.processors import (
    BrainProcessor,
    ProviderTTSProcessor,
    WakeGateProcessor,
)
from dana_agent.observability import get_logger
from dana_agent.providers.wake.assemblyai import AssemblyAIWakeDetector

if TYPE_CHECKING:
    from dana_agent.app import AssembledStack
    from dana_agent.config import AppConfig

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
        return _MockSTTBridge(stack.stt)  # type: ignore[arg-type]

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
) -> tuple[LocalAudioTransport, Pipeline]:
    """Assemble and return the Pipecat cascade pipeline.

    Returns the transport (so the caller can start/stop it) and the pipeline.

    ``bypass_wake_gate=True`` forwards every audio frame to STT immediately,
    skipping wake-word detection — useful for development without a physical
    microphone or wake word engine.
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
    brain = BrainProcessor(stack.brain, session_id=session_id)  # type: ignore[arg-type]
    tts = _build_tts_service(cfg, stack)

    pipeline = Pipeline(
        [
            transport.input(),
            wake_gate,
            stt,
            brain,
            tts,
            transport.output(),
        ]
    )

    log.info(
        "pipeline_assembled",
        stt=cfg.conversation.stt.provider,
        tts=cfg.conversation.tts.provider,
        bypass_wake=bypass_wake_gate,
        session_id=session_id,
    )
    return transport, pipeline


# ---------------------------------------------------------------------------
# MockSTTBridge  (offline tests only)
# ---------------------------------------------------------------------------


class _MockSTTBridge(FrameProcessor):
    """Wraps our mock STTProvider in a pipecat FrameProcessor so the pipeline
    compiles with no network during offline tests. The mock provider never
    produces real transcriptions — it just passes frames through."""

    def __init__(self, stt_provider: object, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self._provider = stt_provider

    async def process_frame(self, frame: FrameProcessor, direction: object) -> None:  # type: ignore[override]
        await super().process_frame(frame, direction)  # type: ignore[arg-type]
        await self.push_frame(frame, direction)  # type: ignore[arg-type]

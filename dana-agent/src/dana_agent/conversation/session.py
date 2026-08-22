"""ConversationSession — the always-on voice loop for Phase 3.

Owns the Pipecat pipeline lifecycle (start → run → stop) and the
latency-measurement observer.  The ``__main__`` entry point calls
``ConversationSession.run()`` after ``build_stack()`` succeeds.

Latency spans are emitted as structlog events on every turn so they can be
forwarded to the control API WebSocket (TASK-007) without coupling the core
pipeline to the FastAPI layer::

    [latency] stage=stt turn_id=… ms=…
    [latency] stage=llm_ttft turn_id=… ms=…
    [latency] stage=tts_ttfb turn_id=… ms=…
    [latency] stage=total turn_id=… ms=…
"""

from __future__ import annotations

import time
import uuid
from typing import TYPE_CHECKING

from pipecat.frames.frames import (
    BotStoppedSpeakingFrame,
    Frame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.observers.base_observer import BaseObserver, FramePushed
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.workers.runner import WorkerRunner

from dana_agent.conversation.pipeline import build_cascade_pipeline, build_s2s_pipeline
from dana_agent.observability import get_logger

if TYPE_CHECKING:
    from dana_agent.app import AssembledStack
    from dana_agent.config import AppConfig
    from dana_agent.conversation.mute import MicGate

log = get_logger(__name__)


class _LatencyObserver(BaseObserver):  # type: ignore[misc]
    """Records per-turn latency taps and emits structlog ``latency`` events.

    Tracks three pipeline stages:
    - ``stt``      — VAD silence detected → final transcript arrives
    - ``llm_ttft`` — final transcript → first LLM token (time-to-first-token)
    - ``tts_ttfb`` — first LLM token → first audio chunk (time-to-first-byte)
    """

    def __init__(self) -> None:
        super().__init__()
        self._turn_id: str = ""
        self._t_vad_stop: float = 0.0
        self._t_transcript: float = 0.0
        self._t_first_token: float = 0.0
        self._first_token_seen: bool = False
        self._first_audio_seen: bool = False

    async def on_push_frame(self, data: FramePushed) -> None:
        frame: Frame = data.frame
        now = time.monotonic()

        if isinstance(frame, VADUserStoppedSpeakingFrame):
            self._turn_id = str(uuid.uuid4())[:8]
            self._t_vad_stop = now
            self._first_token_seen = False
            self._first_audio_seen = False

        elif isinstance(frame, TranscriptionFrame):
            self._t_transcript = now
            if self._t_vad_stop:
                ms = int((now - self._t_vad_stop) * 1000)
                log.info("latency", stage="stt", turn_id=self._turn_id, ms=ms)

        elif isinstance(frame, LLMFullResponseStartFrame):
            if not self._t_transcript:
                self._t_transcript = now

        elif isinstance(frame, LLMTextFrame) and not self._first_token_seen:
            self._first_token_seen = True
            self._t_first_token = now
            if self._t_transcript:
                ms = int((now - self._t_transcript) * 1000)
                log.info("latency", stage="llm_ttft", turn_id=self._turn_id, ms=ms)

        elif isinstance(frame, TTSAudioRawFrame) and not self._first_audio_seen:
            self._first_audio_seen = True
            if self._t_first_token:
                ms = int((now - self._t_first_token) * 1000)
                log.info("latency", stage="tts_ttfb", turn_id=self._turn_id, ms=ms)
            if self._t_vad_stop:
                ms_total = int((now - self._t_vad_stop) * 1000)
                log.info("latency", stage="total", turn_id=self._turn_id, ms=ms_total)

        elif isinstance(frame, BotStoppedSpeakingFrame):
            self._t_vad_stop = 0.0
            self._t_transcript = 0.0
            self._t_first_token = 0.0


class ConversationSession:
    """Manages the Pipecat pipeline for one always-on conversation session.

    Lifecycle::

        session = ConversationSession(cfg, stack)
        await session.run()   # blocks until stop() or pipeline ends
        await session.stop()  # graceful shutdown

    ``bypass_wake_gate=True`` skips wake-word detection — useful during
    development and integration tests when a physical microphone or wake word
    engine is not available.
    """

    def __init__(
        self,
        cfg: AppConfig,
        stack: AssembledStack,
        *,
        bypass_wake_gate: bool = False,
        mic_gate: MicGate | None = None,
    ) -> None:
        self._cfg = cfg
        self._stack = stack
        self._bypass_wake_gate = bypass_wake_gate
        self._mic_gate = mic_gate
        self._session_id = str(uuid.uuid4())
        self._runner: WorkerRunner | None = None

    @property
    def session_id(self) -> str:
        return self._session_id

    async def run(self) -> None:
        """Build and run the pipeline until it ends or ``stop()`` is called."""
        if self._cfg.conversation.mode == "s2s":
            _, pipeline = build_s2s_pipeline(
                self._cfg,
                self._stack,
                session_id=self._session_id,
                mic_gate=self._mic_gate,
            )
        else:
            _, pipeline = build_cascade_pipeline(
                self._cfg,
                self._stack,
                session_id=self._session_id,
                bypass_wake_gate=self._bypass_wake_gate,
                mic_gate=self._mic_gate,
            )

        worker = PipelineWorker(
            pipeline,
            observers=[_LatencyObserver()],
            params=PipelineParams(
                audio_in_sample_rate=16_000,
                audio_out_sample_rate=24_000,
            ),
        )

        self._runner = WorkerRunner()
        await self._runner.add_workers(worker)

        log.info(
            "session_start",
            session_id=self._session_id,
            bypass_wake=self._bypass_wake_gate,
        )

        try:
            await self._runner.run()
        finally:
            self._runner = None
            log.info("session_end", session_id=self._session_id)

    async def stop(self) -> None:
        """Gracefully stop the pipeline."""
        if self._runner:
            await self._runner.cancel()
            self._runner = None

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
    [latency] stage=voice_to_first_audio turn_id=… ms=… over_budget=…
    [latency] stage=total turn_id=… ms=…   (alias, legacy)
"""

from __future__ import annotations

import asyncio
import uuid
from typing import TYPE_CHECKING

from pipecat.frames.frames import (
    BotStoppedSpeakingFrame,
    Frame,
    LLMTextFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.observers.base_observer import BaseObserver, FramePushed
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.workers.runner import WorkerRunner

from dana_agent.conversation.ack import AckAudioRawFrame, AckCache
from dana_agent.conversation.latency import TurnClock
from dana_agent.conversation.pipeline import build_cascade_pipeline, build_s2s_pipeline
from dana_agent.conversation.processors import WakeGateProcessor
from dana_agent.conversation.warmup import warmup_stack
from dana_agent.observability import get_logger

if TYPE_CHECKING:
    from dana_agent.app import AssembledStack
    from dana_agent.config import AppConfig
    from dana_agent.conversation.mute import MicGate

log = get_logger(__name__)


class _LatencyObserver(BaseObserver):  # type: ignore[misc]
    """Feeds pipeline frames into a shared :class:`TurnClock`.

    Observers see each frame once per hop, so frames are de-duplicated by id
    where a repeat would restart the turn; the clock itself ignores repeat marks.
    """

    def __init__(self, clock: TurnClock) -> None:
        super().__init__()
        self._clock = clock
        self._last_stop_id: int | None = None

    async def on_push_frame(self, data: FramePushed) -> None:
        frame: Frame = data.frame

        if isinstance(frame, VADUserStoppedSpeakingFrame):
            if frame.id != self._last_stop_id:
                self._last_stop_id = frame.id
                self._clock.mark_end_of_speech()
        elif isinstance(frame, TranscriptionFrame):
            self._clock.mark_stt_final()
        elif isinstance(frame, LLMTextFrame):
            self._clock.mark_llm_first_token()
        elif isinstance(frame, AckAudioRawFrame):
            self._clock.mark_ack_first_audio()  # not the real reply
        elif isinstance(frame, TTSAudioRawFrame):
            self._clock.mark_tts_first_audio()
        elif isinstance(frame, BotStoppedSpeakingFrame):
            self._clock.reset()


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
        # The cascade wake gate, kept so the control plane can drive it
        # (UI activate button, typed turns). None in s2s mode (no gate/brain).
        self._wake_gate: WakeGateProcessor | None = None

    @property
    def session_id(self) -> str:
        return self._session_id

    async def run(self) -> None:
        """Build and run the pipeline until it ends or ``stop()`` is called."""
        lat = self._cfg.conversation.latency
        clock = TurnClock(lat.target_first_audio_ms, vad_delay_ms=lat.vad_stop_secs * 1000)
        if lat.warmup_on_start:
            try:
                await asyncio.wait_for(warmup_stack(self._stack, self._cfg), timeout=10.0)
            except Exception as exc:  # noqa: BLE001
                log.warning("warmup_skipped", exc=repr(exc))

        ack_cache: AckCache | None = None
        if lat.ack_enabled and self._cfg.conversation.mode != "s2s" and self._stack.tts:
            ack_cache = AckCache()
            try:
                await asyncio.wait_for(
                    ack_cache.fill(self._stack.tts, lat.ack_phrases), timeout=10.0
                )
            except Exception as exc:  # noqa: BLE001 - no ack is fine
                log.warning("ack_prepare_failed", exc=repr(exc))

        if self._cfg.conversation.mode == "s2s":
            _, pipeline = build_s2s_pipeline(
                self._cfg,
                self._stack,
                session_id=self._session_id,
                mic_gate=self._mic_gate,
            )
            self._wake_gate = None
        else:
            _, pipeline, wake_gate = build_cascade_pipeline(
                self._cfg,
                self._stack,
                session_id=self._session_id,
                bypass_wake_gate=self._bypass_wake_gate,
                mic_gate=self._mic_gate,
                clock=clock,
                ack_cache=ack_cache,
            )
            self._wake_gate = wake_gate

        worker = PipelineWorker(
            pipeline,
            observers=[_LatencyObserver(clock)],
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
            self._wake_gate = None

    # --- external control (the Talk UI) -------------------------------------

    async def activate(self) -> None:
        """Trip the wake gate to ACTIVE — the UI equivalent of saying the wake
        phrase, so the next spoken sentence goes straight to STT."""
        if self._wake_gate is not None:
            self._wake_gate.force_active(cause="ui_activate")

    async def submit_text(self, text: str) -> None:
        """Run a typed user turn through the live pipeline: the gate injects a
        synthetic final transcript, the brain answers, and TTS speaks the reply
        through the normal path (with all transcript/assistant WS events)."""
        if self._wake_gate is None:
            raise RuntimeError("typed turns need a cascade session (mode='s2s' has no gate/brain)")
        await self._wake_gate.inject_transcript(text)

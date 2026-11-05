"""Per-turn latency clock.

Marks the four moments of a voice turn and emits the structlog ``latency``
events the control log-bridge turns into ``latency_span`` UI events (shape is
fixed: ``stage``, ``turn_id``, ``ms``)::

    end_of_speech -> stt_final -> llm_first_token -> tts_first_audio
        stage "stt"                  (stt_final - end_of_speech)
        stage "llm_ttft"             (llm_first_token - stt_final)
        stage "tts_ttfb"             (tts_first_audio - llm_first_token)
        stage "voice_to_first_audio" (tts_first_audio - end_of_speech)  <- headline
        stage "total"                (alias of the headline, kept for old consumers)

Each mark is idempotent within a turn (pipecat observers see a frame once per
hop), and out-of-order marks are ignored.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable

from dana_agent.observability import get_logger

log = get_logger(__name__)


class TurnClock:
    """Tracks one turn at a time. ``clock`` returns seconds (injectable)."""

    def __init__(
        self,
        target_ms: int = 600,
        *,
        clock: Callable[[], float] = time.monotonic,
        vad_delay_ms: float = 0.0,
    ) -> None:
        self._target_ms = target_ms
        self._clock = clock
        # VAD fires `stop_secs` after the user actually stopped; shift the
        # end-of-speech mark back so the headline number is honest.
        self._vad_delay_s = max(0.0, vad_delay_ms) / 1000.0
        self.turn_id = ""
        self._eos: float | None = None
        self._stt: float | None = None
        self._tok: float | None = None
        self._audio: float | None = None

    @property
    def active(self) -> bool:
        return self._eos is not None

    def reset(self) -> None:
        self.turn_id = ""
        self._eos = self._stt = self._tok = self._audio = None

    def mark_end_of_speech(self) -> str:
        """Start a new turn (VAD stop or typed-turn injection)."""
        self.reset()
        self.turn_id = uuid.uuid4().hex[:8]
        self._eos = self._clock() - self._vad_delay_s
        return self.turn_id

    def mark_stt_final(self) -> None:
        if self._stt is not None:
            return
        now = self._clock()
        if self._eos is None:  # typed turn / provider without VAD frames
            self.turn_id = uuid.uuid4().hex[:8]
            self._eos = now
        self._stt = now
        self._emit("stt", now - self._eos)

    def mark_llm_first_token(self) -> None:
        if self._stt is None or self._tok is not None:
            return
        now = self._clock()
        self._tok = now
        self._emit("llm_ttft", now - self._stt)

    def mark_tts_first_audio(self) -> None:
        if self._eos is None or self._audio is not None or self._stt is None:
            return
        now = self._clock()
        self._audio = now
        if self._tok is not None:
            self._emit("tts_ttfb", now - self._tok)
        total_ms = self._ms(now - self._eos)
        over = total_ms > self._target_ms
        self._emit("voice_to_first_audio", now - self._eos, over_budget=over)
        self._emit("total", now - self._eos)
        if over:
            log.warning(
                "latency_over_budget",
                turn_id=self.turn_id,
                ms=total_ms,
                target_ms=self._target_ms,
            )

    @staticmethod
    def _ms(seconds: float) -> int:
        return max(0, round(seconds * 1000))

    def _emit(self, stage: str, seconds: float, **extra: object) -> None:
        log.info("latency", stage=stage, turn_id=self.turn_id, ms=self._ms(seconds), **extra)

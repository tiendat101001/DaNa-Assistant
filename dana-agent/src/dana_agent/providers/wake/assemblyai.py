"""AssemblyAI-backed wake phrase detector.

Instead of a per-frame keyword spotter, the detection flow is:

  1. Silero VAD (pipecat transport) emits ``VADUserStartedSpeakingFrame`` /
     ``VADUserStoppedSpeakingFrame`` around each utterance.
  2. ``WakeGateProcessor`` buffers ``InputAudioRawFrame`` bytes during speech.
  3. On ``VADUserStoppedSpeakingFrame``, the buffered clip is sent here.
  4. ``AssemblyAIWakeDetector`` opens a single AssemblyAI streaming session,
     sends the clip, waits for a final transcript, then closes the session.
  5. If the transcript contains the configured wake phrase the detector
     returns ``True`` and the gate transitions to ACTIVE.

This avoids a dedicated KWS model (Porcupine, openWakeWord) and reuses the
same AssemblyAI credentials already in use for the main STT stage.
"""

from __future__ import annotations

import asyncio

from dana_agent.observability import get_logger

log = get_logger(__name__)

_CHUNK_SIZE = 4096  # bytes per streaming chunk to AssemblyAI


class AssemblyAIWakeDetector:
    """One-shot wake phrase detector via AssemblyAI streaming transcription.

    Not a ``WakeWordProvider`` — the ABC is synchronous per-frame, but this
    detector is asynchronous and operates on a complete audio clip.  It is
    constructed by the pipeline factory and injected into ``WakeGateProcessor``.

    API key is sourced from ``cfg.wake.api_key`` (env: ``DANA_WAKE__API_KEY``).
    Secrets are never logged.
    """

    def __init__(
        self,
        api_key: str,
        phrase: str,
        *,
        sample_rate: int = 16_000,
        timeout: float = 8.0,
        language_code: str = "en",
    ) -> None:
        self._api_key = api_key
        self._phrase = phrase.lower()
        self._sample_rate = sample_rate
        self._timeout = timeout
        self._language_code = language_code

    async def detect(self, pcm: bytes) -> bool:
        """Transcribe *pcm* and return True if the wake phrase is present."""
        if not pcm:
            return False
        try:
            return await asyncio.wait_for(self._transcribe_and_check(pcm), self._timeout)
        except TimeoutError:
            log.warning("wake_detect_timeout", phrase=self._phrase)
            return False
        except Exception as exc:
            log.error("wake_detect_error", exc=str(exc))
            return False

    async def _transcribe_and_check(self, pcm: bytes) -> bool:
        try:
            from assemblyai.streaming.v3 import (  # noqa: PLC0415
                AsyncRealTimeTranscriber,
                RealTimeEvents,
            )
            from assemblyai.streaming.v3.models import RealTimeParameters  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "assemblyai>=1.0 is required for wake detection: pip install 'assemblyai>=1.0'"
            ) from exc

        found = False
        done = asyncio.Event()

        def _on_turn(_client: object, event: object) -> None:
            nonlocal found
            text = getattr(event, "transcript", "")
            if text and self._phrase in text.lower():
                log.debug("wake_phrase_found", text=text[:80])
                found = True

        def _on_termination(_client: object, _event: object) -> None:
            done.set()

        def _on_error(_client: object, event: object) -> None:
            log.debug("wake_detect_session_error", msg=str(getattr(event, "message", event)))
            done.set()

        transcriber = AsyncRealTimeTranscriber(api_key=self._api_key)
        transcriber.on(RealTimeEvents.Turn, _on_turn)
        transcriber.on(RealTimeEvents.Termination, _on_termination)
        transcriber.on(RealTimeEvents.Error, _on_error)

        params = RealTimeParameters(
            sample_rate=self._sample_rate,
            language_code=self._language_code,
            language_codes=[self._language_code],
            language_detection=False,
        )
        await transcriber.connect(params)

        for offset in range(0, len(pcm), _CHUNK_SIZE):
            await transcriber.stream(pcm[offset : offset + _CHUNK_SIZE])

        await transcriber.disconnect(terminate=True)
        await done.wait()
        return found

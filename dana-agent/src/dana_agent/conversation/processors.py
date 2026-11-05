"""Custom pipecat FrameProcessors that bridge DaNa's provider layer into the
Pipecat pipeline frame protocol.

Three processors cover the non-standard stages of the DaNa pipeline:

- ``WakeGateProcessor``  — gates InputAudioRawFrame through the IDLE/ACTIVE
  state machine; frames only reach the STT service when ACTIVE.
- ``BrainProcessor``     — receives final TranscriptionFrames, calls the
  LangGraph Brain, and emits LLMTextFrames that feed the TTS service.
- ``ProviderTTSProcessor`` — wraps our ``TTSProvider`` ABC in pipecat's frame
  protocol, doing sentence-level streaming so TTS starts before the LLM
  finishes the full reply.

Frame flow (cascade mode)::

    InputAudioRawFrame
        → WakeGateProcessor    (buffer in IDLE; forward in ACTIVE)
        → AssemblyAISTTService (cloud) / WebRTCVADService (local)
        → BrainProcessor       (TranscriptionFrame → LLMTextFrame stream)
        → ProviderTTSProcessor (LLMTextFrame → TTSAudioRawFrame)
        → LocalAudioOutputTransport

Wake detection (IDLE state):
  Pipecat's transport VAD (SileroVADAnalyzer) emits VADUserStartedSpeakingFrame
  and VADUserStoppedSpeakingFrame.  WakeGateProcessor buffers InputAudioRawFrames
  during speech, then sends the clip to AssemblyAIWakeDetector for a single
  streaming transcription request.  If the transcript contains the wake phrase
  ("Hey DaNa") the gate transitions to ACTIVE.
"""

from __future__ import annotations

import asyncio
import re
import time
import uuid
from collections import deque
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    CancelFrame,
    EndFrame,
    Frame,
    InputAudioRawFrame,
    InterimTranscriptionFrame,
    InterruptionFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.utils.time import time_now_iso8601

from dana_agent.brain.agent import Brain
from dana_agent.brain.events import ConfirmationRequired, TokenDelta
from dana_agent.conversation.mute import BotSpeakingGate, MicGate
from dana_agent.observability import get_logger
from dana_agent.providers.base import S2SProvider, STTProvider, TTSProvider
from dana_agent.types import AudioFrame

if TYPE_CHECKING:
    from dana_agent.conversation.latency import TurnClock
    from dana_agent.providers.wake.assemblyai import AssemblyAIWakeDetector

log = get_logger(__name__)

# Sentence boundary pattern: flush text to TTS when these appear.
_SENTENCE_END = re.compile(r"[.!?。！？…\n]")
_SENTENCE_CHARS = frozenset(".!?。！？…\n")
_CLAUSE_CHARS = frozenset(",;:—–，；：、") | _SENTENCE_CHARS
# "3,5" / "1.5" / "10:30": punctuation right after a digit is held until we see
# what follows (whitespace -> boundary; digit -> part of a number).
_NUMERIC_PUNCT = frozenset(",.:")


def eager_split(buffer: str, min_chars: int) -> int:
    """Index just past the first flushable boundary in ``buffer``, or ``-1``.

    A boundary is a clause char (``, ; : - —``) or sentence end that is followed
    by whitespace (or ends the buffer, unless it sits right after a digit so we
    never split numbers). Clause boundaries need ``>= min_chars`` of text before
    them; sentence ends flush regardless of length.
    """
    n = len(buffer)
    for i, ch in enumerate(buffer):
        if ch not in _CLAUSE_CHARS:
            continue
        nxt_ok = i + 1 == n or buffer[i + 1].isspace()
        if not nxt_ok:
            continue
        if i + 1 == n and ch in _NUMERIC_PUNCT and i > 0 and buffer[i - 1].isdigit():
            continue
        if ch in _SENTENCE_CHARS or len(buffer[: i + 1].strip()) >= min_chars:
            if buffer[: i + 1].strip():
                return i + 1
    return -1


# ---------------------------------------------------------------------------
# MuteProcessor
# ---------------------------------------------------------------------------


class HalfDuplexGateProcessor(FrameProcessor):  # type: ignore[misc]
    """Drops mic audio while the bot is speaking (half-duplex / no-interrupt mode).

    Sits at the head of the pipeline, right after :class:`MuteProcessor`.
    When :class:`~dana_agent.conversation.mute.BotSpeakingGate` signals that
    the bot has started speaking, ``InputAudioRawFrame`` and the VAD boundary
    frames derived from it are silently dropped so the bot's own audio output
    cannot reach the wake gate or STT and be mistaken for new user input.

    Only inserted into the pipeline when ``DANA_CONVERSATION__NO_INTERRUPT=true``.
    """

    def __init__(self, gate: BotSpeakingGate, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self._gate = gate

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if self._gate.speaking and isinstance(
            frame,
            (InputAudioRawFrame, VADUserStartedSpeakingFrame, VADUserStoppedSpeakingFrame),
        ):
            return  # bot is speaking — drop mic frames to prevent loopback
        await self.push_frame(frame, direction)


class MuteProcessor(FrameProcessor):  # type: ignore[misc]
    """Hard mic mute at the head of the pipeline (docs/PLAN.md §8).

    Sits immediately after ``transport.input()``. While the shared
    :class:`~dana_agent.conversation.mute.MicGate` is muted, every
    ``InputAudioRawFrame`` (and the VAD speech-boundary frames derived from it)
    is dropped here, so no captured audio ever reaches the wake gate, STT, the
    brain, or any provider. All non-capture frames pass through untouched, so the
    pipeline still shuts down and plays queued output normally while muted.
    """

    def __init__(self, gate: MicGate, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self._gate = gate

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if self._gate.muted and isinstance(
            frame,
            (
                InputAudioRawFrame,
                VADUserStartedSpeakingFrame,
                VADUserStoppedSpeakingFrame,
            ),
        ):
            return  # drop capture at the source — a real gate, not a flag
        await self.push_frame(frame, direction)


# ---------------------------------------------------------------------------
# WakeGateProcessor
# ---------------------------------------------------------------------------


class WakeGateProcessor(FrameProcessor):  # type: ignore[misc]
    """Gates audio through the IDLE/ACTIVE state machine via AssemblyAI detection.

    In IDLE state the transport's Silero VAD drives speech segmentation:

    - ``VADUserStartedSpeakingFrame`` → begin buffering ``InputAudioRawFrame`` bytes.
    - ``InputAudioRawFrame``          → accumulate in the buffer (not forwarded).
    - ``VADUserStoppedSpeakingFrame`` → send the buffered clip to the detector.

    ``AssemblyAIWakeDetector`` transcribes the clip and checks for the wake phrase.
    On a match the gate enters ACTIVE and forwards all subsequent audio to STT.

    In ACTIVE state all VAD and audio frames are forwarded downstream.
    ``TranscriptionFrame`` is monitored for the end phrase ("Bye DaNa"); on a
    match the gate returns to IDLE and the frame is consumed (not forwarded).

    ``bypass=True`` skips the gate entirely — every frame is forwarded.
    Used in development and integration tests that have no API key or microphone.
    """

    def __init__(
        self,
        detector: AssemblyAIWakeDetector | None = None,
        wake_phrase: str = "Hey DaNa",
        end_phrase: str = "Bye DaNa",
        *,
        bypass: bool = False,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        self._detector = detector
        self._wake_phrase = wake_phrase.lower()
        self._end_phrase = end_phrase.lower()
        self._bypass = bypass
        self._active = False
        self._buffering = False
        self._audio_buffer: list[bytes] = []
        self._detect_task: asyncio.Task[None] | None = None

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)

        if self._bypass:
            await self.push_frame(frame, direction)
            return

        # --- VAD speech-start signal ----------------------------------------
        if isinstance(frame, VADUserStartedSpeakingFrame):
            if not self._active:
                self._buffering = True
                self._audio_buffer = []
                log.debug("wake_buffer_start")
            else:
                await self.push_frame(frame, direction)

        # --- raw audio frame ------------------------------------------------
        elif isinstance(frame, InputAudioRawFrame):
            if self._active:
                await self.push_frame(frame, direction)
            elif self._buffering:
                self._audio_buffer.append(frame.audio)

        # --- VAD speech-stop signal: run detection --------------------------
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            if not self._active and self._buffering:
                self._buffering = False
                pcm = b"".join(self._audio_buffer)
                self._audio_buffer = []
                if self._detector and pcm:
                    self._detect_task = asyncio.ensure_future(self._detect_and_activate(pcm))
            elif self._active:
                await self.push_frame(frame, direction)

        # --- transcription: monitor for end phrase --------------------------
        elif isinstance(frame, TranscriptionFrame):
            if self._active:
                if self._end_phrase in frame.text.lower():
                    log.info("wake_gate_idle", cause="end_phrase")
                    self._active = False
                else:
                    await self.push_frame(frame, direction)
            # In IDLE: discard (shouldn't arrive since audio isn't forwarded)

        # --- interruption / cancel ------------------------------------------
        elif isinstance(frame, (InterruptionFrame, CancelFrame)):
            if self._active:
                self._active = False
                log.info("wake_gate_idle", cause="cancel")
            await self.push_frame(frame, direction)

        # --- pipeline teardown ----------------------------------------------
        elif isinstance(frame, EndFrame):
            if self._detect_task and not self._detect_task.done():
                self._detect_task.cancel()
            await self.push_frame(frame, direction)

        # --- everything else ------------------------------------------------
        else:
            await self.push_frame(frame, direction)

    async def _detect_and_activate(self, pcm: bytes) -> None:
        try:
            found = await self._detector.detect(pcm)  # type: ignore[union-attr]
            if found and not self._active:
                self._active = True
                log.info("wake_gate_active", method="assemblyai_stt")
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            log.error("wake_detect_unexpected", exc=str(exc))

    # --- external control (Talk UI) ----------------------------------------

    def force_active(self, cause: str = "manual") -> None:
        """Trip the gate to ACTIVE from outside the pipeline — the UI's Talk
        button. Same effect as hearing the wake phrase: subsequent mic audio
        flows to STT without one. The ``wake_gate_active`` log event keeps the
        state_transition WS stream truthful."""
        if self._bypass or self._active:
            return
        self._active = True
        self._buffering = False
        self._audio_buffer = []
        log.info("wake_gate_active", method="ui", cause=cause)

    async def inject_transcript(self, text: str) -> None:
        """Feed a typed user turn straight into the conversation downstream of
        the gate. The normal path takes over from there: BrainProcessor runs
        the turn (and logs ``transcript_final``), TTS speaks the reply, every
        WS event fires — a text turn is indistinguishable from a spoken one.
        Typing the end phrase ends the session, exactly like saying it."""
        text = text.strip()
        if not text:
            return
        if self._end_phrase in text.lower():
            self._active = False
            self._buffering = False
            self._audio_buffer = []
            log.info("wake_gate_idle", cause="end_phrase")
            return  # consumed, like the spoken end phrase
        self.force_active(cause="text_turn")
        await self.push_frame(TranscriptionFrame(text=text, user_id="ui", timestamp=""))


# ---------------------------------------------------------------------------
# BrainProcessor
# ---------------------------------------------------------------------------


class BrainProcessor(FrameProcessor):  # type: ignore[misc]
    """Converts final transcript frames into LLM response frames.

    Listens for ``TranscriptionFrame`` from the STT service, calls the
    LangGraph ``Brain``, and emits the streaming token output as
    ``LLMTextFrame``s bracketed by ``LLMFullResponseStartFrame`` /
    ``LLMFullResponseEndFrame`` (the frame contract that pipecat TTS services
    expect).

    An ``InterruptionFrame`` cancels any in-flight turn immediately.
    """

    def __init__(
        self,
        brain: Brain,
        session_id: str | None = None,
        *,
        bot_gate: BotSpeakingGate | None = None,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        self._brain = brain
        self._session_id = session_id or str(uuid.uuid4())
        self._turn_task: asyncio.Task[None] | None = None
        self._bot_gate = bot_gate

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame) and direction == FrameDirection.DOWNSTREAM:
            text = frame.text.strip()
            if not text:
                return
            log.info("brain_turn_start", text=text, session=self._session_id)
            # Neutral event tap (carry-over B): the final user transcript. The
            # control log-bridge republishes this as a `transcript_final` WS event
            # WITHOUT this module importing the control plane.
            log.info("transcript_final", text=text, role="user", session=self._session_id)
            await self._cancel_turn()
            self._turn_task = asyncio.ensure_future(self._run_turn(text))

        elif isinstance(frame, (InterruptionFrame, CancelFrame)):
            await self._cancel_turn()
            await self.push_frame(frame, direction)

        elif isinstance(frame, EndFrame):
            await self._cancel_turn()
            await self.push_frame(frame, direction)

        else:
            await self.push_frame(frame, direction)

    async def _cancel_turn(self) -> None:
        if self._turn_task and not self._turn_task.done():
            self._turn_task.cancel()
            try:
                await self._turn_task
            except (asyncio.CancelledError, Exception) as exc:
                log.debug("brain_cancel_swallowed", exc=str(exc))
        self._turn_task = None

    async def _run_turn(self, user_text: str) -> None:
        if self._bot_gate is not None:
            self._bot_gate.set_speaking(True)
        await self.push_frame(BotStartedSpeakingFrame())
        await self.push_frame(LLMFullResponseStartFrame())
        reply_parts: list[str] = []
        try:
            async for event in self._brain.run_turn(user_text, thread_id=self._session_id):
                if isinstance(event, TokenDelta) and event.text:
                    reply_parts.append(event.text)
                    await self.push_frame(LLMTextFrame(text=event.text))
                elif isinstance(event, ConfirmationRequired):
                    # Neutral event tap (carry-over B): the log-bridge maps this to
                    # a `confirmation_request` WS event so the dashboard sees the
                    # pending destructive action. Arguments are redacted before
                    # logging so secrets never reach the durable trail or WS stream;
                    # the preview is phrased in human units (voice reads it aloud).
                    from dana_agent.actions.audit import (  # noqa: PLC0415
                        human_preview,
                        redact_arguments,
                    )

                    safe_args = redact_arguments(dict(event.arguments))
                    preview = human_preview(event.tool, dict(event.arguments))
                    log.info(
                        "confirmation_required",
                        tool=event.tool,
                        id=event.id,
                        reversibility=event.reversibility.value,
                        args_resolved=safe_args,
                        preview=preview,
                        session=self._session_id,
                    )
        except asyncio.CancelledError:
            log.debug("brain_turn_cancelled")
            raise
        except Exception as exc:
            log.exception("brain_turn_error", exc_info=exc)
        finally:
            reply = "".join(reply_parts).strip()
            if reply:
                # Neutral event tap (carry-over B): the assistant's spoken reply →
                # `assistant_turn` WS event via the log-bridge.
                log.info("assistant_turn", text=reply, session=self._session_id)
            await self.push_frame(LLMFullResponseEndFrame())
            await self.push_frame(BotStoppedSpeakingFrame())
            if self._bot_gate is not None:
                self._bot_gate.set_speaking(False)


# ---------------------------------------------------------------------------
# ProviderTTSProcessor
# ---------------------------------------------------------------------------


class ProviderTTSProcessor(FrameProcessor):  # type: ignore[misc]
    """Wraps our ``TTSProvider`` ABC in pipecat's frame protocol.

    Collects ``LLMTextFrame`` tokens, flushes them to the provider in configurable
    batches, and emits each synthesised PCM chunk as a ``TTSAudioRawFrame``.

    ``chunk_sentences`` controls how many sentence boundaries are accumulated
    before a TTS request is sent:

    * ``0`` — buffer the whole LLM reply and send exactly one TTS request per
      turn.  Fewest API calls; audio doesn't start until the LLM finishes.
    * ``1`` — flush at every sentence boundary (original behaviour).  Lowest
      latency to first audio; one API request per sentence.
    * ``N`` — flush every N sentences.  A middle ground between latency and
      request count.
    """

    def __init__(
        self,
        provider: TTSProvider,
        *,
        chunk_sentences: int = 1,
        first_chunk_min_chars: int = 0,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        self._provider = provider
        self._chunk_sentences = chunk_sentences
        # Eager first chunk (0 = off): the first chunk of each reply flushes at
        # the first clause boundary holding >= N chars, or at a sentence end.
        self._first_min = first_chunk_min_chars
        self._first_done = False
        self._buffer = ""
        self._sentence_count = 0
        self._active = False
        self._tts_task: asyncio.Task[None] | None = None
        self._text_queue: asyncio.Queue[str | None] = asyncio.Queue()

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)

        if isinstance(frame, LLMFullResponseStartFrame):
            self._active = True
            self._buffer = ""
            self._sentence_count = 0
            self._first_done = False
            self._text_queue = asyncio.Queue()
            self._tts_task = asyncio.ensure_future(self._run_tts())
            await self.push_frame(frame, direction)

        elif isinstance(frame, LLMTextFrame) and self._active:
            self._buffer += frame.text
            if self._first_min > 0 and not self._first_done:
                cut = eager_split(self._buffer, self._first_min)
                if cut > 0:
                    chunk = self._buffer[:cut].strip()
                    self._buffer = self._buffer[cut:]
                    self._sentence_count = 0
                    self._first_done = True
                    await self._text_queue.put(chunk)
                return
            if _SENTENCE_END.search(frame.text):
                self._sentence_count += 1
                # chunk_sentences=0 → hold everything until LLMFullResponseEndFrame
                if self._chunk_sentences > 0 and self._sentence_count >= self._chunk_sentences:
                    chunk = self._buffer.strip()
                    self._buffer = ""
                    self._sentence_count = 0
                    if chunk:
                        await self._text_queue.put(chunk)

        elif isinstance(frame, LLMFullResponseEndFrame):
            if self._active:
                # Flush whatever is still buffered (partial sentence or full reply
                # when chunk_sentences=0).
                tail = self._buffer.strip()
                if tail:
                    await self._text_queue.put(tail)
                self._buffer = ""
                self._sentence_count = 0
                await self._text_queue.put(None)  # sentinel
                if self._tts_task:
                    await self._tts_task
                self._active = False
            await self.push_frame(frame, direction)

        elif isinstance(frame, (InterruptionFrame, CancelFrame)):
            await self._cancel_tts()
            await self.push_frame(frame, direction)

        elif isinstance(frame, EndFrame):
            await self._cancel_tts()
            await self.push_frame(frame, direction)

        else:
            await self.push_frame(frame, direction)

    async def _cancel_tts(self) -> None:
        self._active = False
        await self._text_queue.put(None)
        if self._tts_task and not self._tts_task.done():
            self._tts_task.cancel()
            try:
                await self._tts_task
            except (asyncio.CancelledError, Exception) as exc:
                log.debug("tts_cancel_swallowed", exc=str(exc))
        self._tts_task = None

    async def _run_tts(self) -> None:
        """Drive the provider's async-generator over the sentence queue."""
        try:
            async for chunk in self._provider.stream(self._sentence_iter()):
                frame = TTSAudioRawFrame(
                    audio=chunk.pcm,
                    sample_rate=chunk.sample_rate,
                    num_channels=1,
                )
                await self.push_frame(frame)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.exception("tts_error", exc_info=exc)

    async def _sentence_iter(self) -> AsyncIterator[str]:
        while True:
            sentence = await self._text_queue.get()
            if sentence is None:
                break
            yield sentence


# ---------------------------------------------------------------------------
# S2SProcessor  (speech-to-speech mode)
# ---------------------------------------------------------------------------


class S2SProcessor(FrameProcessor):  # type: ignore[misc]
    """Bridges an ``S2SProvider`` into the pipecat frame protocol for the
    speech-to-speech pipeline (the alternative to the STT+brain+TTS cascade).

    Captured ``InputAudioRawFrame`` bytes are handed to the provider's
    ``stream()`` as neutral :class:`~dana_agent.types.AudioFrame`s; each response
    :class:`~dana_agent.types.AudioChunk` is emitted downstream as a
    ``TTSAudioRawFrame`` for the output transport. This mirrors
    :class:`ProviderTTSProcessor` but for a single speech-to-speech model — the
    provider owns the whole turn (STT + reasoning + TTS) internally.

    The mock S2S provider makes this fully offline-constructible; a real provider
    (``openai_realtime``) opens a live session and is exercised only under the
    ``live`` marker.
    """

    def __init__(
        self, provider: S2SProvider, *, sample_rate: int = 16_000, **kwargs: object
    ) -> None:
        super().__init__(**kwargs)
        self._provider = provider
        self._sample_rate = sample_rate
        self._audio_queue: asyncio.Queue[AudioFrame | None] = asyncio.Queue()
        self._task: asyncio.Task[None] | None = None

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)

        if isinstance(frame, InputAudioRawFrame):
            if self._task is None:
                self._task = asyncio.ensure_future(self._run())
            await self._audio_queue.put(AudioFrame(pcm=frame.audio, sample_rate=self._sample_rate))

        elif isinstance(frame, (InterruptionFrame, CancelFrame, EndFrame)):
            await self._stop()
            await self.push_frame(frame, direction)

        else:
            await self.push_frame(frame, direction)

    async def _stop(self) -> None:
        await self._audio_queue.put(None)
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception) as exc:  # noqa: BLE001
                log.debug("s2s_cancel_swallowed", exc=str(exc))
        self._task = None

    async def _run(self) -> None:
        try:
            async for chunk in self._provider.stream(self._audio_iter()):
                await self.push_frame(
                    TTSAudioRawFrame(
                        audio=chunk.pcm,
                        sample_rate=chunk.sample_rate,
                        num_channels=1,
                    )
                )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.exception("s2s_error", exc_info=exc)

    async def _audio_iter(self) -> AsyncIterator[AudioFrame]:
        while True:
            frame = await self._audio_queue.get()
            if frame is None:
                break
            yield frame


# ---------------------------------------------------------------------------
# ProviderSTTProcessor  (any STTProvider ABC -> pipecat)
# ---------------------------------------------------------------------------

_MAX_UTTERANCE_S = 30.0
_PREROLL_S = 0.3


class ProviderSTTProcessor(FrameProcessor):  # type: ignore[misc]
    """Bridges any :class:`STTProvider` (litert, faster_whisper, mock, ...) into
    pipecat, using the transport's VAD boundaries to segment utterances.

    A ~300 ms pre-roll ring of ``InputAudioRawFrame`` is always kept, so the
    soft onset VAD needs ``start_secs`` to confirm is not clipped. On
    ``VADUserStartedSpeakingFrame`` the pre-roll seeds a new utterance; on
    ``VADUserStoppedSpeakingFrame`` the utterance is handed to
    ``provider.stream()`` in a background task (the pipeline is never blocked),
    partials become ``InterimTranscriptionFrame`` and finals
    ``TranscriptionFrame``. Every incoming frame is passed through unchanged.
    Provider errors are logged, never raised.
    """

    def __init__(
        self,
        provider: STTProvider,
        *,
        sample_rate: int = 16_000,
        preroll_s: float = _PREROLL_S,
        max_utterance_s: float = _MAX_UTTERANCE_S,
        clock: TurnClock | None = None,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        self._provider = provider
        self._sample_rate = sample_rate
        self._preroll_bytes = int(preroll_s * sample_rate) * 2
        self._max_bytes = int(max_utterance_s * sample_rate) * 2
        self._clock = clock
        self._preroll: deque[InputAudioRawFrame] = deque()
        self._preroll_len = 0
        self._utterance: list[InputAudioRawFrame] | None = None
        self._utt_len = 0
        self._truncated = False
        self._tasks: set[asyncio.Task[None]] = set()

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)

        if isinstance(frame, InputAudioRawFrame):
            self._on_audio(frame)
        elif isinstance(frame, VADUserStartedSpeakingFrame):
            self._utterance = list(self._preroll)
            self._utt_len = self._preroll_len
            self._truncated = False
            self._preroll.clear()
            self._preroll_len = 0
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            if self._utterance:
                frames, self._utterance = self._utterance, None
                task = asyncio.ensure_future(self._transcribe(frames))
                self._tasks.add(task)
                task.add_done_callback(self._tasks.discard)
            else:
                self._utterance = None
        elif isinstance(frame, CancelFrame):
            await self._cancel_tasks()
        elif isinstance(frame, EndFrame):
            await self.wait_idle(wait_s=5.0)  # let the last transcript out
        await self.push_frame(frame, direction)

    def _on_audio(self, frame: InputAudioRawFrame) -> None:
        if self._utterance is not None:
            if self._utt_len + len(frame.audio) > self._max_bytes:
                if not self._truncated:
                    self._truncated = True
                    log.warning("stt_utterance_capped", max_bytes=self._max_bytes)
                return
            self._utterance.append(frame)
            self._utt_len += len(frame.audio)
            return
        self._preroll.append(frame)
        self._preroll_len += len(frame.audio)
        while self._preroll_len > self._preroll_bytes and len(self._preroll) > 1:
            self._preroll_len -= len(self._preroll.popleft().audio)

    async def wait_idle(self, wait_s: float | None = None) -> None:
        """Wait for in-flight transcriptions (used on EndFrame and in tests)."""
        if self._tasks:
            await asyncio.wait(set(self._tasks), timeout=wait_s)

    async def _cancel_tasks(self) -> None:
        for t in list(self._tasks):
            t.cancel()
        await self.wait_idle(wait_s=1.0)

    async def _transcribe(self, frames: list[InputAudioRawFrame]) -> None:
        t0 = time.monotonic()
        last_partial = ""
        language = "en"
        got_final = False

        async def audio() -> AsyncIterator[AudioFrame]:
            for f in frames:
                yield AudioFrame(
                    pcm=f.audio,
                    sample_rate=f.sample_rate or self._sample_rate,
                    channels=f.num_channels or 1,
                )

        try:
            async for tr in self._provider.stream(audio()):
                text = tr.text.strip()
                language = tr.language or language
                if not tr.is_final:
                    if text:
                        last_partial = text
                        await self.push_frame(
                            InterimTranscriptionFrame(
                                text=text, user_id="user", timestamp=time_now_iso8601()
                            )
                        )
                    continue
                got_final = True
                if text:
                    await self._push_final(text, t0)
            if not got_final and last_partial:
                await self._push_final(last_partial, t0)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - never crash the pipeline
            log.error("stt_error", exc=str(exc))

    async def _push_final(self, text: str, t0: float) -> None:
        if self._clock is not None:
            self._clock.mark_stt_final()
        else:
            log.info(
                "latency",
                stage="stt",
                turn_id=uuid.uuid4().hex[:8],
                ms=round((time.monotonic() - t0) * 1000),
            )
        await self.push_frame(
            TranscriptionFrame(text=text, user_id="user", timestamp=time_now_iso8601())
        )

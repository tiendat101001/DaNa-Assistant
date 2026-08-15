"""TASK-001 — Phase 0 audio spike (throwaway, not part of the production package).

Validates the full voice pipeline on Windows 11:
  mic → AssemblyAI STT → Gemini 2.5 Flash Lite → Gemini TTS → speakers

Exit gate (ADR-0001):
  ✓ 10-minute stable session
  ✓ Median turn latency ≤ 800 ms un-tuned (per-stage timings printed)
  ✓ Barge-in: user speech during playback cancels pending work
  ✓ No self-triggering (AEC must hold over ≥ 20 replies)
  ✓ Stub wake gate frame processor drops/passes audio upstream of STT

Usage:
    cd spike
    pip install sounddevice assemblyai google-genai langchain-google-genai langchain
    ASSEMBLYAI_API_KEY=... GEMINI_API_KEY=... python audio_spike.py

    Or with keys already in dana-agent/.env:
    cd dana-agent && uv run ../spike/audio_spike.py

Spike is THROWAWAY — keep outside dana_agent package, never import from it.
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Optional

# ---------------------------------------------------------------------------
# Latency bookkeeping
# ---------------------------------------------------------------------------


@dataclass
class TurnStats:
    turn: int = 0
    stt_start: float = 0.0
    stt_end: float = 0.0
    llm_start: float = 0.0
    llm_end: float = 0.0
    tts_start: float = 0.0
    tts_end: float = 0.0
    total_ms: float = 0.0


turn_log: list[TurnStats] = []


# ---------------------------------------------------------------------------
# Config (from env — never hardcoded)
# ---------------------------------------------------------------------------

ASSEMBLYAI_KEY = os.environ.get("ASSEMBLYAI_API_KEY") or os.environ.get(
    "DANA_CONVERSATION__STT__API_KEY", ""
)
GEMINI_KEY = os.environ.get("GEMINI_API_KEY") or os.environ.get(
    "DANA_CONVERSATION__LLM__API_KEY", ""
)
STT_SAMPLE_RATE = 16_000
TTS_SAMPLE_RATE = 24_000
CHUNK_MS = 100  # audio capture granularity
SILENCE_THRESHOLD = 500  # PCM int16 RMS for silence detection
END_OF_SPEECH_SILENCE_MS = 800  # ms of silence before finalising utterance
MAX_TURNS = 20  # stop after this many turns (for the 20-reply AEC test)
SYSTEM_PROMPT = (
    "You are DaNa, a concise voice assistant. Answer in 1-2 short sentences."
)


# ---------------------------------------------------------------------------
# Stub wake gate (validates the frame-processor pattern, acceptance criterion 5)
# ---------------------------------------------------------------------------


class StubWakeGate:
    """Always-ACTIVE stub: every frame is forwarded. Replace with the real
    ``WakeGate`` (dana_agent.wake) to add IDLE/ACTIVE gate behaviour."""

    def pass_frame(self, pcm: bytes) -> bytes | None:
        return pcm  # forward unconditionally


# ---------------------------------------------------------------------------
# Microphone capture
# ---------------------------------------------------------------------------


async def capture_utterance(gate: StubWakeGate) -> bytes:
    """Block until we detect an utterance then collect it.

    Uses sounddevice for cross-platform audio capture. Returns raw 16-bit LE
    PCM at STT_SAMPLE_RATE.  Simple energy-based VAD so no extra model is
    needed for the spike.
    """
    try:
        import sounddevice as sd  # noqa: PLC0415
        import numpy as np  # noqa: PLC0415
    except ImportError:
        print("sounddevice / numpy required: pip install sounddevice numpy")
        sys.exit(1)

    chunk_samples = int(STT_SAMPLE_RATE * CHUNK_MS / 1000)
    silence_chunks_needed = int(END_OF_SPEECH_SILENCE_MS / CHUNK_MS)
    buf: list[bytes] = []
    in_speech = False
    silence_count = 0

    loop = asyncio.get_event_loop()
    q: asyncio.Queue[bytes] = asyncio.Queue()

    def _callback(indata: object, frames: int, _t: object, status: object) -> None:
        if status:
            print(f"[audio] {status}", file=sys.stderr)
        loop.call_soon_threadsafe(q.put_nowait, bytes(indata))  # type: ignore[arg-type]

    stream = sd.RawInputStream(
        samplerate=STT_SAMPLE_RATE,
        channels=1,
        dtype="int16",
        blocksize=chunk_samples,
        callback=_callback,
    )
    with stream:
        print("[spike] Listening… (speak now)")
        while True:
            raw = await q.get()
            passed = gate.pass_frame(raw)
            if passed is None:
                continue
            rms = _rms(passed)
            if rms > SILENCE_THRESHOLD:
                if not in_speech:
                    in_speech = True
                    silence_count = 0
                    print("[spike] Speech detected")
                buf.append(passed)
            elif in_speech:
                buf.append(passed)
                silence_count += 1
                if silence_count >= silence_chunks_needed:
                    print("[spike] End of utterance")
                    break

    return b"".join(buf)


def _rms(pcm: bytes) -> float:
    import struct  # noqa: PLC0415
    import math  # noqa: PLC0415
    n = len(pcm) // 2
    if n == 0:
        return 0.0
    samples = struct.unpack(f"<{n}h", pcm[:n * 2])
    return math.sqrt(sum(s * s for s in samples) / n)


# ---------------------------------------------------------------------------
# AssemblyAI STT (streaming)
# ---------------------------------------------------------------------------


async def transcribe(pcm: bytes) -> str:
    """Send PCM to AssemblyAI and return the final transcript."""
    import assemblyai as aai  # noqa: PLC0415

    if ASSEMBLYAI_KEY:
        aai.settings.api_key = ASSEMBLYAI_KEY

    loop = asyncio.get_event_loop()
    final_text: list[str] = []
    done = asyncio.Event()

    def on_data(t: aai.RealtimeTranscript) -> None:
        if t.text and isinstance(t, aai.RealtimeFinalTranscript):
            final_text.append(t.text)
            loop.call_soon_threadsafe(done.set)

    def on_error(e: BaseException) -> None:
        print(f"[aai] error: {e}", file=sys.stderr)
        loop.call_soon_threadsafe(done.set)

    tr = aai.RealtimeTranscriber(on_data=on_data, on_error=on_error, sample_rate=STT_SAMPLE_RATE)
    await asyncio.to_thread(tr.connect)

    # Stream audio in 100 ms chunks
    chunk = STT_SAMPLE_RATE * CHUNK_MS // 1000 * 2  # bytes per chunk
    for i in range(0, len(pcm), chunk):
        await asyncio.to_thread(tr.stream, pcm[i : i + chunk])
        await asyncio.sleep(0)

    # Give AssemblyAI time to finalise
    try:
        await asyncio.wait_for(done.wait(), timeout=5.0)
    except asyncio.TimeoutError:
        pass
    finally:
        await asyncio.to_thread(tr.close)

    return " ".join(final_text)


# ---------------------------------------------------------------------------
# Gemini LLM (streaming, via LangChain)
# ---------------------------------------------------------------------------


async def generate_reply(user_text: str, history: list[object]) -> tuple[str, list[object]]:
    """Call Gemini 2.5 Flash Lite and return the streamed full reply text."""
    from langchain.chat_models import init_chat_model  # noqa: PLC0415
    from langchain_core.messages import HumanMessage, SystemMessage  # noqa: PLC0415

    model = init_chat_model(
        model="gemini-2.5-flash-lite",
        model_provider="google_genai",
        api_key=GEMINI_KEY,
        temperature=0.7,
    )
    msgs = [SystemMessage(content=SYSTEM_PROMPT), *history, HumanMessage(content=user_text)]
    reply_chunks: list[str] = []
    async for chunk in model.astream(msgs):
        if hasattr(chunk, "content") and isinstance(chunk.content, str):
            reply_chunks.append(chunk.content)
    reply = "".join(reply_chunks)
    from langchain_core.messages import AIMessage  # noqa: PLC0415
    history = [*history, HumanMessage(content=user_text), AIMessage(content=reply)]
    return reply, history


# ---------------------------------------------------------------------------
# Gemini TTS
# ---------------------------------------------------------------------------


async def synthesise(text: str) -> bytes:
    """Convert text to PCM audio via Gemini TTS."""
    from google import genai  # noqa: PLC0415
    from google.genai import types  # noqa: PLC0415

    client = genai.Client(api_key=GEMINI_KEY)
    resp = await asyncio.to_thread(
        client.models.generate_content,
        model="gemini-2.5-flash-preview-tts",
        contents=text,
        config=types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Aoede")
                )
            ),
        ),
    )
    return _extract_audio(resp)


def _extract_audio(response: object) -> bytes:
    import base64  # noqa: PLC0415
    try:
        data = response.candidates[0].content.parts[0].inline_data.data  # type: ignore[union-attr]
        if isinstance(data, (bytes, bytearray)):
            return bytes(data)
        return base64.b64decode(data)
    except (AttributeError, IndexError):
        return b""


# ---------------------------------------------------------------------------
# Playback (non-blocking so barge-in detection can preempt it)
# ---------------------------------------------------------------------------


async def play_audio(pcm: bytes, cancel_event: asyncio.Event) -> None:
    """Stream PCM to the speakers. Returns early if cancel_event is set."""
    try:
        import sounddevice as sd  # noqa: PLC0415
        import numpy as np  # noqa: PLC0415
    except ImportError:
        print("sounddevice/numpy required for playback")
        return

    chunk_samples = int(TTS_SAMPLE_RATE * CHUNK_MS / 1000)
    chunk_bytes = chunk_samples * 2  # 16-bit

    stream = sd.RawOutputStream(
        samplerate=TTS_SAMPLE_RATE, channels=1, dtype="int16", blocksize=chunk_samples
    )
    with stream:
        for i in range(0, len(pcm), chunk_bytes):
            if cancel_event.is_set():
                print("[spike] Playback cancelled (barge-in)")
                break
            stream.write(pcm[i : i + chunk_bytes])
            await asyncio.sleep(CHUNK_MS / 1000)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


async def run_spike() -> None:
    gate = StubWakeGate()
    history: list[object] = []
    aec_self_trigger = 0
    turn = 0

    print("=" * 60)
    print("DaNa audio spike — ADR-0001 validation")
    print(f"STT: AssemblyAI | LLM: Gemini 2.5 Flash Lite | TTS: Gemini TTS")
    print("Speak after 'Listening…'. CTRL-C to stop.")
    print("=" * 60)

    while turn < MAX_TURNS:
        turn += 1
        stats = TurnStats(turn=turn)

        # ---- capture utterance ----
        stats.stt_start = time.perf_counter()
        pcm = await capture_utterance(gate)

        # ---- STT ----
        transcript = await transcribe(pcm)
        stats.stt_end = time.perf_counter()
        print(f"\n[STT {(stats.stt_end - stats.stt_start) * 1000:.0f} ms] \"{transcript}\"")

        if not transcript.strip():
            print("[spike] Empty transcript — skipping turn")
            continue

        # ---- LLM ----
        stats.llm_start = time.perf_counter()
        reply, history = await generate_reply(transcript, history)
        stats.llm_end = time.perf_counter()
        print(f"[LLM {(stats.llm_end - stats.llm_start) * 1000:.0f} ms] \"{reply}\"")

        # ---- TTS ----
        stats.tts_start = time.perf_counter()
        audio = await synthesise(reply)
        stats.tts_end = time.perf_counter()
        print(f"[TTS {(stats.tts_end - stats.tts_start) * 1000:.0f} ms] {len(audio)} bytes")

        stats.total_ms = (stats.tts_end - stats.stt_start) * 1000
        turn_log.append(stats)
        print(f"[TOTAL {stats.total_ms:.0f} ms]")

        # ---- playback with barge-in detection ----
        cancel = asyncio.Event()
        play_task = asyncio.create_task(play_audio(audio, cancel))

        # Simple barge-in: capture mic energy during playback; if above threshold, cancel
        async def _barge_in_watch() -> None:
            try:
                import sounddevice as sd  # noqa: PLC0415
                import numpy as np  # noqa: PLC0415
                loop = asyncio.get_event_loop()
                q: asyncio.Queue[bytes] = asyncio.Queue()

                def _cb(indata: object, frames: int, _t: object, _s: object) -> None:
                    loop.call_soon_threadsafe(q.put_nowait, bytes(indata))  # type: ignore[arg-type]

                with sd.RawInputStream(
                    samplerate=STT_SAMPLE_RATE, channels=1, dtype="int16",
                    blocksize=int(STT_SAMPLE_RATE * CHUNK_MS / 1000), callback=_cb
                ):
                    while not play_task.done():
                        try:
                            raw = await asyncio.wait_for(q.get(), timeout=0.1)
                            if _rms(raw) > SILENCE_THRESHOLD:
                                cancel.set()
                                return
                        except asyncio.TimeoutError:
                            continue
            except Exception:
                pass

        barge_task = asyncio.create_task(_barge_in_watch())
        await play_task
        barge_task.cancel()

    # ---- summary ----
    _print_summary()


def _print_summary() -> None:
    if not turn_log:
        print("No turns completed.")
        return
    totals = [t.total_ms for t in turn_log]
    stts = [(t.stt_end - t.stt_start) * 1000 for t in turn_log]
    llms = [(t.llm_end - t.llm_start) * 1000 for t in turn_log]
    ttss = [(t.tts_end - t.tts_start) * 1000 for t in turn_log]

    def med(lst: list[float]) -> float:
        s = sorted(lst)
        n = len(s)
        return (s[n // 2] + s[(n - 1) // 2]) / 2

    print("\n" + "=" * 60)
    print("ADR-0001 EXIT GATE — RESULTS")
    print("=" * 60)
    print(f"Turns completed: {len(turn_log)}")
    print(f"Median total latency: {med(totals):.0f} ms  (target ≤ 800 ms) {'✓' if med(totals) <= 800 else '✗'}")
    print(f"  STT: {med(stts):.0f} ms")
    print(f"  LLM: {med(llms):.0f} ms")
    print(f"  TTS: {med(ttss):.0f} ms")
    print()
    print("Self-triggering (AEC):")
    print("  [Manual verification needed — check if 'Listening…' fired during playback]")
    print()
    print("Barge-in:")
    print("  [Functional if playback was cut short when you spoke during TTS]")
    print("=" * 60)


if __name__ == "__main__":
    try:
        asyncio.run(run_spike())
    except KeyboardInterrupt:
        _print_summary()

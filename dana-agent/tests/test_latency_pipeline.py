"""Offline tests for the low-latency pipeline pieces (STT bridge, eager TTS
flush, TurnClock, warmup)."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any

import pytest
from pipecat.frames.frames import (
    InputAudioRawFrame,
    InterimTranscriptionFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection

from dana_agent.conversation.latency import TurnClock
from dana_agent.conversation.processors import (
    ProviderSTTProcessor,
    ProviderTTSProcessor,
    eager_split,
)
from dana_agent.conversation.warmup import warmup_stack
from dana_agent.types import AudioChunk, AudioFrame, Transcript

D = FrameDirection.DOWNSTREAM


def pcm(n: int = 3200, v: int = 0) -> InputAudioRawFrame:
    return InputAudioRawFrame(audio=bytes([v]) * n, num_channels=1, sample_rate=16_000)


class FakeSTT:
    def __init__(self, fail: bool = False, final: bool = True) -> None:
        self.calls: list[list[AudioFrame]] = []
        self.fail = fail
        self.final = final

    async def stream(self, audio: AsyncIterator[AudioFrame]) -> AsyncIterator[Transcript]:
        frames = [f async for f in audio]
        self.calls.append(frames)
        if self.fail:
            raise RuntimeError("boom")
        yield Transcript(text="xin", is_final=False, language="vi")
        if self.final:
            yield Transcript(text="xin chào", is_final=True, language="vi")


async def run(proc: Any, frames: list[Any]) -> list[Any]:
    out: list[Any] = []

    async def capture(frame: Any, direction: Any = D) -> None:
        out.append(frame)

    proc.push_frame = capture
    for f in frames:
        await proc.process_frame(f, D)
    if hasattr(proc, "wait_idle"):
        await proc.wait_idle()
    return out


async def test_stt_processor_transcribes_with_preroll() -> None:
    stt = FakeSTT()
    proc = ProviderSTTProcessor(stt)  # type: ignore[arg-type]
    # 5 pre-speech frames of 100 ms (3200 B); only ~300 ms should be kept.
    pre = [pcm(v=i + 1) for i in range(5)]
    during = pcm(v=9)
    out = await run(
        proc,
        [*pre, VADUserStartedSpeakingFrame(), during, VADUserStoppedSpeakingFrame()],
    )
    assert len(stt.calls) == 1
    got = stt.calls[0]
    assert got[0].sample_rate == 16_000
    # preroll trimmed to <= 300 ms: 3 frames (values 3,4,5) + speech frame
    assert [f.pcm[0] for f in got] == [3, 4, 5, 9]
    finals = [f for f in out if isinstance(f, TranscriptionFrame)]
    interims = [f for f in out if isinstance(f, InterimTranscriptionFrame)]
    assert [f.text for f in finals] == ["xin chào"]
    assert [f.text for f in interims] == ["xin"]
    # pass-through of everything we sent
    assert sum(isinstance(f, InputAudioRawFrame) for f in out) == 6


async def test_stt_processor_partial_fallback_and_error() -> None:
    proc = ProviderSTTProcessor(FakeSTT(final=False))  # type: ignore[arg-type]
    out = await run(proc, [VADUserStartedSpeakingFrame(), pcm(), VADUserStoppedSpeakingFrame()])
    assert [f.text for f in out if type(f) is TranscriptionFrame] == ["xin"]

    bad = ProviderSTTProcessor(FakeSTT(fail=True))  # type: ignore[arg-type]
    out = await run(bad, [VADUserStartedSpeakingFrame(), pcm(), VADUserStoppedSpeakingFrame()])
    assert not [f for f in out if isinstance(f, TranscriptionFrame)]
    assert any(isinstance(f, VADUserStoppedSpeakingFrame) for f in out)  # still alive


async def test_stt_processor_caps_utterance() -> None:
    stt = FakeSTT()
    proc = ProviderSTTProcessor(stt, max_utterance_s=0.5)  # type: ignore[arg-type]
    frames = [pcm() for _ in range(20)]  # 2 s of audio
    await run(proc, [VADUserStartedSpeakingFrame(), *frames, VADUserStoppedSpeakingFrame()])
    assert sum(len(f.pcm) for f in stt.calls[0]) <= 16_000


async def test_stt_processor_uses_clock() -> None:
    marks: list[str] = []
    clock = SimpleNamespace(mark_stt_final=lambda: marks.append("stt"))
    proc = ProviderSTTProcessor(FakeSTT(), clock=clock)  # type: ignore[arg-type]
    await run(proc, [VADUserStartedSpeakingFrame(), pcm(), VADUserStoppedSpeakingFrame()])
    assert marks == ["stt"]


# --- eager flush ---------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "min_chars", "expected"),
    [
        ("Sure, I can do that. Next", 20, "Sure, I can do that."),
        ("Hello there, my friend, how are you", 10, "Hello there,"),
        ("Hi. More", 20, "Hi."),
        (
            "Được rồi, tôi sẽ mở trình duyệt ngay bây giờ. Xong",
            10,
            "Được rồi, tôi sẽ mở trình duyệt ngay bây giờ.",
        ),
        ("Giá là 3,5 triệu đồng, được không", 20, "Giá là 3,5 triệu đồng,"),
        ("Pi is 1.5 roughly", 5, None),
        ("Total 3,", 5, None),  # trailing number-comma held
        ("no punctuation here at all", 5, None),
    ],
)
def test_eager_split(text: str, min_chars: int, expected: str | None) -> None:
    cut = eager_split(text, min_chars)
    if expected is None:
        assert cut == -1
    else:
        assert text[:cut].strip() == expected


class RecTTS:
    def __init__(self) -> None:
        self.chunks: list[str] = []

    async def stream(self, text: AsyncIterator[str]) -> AsyncIterator[AudioChunk]:
        async for t in text:
            self.chunks.append(t)
            yield AudioChunk(pcm=b"\x00\x00", sample_rate=24_000)


async def _tts(tokens: list[str], **kw: Any) -> list[str]:
    tts = RecTTS()
    proc = ProviderTTSProcessor(tts, **kw)  # type: ignore[arg-type]
    await run(
        proc,
        [
            LLMFullResponseStartFrame(),
            *[LLMTextFrame(text=t) for t in tokens],
            LLMFullResponseEndFrame(),
        ],
    )
    return tts.chunks


async def test_eager_first_chunk_then_normal() -> None:
    toks = ["Sure", ",", " I", " can", " open", " it", " now", ".", " Done", ".", " Bye", "."]
    chunks = await _tts(toks, chunk_sentences=1, first_chunk_min_chars=20)
    # first clause "Sure, I can open it now." (>=20 chars only at sentence end)
    assert chunks[0] == "Sure, I can open it now."
    assert chunks[1:] == ["Done.", "Bye."]


async def test_eager_early_clause_flush() -> None:
    toks = ["Được", " rồi", " thưa", " sếp", ",", " tôi", " làm", " ngay", "."]
    chunks = await _tts(toks, chunk_sentences=1, first_chunk_min_chars=10)
    assert chunks[0] == "Được rồi thưa sếp,"
    assert chunks[1] == "tôi làm ngay."


async def test_eager_does_not_split_numbers() -> None:
    chunks = await _tts(["Giá", " 3", ",", "5", " triệu", "."], first_chunk_min_chars=3)
    assert chunks == ["Giá 3,5 triệu."]


async def test_eager_off() -> None:
    toks = ["Sure", ",", " fine", ".", " Ok", "."]
    chunks = await _tts(toks, chunk_sentences=0, first_chunk_min_chars=0)
    assert chunks == ["Sure, fine. Ok."]


# --- TurnClock -----------------------------------------------------------


def test_turn_clock_math(caplog: pytest.LogCaptureFixture) -> None:
    from structlog.testing import capture_logs

    t = [0.0]
    clock = TurnClock(600, clock=lambda: t[0])
    with capture_logs() as logs:
        clock.mark_end_of_speech()
        t[0] = 0.15
        clock.mark_stt_final()
        clock.mark_stt_final()  # idempotent
        t[0] = 0.35
        clock.mark_llm_first_token()
        t[0] = 0.70
        clock.mark_tts_first_audio()
        clock.mark_tts_first_audio()
    lat = {e["stage"]: e for e in logs if e["event"] == "latency"}
    assert lat["stt"]["ms"] == 150
    assert lat["llm_ttft"]["ms"] == 200
    assert lat["tts_ttfb"]["ms"] == 350
    assert lat["voice_to_first_audio"]["ms"] == 700
    assert lat["voice_to_first_audio"]["over_budget"] is True
    assert len({e["turn_id"] for e in lat.values()}) == 1
    assert len([e for e in logs if e.get("stage") == "stt"]) == 1


def test_turn_clock_typed_turn_and_vad_delay() -> None:
    from structlog.testing import capture_logs

    t = [10.0]
    clock = TurnClock(600, clock=lambda: t[0], vad_delay_ms=200)
    with capture_logs() as logs:
        clock.mark_llm_first_token()  # no turn yet: ignored
        clock.mark_stt_final()  # typed turn starts here
        t[0] = 10.3
        clock.mark_llm_first_token()
        t[0] = 10.4
        clock.mark_tts_first_audio()
    lat = {e["stage"]: e for e in logs if e["event"] == "latency"}
    assert lat["stt"]["ms"] == 0
    assert lat["voice_to_first_audio"]["ms"] == 400
    assert lat["voice_to_first_audio"]["over_budget"] is False

    t[0] = 20.0
    with capture_logs() as logs:
        clock.mark_end_of_speech()  # eos shifted back 200 ms
        t[0] = 20.1
        clock.mark_stt_final()
    assert next(e for e in logs if e["stage"] == "stt")["ms"] == 300


# --- warmup --------------------------------------------------------------


class Warm:
    def __init__(self, fail: bool = False) -> None:
        self.called = False
        self.fail = fail

    async def warmup(self) -> None:
        self.called = True
        if self.fail:
            raise RuntimeError("no model")


async def test_warmup_stack_isolates_failures() -> None:
    stt, tts = Warm(fail=True), Warm()
    stack = SimpleNamespace(stt=stt, tts=tts, s2s=None, brain=None)
    res = await warmup_stack(stack, SimpleNamespace())  # type: ignore[arg-type]
    assert stt.called and tts.called
    assert res["stt"] == -1.0
    assert res["tts"] >= 0.0


async def test_warmup_stack_noop_and_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    assert await warmup_stack(SimpleNamespace(stt=object(), brain=None), SimpleNamespace()) == {}  # type: ignore[arg-type]
    import sys
    import types

    seen: list[object] = []

    async def fake_warm(model: object) -> float:
        seen.append(model)
        return 1.0

    mod = types.ModuleType("dana_agent.brain.model")
    mod.warmup_chat_model = fake_warm  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "dana_agent.brain.model", mod)
    brain = SimpleNamespace(_model="M")
    res = await warmup_stack(SimpleNamespace(stt=None, brain=brain), SimpleNamespace())  # type: ignore[arg-type]
    assert seen == ["M"] and "llm" in res
    _ = asyncio  # keep import used


# --- speculative STT + keepalive ----------------------------------------------


def _loud(ms: int = 100) -> InputAudioRawFrame:
    import struct

    n = 16 * ms
    return InputAudioRawFrame(
        audio=struct.pack(f"<{n}h", *([3000] * n)), num_channels=1, sample_rate=16_000
    )


def _quiet(ms: int = 40) -> InputAudioRawFrame:
    return InputAudioRawFrame(audio=b"\x00\x00" * (16 * ms), num_channels=1, sample_rate=16_000)


class CountingSTT(FakeSTT):
    async def stream(self, audio: AsyncIterator[AudioFrame]) -> AsyncIterator[Transcript]:
        async for tr in FakeSTT.stream(self, audio):
            yield tr


async def _spec_run(stt: FakeSTT, frames: list[Any], **kw: Any) -> list[Any]:
    proc = ProviderSTTProcessor(stt, speculative=True, **kw)  # type: ignore[arg-type]
    out: list[Any] = []

    async def capture(frame: Any, direction: Any = D) -> None:
        out.append(frame)

    proc.push_frame = capture
    for f in frames:
        await proc.process_frame(f, D)
        await asyncio.sleep(0)  # let a speculative task start
    await proc.wait_idle()
    return out


async def test_speculative_reuse_runs_provider_once() -> None:
    stt = FakeSTT()
    out = await _spec_run(
        stt,
        [
            VADUserStartedSpeakingFrame(),
            _loud(),
            _quiet(40),
            _quiet(40),
            VADUserStoppedSpeakingFrame(),
        ],
    )
    assert len(stt.calls) == 1  # speculative result reused, not re-run
    assert [f.text for f in out if type(f) is TranscriptionFrame] == ["xin chào"]


async def test_speculative_cancelled_on_resume() -> None:
    stt = FakeSTT()
    out = await _spec_run(
        stt,
        [
            VADUserStartedSpeakingFrame(),
            _loud(),
            _quiet(80),  # speculation starts
            _loud(),  # speech resumes -> discard
            VADUserStoppedSpeakingFrame(),
        ],
    )
    # the final run saw the whole utterance (3 audio frames, not the 2-frame speculative snapshot)
    assert len(stt.calls[-1]) == 3
    assert [f.text for f in out if type(f) is TranscriptionFrame] == ["xin chào"]


async def test_speculative_disabled_and_error_fallback() -> None:
    stt = FakeSTT()
    proc = ProviderSTTProcessor(stt)  # type: ignore[arg-type]
    await run(
        proc, [VADUserStartedSpeakingFrame(), _loud(), _quiet(80), VADUserStoppedSpeakingFrame()]
    )
    assert len(stt.calls) == 1

    class FlakySTT(FakeSTT):
        async def stream(self, audio: AsyncIterator[AudioFrame]) -> AsyncIterator[Transcript]:
            frames = [f async for f in audio]
            self.calls.append(frames)
            if len(self.calls) == 1:
                raise RuntimeError("spec failed")
            yield Transcript(text="ok", is_final=True)

    flaky = FlakySTT()
    out = await _spec_run(
        flaky, [VADUserStartedSpeakingFrame(), _loud(), _quiet(80), VADUserStoppedSpeakingFrame()]
    )
    assert len(flaky.calls) == 2  # speculative failed -> normal run
    assert [f.text for f in out if type(f) is TranscriptionFrame] == ["ok"]


async def test_keepalive_ping_uses_models_list_and_fallbacks() -> None:
    from dana_agent.conversation.warmup import keepalive_ping

    calls: list[str] = []

    class Models:
        async def list(self) -> None:
            calls.append("list")

    model = SimpleNamespace(root_async_client=SimpleNamespace(models=Models()))
    assert await keepalive_ping(model) == "ping" and calls == ["list"]

    class Boom:
        async def list(self) -> None:
            raise RuntimeError("net")

    bad = SimpleNamespace(root_async_client=SimpleNamespace(models=Boom()))
    assert await keepalive_ping(bad) == "failed"


async def test_keepalive_loop_pings_and_cancels() -> None:
    from dana_agent.conversation.warmup import keepalive_loop

    n = {"c": 0}

    class Models:
        async def list(self) -> None:
            n["c"] += 1

    brain = SimpleNamespace(
        model=SimpleNamespace(root_async_client=SimpleNamespace(models=Models()))
    )
    task = asyncio.ensure_future(keepalive_loop(SimpleNamespace(brain=brain), 0.01))  # type: ignore[arg-type]
    await asyncio.sleep(0.08)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert n["c"] >= 2


async def _commit_run(stt: FakeSTT, frames: list[Any], commit_ms: int) -> tuple[list[Any], Any]:
    proc = ProviderSTTProcessor(stt, speculative=True, early_commit_ms=commit_ms)  # type: ignore[arg-type]
    out: list[Any] = []

    async def capture(frame: Any, direction: Any = D) -> None:
        out.append(frame)

    proc.push_frame = capture
    for f in frames:
        await proc.process_frame(f, D)
        for _ in range(3):
            await asyncio.sleep(0)
    await proc.wait_idle()
    return out, proc


async def test_early_commit_emits_before_vad_stop_and_ignores_stop() -> None:
    stt = FakeSTT()
    frames: list[Any] = [VADUserStartedSpeakingFrame(), _loud()]
    frames += [_quiet(40) for _ in range(5)]  # 200 ms silence, VAD stop not yet sent
    out, proc = await _commit_run(stt, frames, commit_ms=120)
    assert [f.text for f in out if type(f) is TranscriptionFrame] == ["xin chào"]
    # the later VAD stop must not re-run STT or emit a second transcript
    out2 = []

    async def cap(frame: Any, direction: Any = D) -> None:
        out2.append(frame)

    proc.push_frame = cap
    await proc.process_frame(VADUserStoppedSpeakingFrame(), D)
    await proc.wait_idle()
    assert len(stt.calls) == 1
    assert not [f for f in out2 if isinstance(f, TranscriptionFrame)]


async def test_early_commit_off_waits_for_vad_stop() -> None:
    stt = FakeSTT()
    frames: list[Any] = [VADUserStartedSpeakingFrame(), _loud()]
    frames += [_quiet(40) for _ in range(5)]
    out, _ = await _commit_run(stt, frames, commit_ms=0)
    assert not [f for f in out if type(f) is TranscriptionFrame]


async def test_early_commit_resume_is_new_turn() -> None:
    stt = FakeSTT()
    frames: list[Any] = [VADUserStartedSpeakingFrame(), _loud()]
    frames += [_quiet(40) for _ in range(5)]
    frames += [VADUserStoppedSpeakingFrame(), VADUserStartedSpeakingFrame(), _loud()]
    frames += [VADUserStoppedSpeakingFrame()]
    out, _ = await _commit_run(stt, frames, commit_ms=120)
    assert len([f for f in out if type(f) is TranscriptionFrame]) == 2
    assert len(stt.calls) == 2


async def test_early_commit_not_before_silence_threshold() -> None:
    stt = FakeSTT()
    frames: list[Any] = [VADUserStartedSpeakingFrame(), _loud(), _quiet(40), _quiet(40)]  # 80 ms
    out, _ = await _commit_run(stt, frames, commit_ms=120)
    assert not [f for f in out if type(f) is TranscriptionFrame]


# --- reviewer probes (regressions) ---
from pipecat.frames.frames import CancelFrame  # noqa: E402


def _pl(n: int = 3200) -> InputAudioRawFrame:
    return InputAudioRawFrame(audio=(b"\xe8\x03" * (n // 2)), num_channels=1, sample_rate=16_000)


def _pq(n: int = 3200) -> InputAudioRawFrame:
    return pcm(n)


class SlowSTT:  # noqa: D101
    def __init__(s):
        s.started = 0
        s.cancelled = 0

    async def stream(s, audio):
        s.started += 1
        _ = [f async for f in audio]
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            s.cancelled += 1
            raise
        yield Transcript(text="x", is_final=True)


async def test_spec_leak_on_cancel() -> None:
    stt = SlowSTT()
    p = ProviderSTTProcessor(stt, speculative=True, onset_ms=60)
    await run(p, [VADUserStartedSpeakingFrame(), _pl(), _pq(), _pq()])
    await asyncio.sleep(0.05)
    assert p._spec_task is not None
    await run(p, [CancelFrame()])
    await asyncio.sleep(0.05)
    assert stt.cancelled == 1, "spec task still running after CancelFrame"


async def test_early_commit_breaks_clock() -> None:
    stt = FakeSTT()
    clk = TurnClock(clock=lambda: 0.0)
    p = ProviderSTTProcessor(stt, speculative=True, onset_ms=60, early_commit_ms=150, clock=clk)
    p.push_frame = lambda *a, **k: asyncio.sleep(0)
    for f in [VADUserStartedSpeakingFrame(), _pl(), _pq(), _pq()]:
        await p.process_frame(f, D)
    await asyncio.sleep(0.05)
    await p.process_frame(_pq(), D)  # triggers commit
    await p.wait_idle()
    clk.mark_end_of_speech()  # VAD stop arrives later (observer)
    clk.mark_llm_first_token()
    clk.mark_tts_first_audio()
    assert clk._audio is not None, (
        "voice_to_first_audio never emitted after early commit (stt mark wiped by later VAD stop)"
    )


async def test_resumed_speech_after_early_commit_lost() -> None:
    stt = FakeSTT()
    p = ProviderSTTProcessor(stt, speculative=True, onset_ms=60, early_commit_ms=150)
    out = []

    async def cap(f, d=D):
        out.append(f)

    p.push_frame = cap
    for f in [VADUserStartedSpeakingFrame(), _pl(), _pq(), _pq()]:
        await p.process_frame(f, D)
    await asyncio.sleep(0.05)
    await p.process_frame(_pq(), D)
    await p.wait_idle()
    n = len(stt.calls)
    # user keeps talking; VAD never fired stop so no new Started
    for f in [_pl(), _pl(), VADUserStoppedSpeakingFrame()]:
        await p.process_frame(f, D)
    await p.wait_idle()
    assert len(stt.calls) == n + 1, "continued speech after early commit dropped"


async def test_tts_double_start_leaks() -> None:
    class P:
        async def stream(s, t):
            async for _x in t:
                yield AudioChunk(pcm=b"\0\0", sample_rate=24000)

    p = ProviderTTSProcessor(P())
    out = []

    async def cap(f, d=D):
        out.append(f)

    p.push_frame = cap
    await p.process_frame(LLMFullResponseStartFrame(), D)
    t1 = p._tts_task
    await p.process_frame(LLMTextFrame(text="hi."), D)
    await p.process_frame(LLMFullResponseStartFrame(), D)  # overlapping turn w/o End
    await asyncio.sleep(0.05)
    assert t1.done(), "first TTS task orphaned/never ended"
    p._tts_task.cancel()

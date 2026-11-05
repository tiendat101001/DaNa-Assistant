"""True end-to-end latency: last speech frame -> first reply audio.

Feeds a real wav as 20 ms ``InputAudioRawFrame``s plus synthetic VAD start/stop
frames (stop = ``--vad-stop-ms`` after the last speech frame, silence frames fed
in real time) through the REAL chain with no audio devices::

    ProviderSTTProcessor(LiteRT) -> BrainProcessor(real Brain, tools bound)
        -> ProviderTTSProcessor(LiteRT KittenTTS)

and prints a markdown table (median of N warm runs) with the stage breakdown.
LLM creds come from the environment only (TG_API_KEY / TG_BASE_URL / TG_MODEL_ID).

    uv run python scripts/e2e_latency.py [--runs 3] [--cases a,b,c]

Cases: moonshine, moonshine_spec, moonshine_spec_commit, ctc_spec_commit (needs
DANA_TEST_CTC_MODEL / DANA_TEST_CTC_NAME).
"""

from __future__ import annotations

import argparse
import asyncio
import os
import statistics
import time
import wave
from typing import Any

from pipecat.frames.frames import (
    InputAudioRawFrame,
    LLMTextFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection

import dana_agent.providers  # noqa: F401  register providers

_ROOT = "/tmp/claude-0"  # noqa: S108
D = FrameDirection.DOWNSTREAM


def _env(n: str, d: str) -> str:
    return os.environ.get(n) or d


def _cases() -> dict[str, dict[str, Any]]:
    moon = _env(
        "DANA_TEST_MOONSHINE", f"{_ROOT}/hf_models/moonshine-tiny/moonshine_tiny_5s_i8.tflite"
    )
    cases: dict[str, dict[str, Any]] = {
        "moonshine": {"name": "moonshine-tiny", "path": moon, "spec": False, "commit": 0},
        "moonshine_spec": {"name": "moonshine-tiny", "path": moon, "spec": True, "commit": 0},
        "moonshine_spec_commit": {
            "name": "moonshine-tiny",
            "path": moon,
            "spec": True,
            "commit": 120,
        },
    }
    ctc = os.environ.get("DANA_TEST_CTC_MODEL")
    if ctc:
        cases["ctc_spec_commit"] = {
            "name": _env("DANA_TEST_CTC_NAME", "ctc"),
            "path": ctc,
            "spec": True,
            "commit": 120,
        }
    return cases


def _load_wav(path: str) -> tuple[bytes, int]:
    with wave.open(path, "rb") as w:
        return w.readframes(w.getnframes()), w.getframerate()


def _build_brain() -> Any:
    from dana_agent.actions.gate import ActionGate  # noqa: PLC0415
    from dana_agent.brain import Brain, build_chat_model  # noqa: PLC0415
    from dana_agent.brain.tools import build_action_tools  # noqa: PLC0415
    from dana_agent.config import ActionsConfig, AppConfig  # noqa: PLC0415
    from dana_agent.providers.registry import build_actions  # noqa: PLC0415

    cfg = AppConfig.model_validate(
        {
            "conversation": {
                "llm": {
                    "provider": "together",
                    "max_tokens": 80,
                    "extra": {"extra_body": {"reasoning": {"enabled": False}}},
                }
            },
            "brain": {"memory": {"backend": "memory"}},
        }
    )
    actions = ActionsConfig(
        provider="browser_os",
        os_actions_enabled=True,
        allowed_domains=["youtube.com", "google.com", "duckduckgo.com", "wikipedia.org"],
    )
    gate = ActionGate(build_actions(ActionsConfig(provider="mock")))
    tools = build_action_tools(actions, gate)
    brain = Brain(
        build_chat_model(cfg.conversation.llm),
        gate,
        system_prompt=cfg.brain.system_prompt,
        tools=tools,
        max_steps=cfg.brain.max_steps,
    )
    return brain, len(tools), cfg.conversation.llm.model


async def run_case(
    case: dict[str, Any],
    brain: Any,
    tts: Any,
    pcm: bytes,
    rate: int,
    runs: int,
    vad_stop_ms: int,
    first_chunk: int,
) -> dict[str, Any]:
    from dana_agent.config import STTConfig  # noqa: PLC0415
    from dana_agent.conversation.processors import (  # noqa: PLC0415
        BrainProcessor,
        ProviderSTTProcessor,
        ProviderTTSProcessor,
    )
    from dana_agent.providers.stt.litert import LiteRTWhisperSTT  # noqa: PLC0415

    stt = LiteRTWhisperSTT(
        STTConfig(
            provider="litert", model=case["name"], model_path=case["path"], language_code="en"
        ),
        models_dir=_env("DANA_TEST_MODELS_DIR", f"{_ROOT}/hf_models"),
    )
    await stt.warmup()

    step = rate // 50 * 2  # 20 ms
    speech = [pcm[i : i + step] for i in range(0, len(pcm) - step + 1, step)]
    silence = b"\x00\x00" * (rate // 50)
    n_sil = vad_stop_ms // 20

    results: list[dict[str, float]] = []
    for run in range(runs + 1):  # first is an extra warm-up, discarded
        stt_p = ProviderSTTProcessor(
            stt, sample_rate=rate, speculative=case["spec"], early_commit_ms=case["commit"]
        )
        brain_p = BrainProcessor(brain, session_id=f"e2e-{case['name']}-{run}-{time.time_ns()}")
        tts_p = ProviderTTSProcessor(tts, chunk_sentences=1, first_chunk_min_chars=first_chunk)
        procs = [stt_p, brain_p, tts_p]
        t: dict[str, float] = {}
        first_audio = asyncio.Event()

        def make_push(
            i: int, t: dict[str, float], first_audio: asyncio.Event, procs: list[Any]
        ) -> Any:
            async def push(frame: Any, direction: Any = D) -> None:
                now = time.perf_counter()
                if isinstance(frame, TranscriptionFrame):
                    t.setdefault("stt_final", now)
                elif isinstance(frame, LLMTextFrame):
                    t.setdefault("llm_first", now)
                elif isinstance(frame, TTSAudioRawFrame):
                    t.setdefault("tts_first", now)
                    first_audio.set()
                if i + 1 < len(procs):
                    await procs[i + 1].process_frame(frame, direction)

            return push

        for i, p in enumerate(procs):
            p.push_frame = make_push(i, t, first_audio, procs)

        await stt_p.process_frame(VADUserStartedSpeakingFrame(), D)
        for chunk in speech:
            await stt_p.process_frame(
                InputAudioRawFrame(audio=chunk, num_channels=1, sample_rate=rate), D
            )
            await asyncio.sleep(0)
        t["last_speech"] = time.perf_counter()
        for _ in range(n_sil):  # real-time silence until the (synthetic) VAD stop
            await asyncio.sleep(0.02)
            await stt_p.process_frame(
                InputAudioRawFrame(audio=silence, num_channels=1, sample_rate=rate), D
            )
        t["vad_stop"] = time.perf_counter()
        await stt_p.process_frame(VADUserStoppedSpeakingFrame(), D)
        try:
            await asyncio.wait_for(first_audio.wait(), timeout=30)
        except TimeoutError:
            pass
        await brain_p._cancel_turn()
        await tts_p._cancel_tts()
        await stt_p.wait_idle(2.0)
        if run == 0 or "tts_first" not in t:
            continue
        ls = t["last_speech"]
        results.append(
            {
                "total": (t["tts_first"] - ls) * 1000,
                "stt": (t["stt_final"] - ls) * 1000,  # from last speech frame
                "stt_after_stop": (t["stt_final"] - t["vad_stop"]) * 1000,
                "llm_ttft": (t.get("llm_first", t["tts_first"]) - t["stt_final"]) * 1000,
                "tts": (t["tts_first"] - t.get("llm_first", t["stt_final"])) * 1000,
            }
        )
    med = {k: statistics.median(r[k] for r in results) for k in results[0]} if results else {}
    med["n"] = len(results)
    return med


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--cases", default="")
    ap.add_argument("--wav", default=_env("DANA_TEST_WAV", f"{_ROOT}/en.wav"))
    ap.add_argument(
        "--kitten", default=_env("DANA_TEST_KITTEN", f"{_ROOT}/hf_models/kitten-tts-nano-0.8")
    )
    ap.add_argument("--vad-stop-ms", type=int, default=200)
    ap.add_argument("--first-chunk", type=int, default=16)
    a = ap.parse_args()

    from dana_agent.config import TTSConfig  # noqa: PLC0415
    from dana_agent.providers.registry import build_tts  # noqa: PLC0415

    all_cases = _cases()
    names = [c for c in a.cases.split(",") if c] or list(all_cases)
    pcm, rate = _load_wav(a.wav)
    brain, n_tools, model = _build_brain()
    tts = build_tts(TTSConfig(provider="litert", model_path=a.kitten))
    await tts.warmup()  # type: ignore[attr-defined]

    print(
        f"LLM {model}, {n_tools} tools, VAD stop {a.vad_stop_ms} ms, "
        f"first_chunk_min_chars {a.first_chunk}, median of {a.runs} warm runs\n"
    )
    print(
        "| case | last speech -> first audio | STT (from last speech) | STT after VAD stop "
        "| LLM TTFT | TTS (token -> audio) | n |\n|---|---|---|---|---|---|---|"
    )
    for n in names:
        if n not in all_cases:
            print(f"| {n} | not configured | | | | | |")
            continue
        try:
            m = await run_case(
                all_cases[n], brain, tts, pcm, rate, a.runs, a.vad_stop_ms, a.first_chunk
            )
        except Exception as exc:  # noqa: BLE001
            print(f"| {n} | FAILED {type(exc).__name__}: {str(exc)[:60]} | | | | | |")
            continue
        if not m or m.get("n") == 0:
            print(f"| {n} | no audio produced | | | | | 0 |")
            continue
        print(
            f"| {n} | **{m['total']:.0f} ms** | {m['stt']:.0f} | {m['stt_after_stop']:.0f} "
            f"| {m['llm_ttft']:.0f} | {m['tts']:.0f} | {m['n']:.0f} |"
        )


if __name__ == "__main__":
    asyncio.run(main())

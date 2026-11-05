"""Latency bench on real components -> markdown table (never prints secrets).

Measures warm, N runs each:
  (a) STT: LiteRT Moonshine-tiny and Whisper-tiny on a wav
  (b) LLM TTFT through the real Brain with the full action tool set bound
      (+ the plain ``probe_llm`` number)
  (c) TTS: LiteRT KittenTTS first chunk for a ~20-char first clause
  (d) projected voice_to_first_audio = vad_stop + STT + LLM TTFT + TTS vs budget

Run (from dana-agent/; LLM creds come from the environment only, TG_API_KEY /
TG_BASE_URL / TG_MODEL_ID):

    uv run python scripts/latency_bench.py [--runs 3] [--skip llm,tts,stt]

Paths: args or env (DANA_TEST_WAV, DANA_TEST_MOONSHINE, DANA_TEST_WHISPER,
DANA_TEST_KITTEN, DANA_TEST_MODELS_DIR -- the dir holding tokenizer folders).
"""

from __future__ import annotations

import argparse
import asyncio
import os
import statistics
import time
import wave
from collections.abc import AsyncIterator

import dana_agent.providers  # noqa: F401  register providers
from dana_agent.types import AudioFrame

_ROOT = "/tmp/claude-0"  # noqa: S108
FIRST_CLAUSE = "Sure, I can open that"  # ~20 chars
PROMPT = "Hi DaNa, what can you help me with today?"


def _env(name: str, default: str) -> str:
    return os.environ.get(name) or default


def _med(xs: list[float]) -> float:
    return statistics.median(xs) if xs else float("nan")


def _fmt(xs: list[float]) -> str:
    if not xs:
        return "n/a"
    return f"{_med(xs):.0f} (min {min(xs):.0f} / max {max(xs):.0f})"


async def _wav_frames(path: str) -> list[AudioFrame]:
    with wave.open(path, "rb") as w:
        rate, ch, data = w.getframerate(), w.getnchannels(), w.readframes(w.getnframes())
    step = rate * 2 * ch // 10  # 100 ms
    return [
        AudioFrame(pcm=data[i : i + step], sample_rate=rate, channels=ch)
        for i in range(0, len(data), step)
    ]


async def bench_stt(label: str, model_file: str, name: str, wav: str, models_dir: str, runs: int):
    from dana_agent.config import STTConfig  # noqa: PLC0415
    from dana_agent.providers.stt.litert import LiteRTWhisperSTT  # noqa: PLC0415

    frames = await _wav_frames(wav)
    audio_s = sum(len(f.pcm) for f in frames) / (frames[0].sample_rate * 2 * frames[0].channels)
    cfg = STTConfig(provider="litert", model=name, model_path=model_file, language_code="en")
    stt = LiteRTWhisperSTT(cfg, models_dir=models_dir)
    t0 = time.perf_counter()
    await stt.warmup()
    load_ms = (time.perf_counter() - t0) * 1000

    async def src() -> AsyncIterator[AudioFrame]:
        for f in frames:
            yield f

    times: list[float] = []
    text = ""
    for _ in range(runs):
        t = time.perf_counter()
        async for tr in stt.stream(src()):
            if tr.is_final:
                text = tr.text
        times.append((time.perf_counter() - t) * 1000)
    return {"label": label, "ms": times, "load_ms": load_ms, "audio_s": audio_s, "text": text}


async def bench_llm(runs: int):
    from dana_agent.actions.gate import ActionGate  # noqa: PLC0415
    from dana_agent.brain import Brain, build_chat_model  # noqa: PLC0415
    from dana_agent.brain.events import TokenDelta  # noqa: PLC0415
    from dana_agent.brain.probe import probe_llm  # noqa: PLC0415
    from dana_agent.brain.tools import build_action_tools  # noqa: PLC0415
    from dana_agent.config import (
        ActionsConfig,  # noqa: PLC0415
        AppConfig,  # noqa: PLC0415
    )
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
    # Real model + the full gated browser+OS tool surface (schemas bound, nothing
    # executes -- same construction as scripts/action_routing_eval.py).
    actions_cfg = ActionsConfig(
        provider="browser_os",
        os_actions_enabled=True,
        allowed_domains=["youtube.com", "google.com", "duckduckgo.com", "wikipedia.org"],
    )
    gate = ActionGate(build_actions(ActionsConfig(provider="mock")))
    tools = build_action_tools(actions_cfg, gate)
    brain = Brain(
        build_chat_model(cfg.conversation.llm),
        gate,
        system_prompt=cfg.brain.system_prompt,
        tools=tools,
        max_steps=cfg.brain.max_steps,
    )
    n_tools = len(getattr(brain, "_tools", []))
    out: dict[str, object] = {"model": cfg.conversation.llm.model, "tools": n_tools}

    async def one(i: int) -> float | None:
        t = time.perf_counter()
        async for ev in brain.run_turn(PROMPT, thread_id=f"bench-{i}-{time.time_ns()}"):
            if isinstance(ev, TokenDelta) and ev.text:
                return (time.perf_counter() - t) * 1000
        return None

    if True:
        await one(-1)  # warm connection / model
        brain_ms = [m for m in [await one(i) for i in range(runs)] if m is not None]
        await probe_llm(cfg.conversation.llm)  # warm the plain path too
        probe_ms = []
        for _ in range(runs):
            r = await probe_llm(cfg.conversation.llm)
            if r.get("ok") and r.get("ttft_ms") is not None:
                probe_ms.append(float(r["ttft_ms"]))
        out.update(brain_ms=brain_ms, probe_ms=probe_ms)
    return out


async def bench_tts(kitten: str, runs: int):
    from dana_agent.config import TTSConfig  # noqa: PLC0415
    from dana_agent.providers.registry import build_tts  # noqa: PLC0415

    tts = build_tts(TTSConfig(provider="litert", model_path=kitten))
    t0 = time.perf_counter()
    await tts.warmup()  # type: ignore[attr-defined]
    load_ms = (time.perf_counter() - t0) * 1000

    async def one() -> float:
        async def text() -> AsyncIterator[str]:
            yield FIRST_CLAUSE

        t = time.perf_counter()
        async for _ in tts.stream(text()):
            return (time.perf_counter() - t) * 1000
        return float("nan")

    return {"ms": [await one() for _ in range(runs)], "load_ms": load_ms}


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--skip", default="", help="comma list: stt,llm,tts")
    ap.add_argument("--wav", default=_env("DANA_TEST_WAV", f"{_ROOT}/en.wav"))
    ap.add_argument("--models-dir", default=_env("DANA_TEST_MODELS_DIR", f"{_ROOT}/hf_models"))
    ap.add_argument(
        "--moonshine",
        default=_env(
            "DANA_TEST_MOONSHINE", f"{_ROOT}/hf_models/moonshine-tiny/moonshine_tiny_5s_i8.tflite"
        ),
    )
    ap.add_argument(
        "--whisper",
        default=_env(
            "DANA_TEST_WHISPER", f"{_ROOT}/hf_models/whisper-tiny/whisper_tiny_30s_i8.tflite"
        ),
    )
    ap.add_argument(
        "--kitten", default=_env("DANA_TEST_KITTEN", f"{_ROOT}/hf_models/kitten-tts-nano-0.8")
    )
    ap.add_argument("--vad-stop-ms", type=float, default=200)
    ap.add_argument("--budget-ms", type=float, default=600)
    a = ap.parse_args()
    skip = {s.strip() for s in a.skip.split(",") if s.strip()}
    rows: list[tuple[str, str, str]] = []
    best: dict[str, float] = {}
    notes: list[str] = []

    if "stt" not in skip:
        for label, f, name in (
            ("STT Moonshine-tiny (LiteRT i8)", a.moonshine, "moonshine-tiny"),
            ("STT Whisper-tiny (LiteRT i8)", a.whisper, "whisper-tiny"),
        ):
            try:
                r = await bench_stt(label, f, name, a.wav, a.models_dir, a.runs)
                rows.append(
                    (
                        label,
                        _fmt(r["ms"]),
                        f"warm; load {r['load_ms']:.0f} ms; {r['audio_s']:.1f}s audio",
                    )
                )  # type: ignore[arg-type]
                best[label] = _med(r["ms"])  # type: ignore[arg-type]
                notes.append(f"{label} text: {str(r['text'])[:80]!r}")
            except Exception as exc:  # noqa: BLE001
                rows.append((label, "FAILED", f"{type(exc).__name__}: {str(exc)[:80]}"))
    if "llm" not in skip:
        try:
            r = await bench_llm(a.runs)
            rows.append(
                (
                    "LLM TTFT via Brain (tools bound)",
                    _fmt(r["brain_ms"]),
                    f"{r['model']}; {r['tools']} tools; max_tokens 80",
                )
            )  # type: ignore[arg-type]
            rows.append(("LLM TTFT plain probe_llm", _fmt(r["probe_ms"]), "no tools"))  # type: ignore[arg-type]
            best["llm"] = _med(r["brain_ms"])  # type: ignore[arg-type]
        except Exception as exc:  # noqa: BLE001
            rows.append(("LLM TTFT via Brain", "FAILED", f"{type(exc).__name__}: {str(exc)[:80]}"))
    if "tts" not in skip:
        try:
            r = await bench_tts(a.kitten, a.runs)
            rows.append(
                (
                    "TTS KittenTTS first chunk",
                    _fmt(r["ms"]),
                    f"{FIRST_CLAUSE!r} ({len(FIRST_CLAUSE)} chars); load {r['load_ms']:.0f} ms",
                )
            )  # type: ignore[arg-type]
            best["tts"] = _med(r["ms"])  # type: ignore[arg-type]
        except Exception as exc:  # noqa: BLE001
            rows.append(
                ("TTS KittenTTS first chunk", "FAILED", f"{type(exc).__name__}: {str(exc)[:80]}")
            )

    print("| Stage | median ms (min / max) | notes |\n|---|---|---|")
    for r in rows:
        print(f"| {r[0]} | {r[1]} | {r[2]} |")
    llm, tts = best.get("llm"), best.get("tts")
    print(
        f"\n**Projected voice_to_first_audio** = {a.vad_stop_ms:.0f} ms VAD stop + STT "
        "+ LLM TTFT + TTS first chunk "
        f"(budget {a.budget_ms:.0f} ms)\n"
    )
    print("| STT | VAD | STT | LLM | TTS | total | vs budget |\n|---|---|---|---|---|---|---|")
    if llm is not None and tts is not None:
        for k, v in best.items():
            if not k.startswith("STT"):
                continue
            tot = a.vad_stop_ms + v + llm + tts
            verdict = "OK" if tot <= a.budget_ms else f"OVER by {tot - a.budget_ms:.0f}"
            print(
                f"| {k} | {a.vad_stop_ms:.0f} | {v:.0f} | {llm:.0f} | {tts:.0f} | {tot:.0f} "
                f"| {verdict} |"
            )
    else:
        print("(projection needs llm and tts results)")
    if llm is not None and tts is not None:
        for k, v in best.items():
            if k.startswith("STT"):
                stages = {"VAD stop": a.vad_stop_ms, "STT": v, "LLM TTFT": llm, "TTS": tts}
                top = max(stages, key=lambda x: stages[x])
                print(f"- Dominant stage with {k}: **{top}** ({stages[top]:.0f} ms)")
    for n in notes:
        print(f"- {n}")


if __name__ == "__main__":
    asyncio.run(main())

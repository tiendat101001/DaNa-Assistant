"""Shared contract suite: every offline-safe (mock) provider is run through its
ABC so all providers honor the same conventions (provider-plugin skill)."""

from __future__ import annotations

from dana_agent.config import (
    ActionsConfig,
    LLMConfig,
    S2SConfig,
    STTConfig,
    TTSConfig,
    VADConfig,
    WakeConfig,
)
from dana_agent.providers.base import (
    ActionProvider,
    Capabilities,
    LLMProvider,
    S2SProvider,
    STTProvider,
    TTSProvider,
    VADProvider,
    WakeWordProvider,
)
from dana_agent.providers.registry import (
    build_actions,
    build_llm,
    build_s2s,
    build_stt,
    build_tts,
    build_vad,
    build_wake,
)
from dana_agent.types import Action, AudioFrame, Message, Role, Tool
from tests.conftest import async_iter, collect


def test_all_mock_providers_declare_capabilities() -> None:
    providers = [
        build_vad(VADConfig()),
        build_wake(WakeConfig()),
        build_stt(STTConfig()),
        build_llm(LLMConfig()),
        build_tts(TTSConfig()),
        build_s2s(S2SConfig()),
        build_actions(ActionsConfig()),
    ]
    for p in providers:
        assert isinstance(p.capabilities, Capabilities)


def test_vad_detects_speech_vs_silence() -> None:
    vad = build_vad(VADConfig())
    assert isinstance(vad, VADProvider)
    assert vad.is_speech(AudioFrame(b"\x00\x00" * 160, 16000)) is False
    assert vad.is_speech(AudioFrame(b"loud", 16000)) is True


def test_wake_fires_on_sentinel_and_ignores_noise() -> None:
    wake = build_wake(WakeConfig(phrase="Hey DaNa", end_phrase="Bye DaNa"))
    assert isinstance(wake, WakeWordProvider)
    assert wake.process(AudioFrame(b"random-noise", 16000)) is None
    hit = wake.process(AudioFrame(b"Hey DaNa", 16000))
    assert hit is not None and hit.type.value == "wake"
    bye = wake.process(AudioFrame(b"Bye DaNa", 16000))
    assert bye is not None and bye.type.value == "end"


async def test_stt_streams_partials_then_final() -> None:
    stt = build_stt(STTConfig())
    assert isinstance(stt, STTProvider)
    frames = [AudioFrame(b"hello", 16000), AudioFrame(b"world", 16000)]
    out = await collect(stt.stream(async_iter(frames)))
    assert out[-1].is_final is True
    assert out[-1].text == "hello world"
    assert all(not t.is_final for t in out[:-1])


async def test_llm_streams_text_then_done() -> None:
    llm = build_llm(LLMConfig())
    assert isinstance(llm, LLMProvider)
    msgs = [Message(Role.USER, "hello there")]
    chunks = await collect(llm.stream(msgs))
    assert chunks[-1].done is True
    assert "hello there" in "".join(c.text for c in chunks)


async def test_llm_emits_tool_call_when_named() -> None:
    llm = build_llm(LLMConfig())
    tools = [Tool(name="weather", description="get weather")]
    msgs = [Message(Role.USER, "what is the weather")]
    chunks = await collect(llm.stream(msgs, tools))
    calls = [tc for c in chunks for tc in c.tool_calls]
    assert calls and calls[0].name == "weather"


async def test_tts_streams_audio_chunks() -> None:
    tts = build_tts(TTSConfig())
    assert isinstance(tts, TTSProvider)
    out = await collect(tts.stream(async_iter(["Hello.", " World."])))
    assert len(out) == 2
    assert all(c.pcm for c in out)


async def test_s2s_streams_audio_from_audio() -> None:
    s2s = build_s2s(S2SConfig())
    assert isinstance(s2s, S2SProvider)
    out = await collect(s2s.stream(async_iter([AudioFrame(b"x", 16000)])))
    assert len(out) == 1


async def test_actions_classify_and_execute() -> None:
    actions = build_actions(ActionsConfig())
    assert isinstance(actions, ActionProvider)
    assert actions.classify(Action("read_page", {})).value == "read_only"
    assert actions.classify(Action("delete_file", {})).value == "destructive"
    assert actions.classify(Action("fill_form", {})).value == "reversible"
    result = await actions.execute(Action("read_page", {}))
    assert result.ok is True

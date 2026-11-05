"""Offline tests for the piper / openai / litert TTS providers (fakes only)."""

from __future__ import annotations

import json
import os
import sys
import types
import wave
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from dana_agent.config import TTSConfig
from dana_agent.providers.base import TTSProvider
from dana_agent.providers.registry import build_tts
from dana_agent.providers.tts import litert as litert_mod
from dana_agent.types import AudioChunk
from tests.conftest import async_iter, collect

RATE = 16_000


def _pcm(n_samples: int) -> bytes:
    return (np.arange(n_samples, dtype="<i2")).tobytes()


# --- construction (offline, no disk/network) -------------------------------


@pytest.mark.parametrize("name", ["piper", "openai"])
def test_constructs_with_default_config(name: str) -> None:
    p = build_tts(TTSConfig(provider=name))
    assert isinstance(p, TTSProvider)
    assert p.capabilities.streaming is True
    assert {"en", "vi"} <= p.capabilities.languages


def test_litert_is_english_only_and_constructs_offline() -> None:
    caps = build_tts(TTSConfig(provider="litert")).capabilities
    assert caps.languages == frozenset({"en"}) and caps.streaming


def test_local_providers_are_self_hosted_realtime() -> None:
    for name in ("piper", "litert"):
        caps = build_tts(TTSConfig(provider=name)).capabilities
        assert caps.self_hosted and caps.latency_class == "realtime"


# --- piper -----------------------------------------------------------------


class _NewChunk:
    def __init__(self, pcm: bytes, rate: int) -> None:
        self.audio_int16_bytes = pcm
        self.sample_rate = rate


class _NewVoice:
    """piper-tts >= 1.3 surface."""

    def __init__(self) -> None:
        self.config = types.SimpleNamespace(sample_rate=RATE)
        self.calls: list[str] = []

    def synthesize(self, text: str) -> Any:
        self.calls.append(text)
        yield _NewChunk(_pcm(RATE // 2), RATE)  # 500 ms


class _OldVoice:
    """piper-tts 1.2 surface."""

    def __init__(self) -> None:
        self.config = types.SimpleNamespace(sample_rate=RATE)
        self.calls: list[str] = []

    def synthesize_stream_raw(self, text: str) -> Any:
        self.calls.append(text)
        yield _pcm(RATE // 2)


def _install_piper(monkeypatch: pytest.MonkeyPatch, voice: Any, tmp_path: Path) -> list[Any]:
    loads: list[Any] = []

    class PiperVoice:
        @staticmethod
        def load(model: str, config_path: str | None = None) -> Any:
            loads.append((model, config_path))
            return voice

    mod = types.ModuleType("piper")
    mod.PiperVoice = PiperVoice  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "piper", mod)
    onnx = tmp_path / "v.onnx"
    onnx.write_bytes(b"x")
    (tmp_path / "v.onnx.json").write_text("{}")
    return loads


def _piper(tmp_path: Path) -> TTSProvider:
    return build_tts(TTSConfig(provider="piper", model_path=str(tmp_path / "v.onnx")))


@pytest.mark.parametrize("voice_cls", [_NewVoice, _OldVoice])
async def test_piper_api_generations(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, voice_cls: type
) -> None:
    voice = voice_cls()
    loads = _install_piper(monkeypatch, voice, tmp_path)
    out = await collect(_piper(tmp_path).stream(async_iter(["Hello there."])))
    assert loads[0][1].endswith("v.onnx.json")
    assert all(isinstance(c, AudioChunk) and c.sample_rate == RATE for c in out)
    assert all(len(c.pcm) % 2 == 0 for c in out)
    # 500 ms in ~150 ms chunks -> 4 pieces, none larger than 150 ms
    assert len(out) == 4
    assert max(len(c.pcm) for c in out) == RATE * 150 // 1000 * 2
    assert sum(len(c.pcm) for c in out) == RATE  # 500 ms * 2 bytes


async def test_piper_multiple_sentences_in_order_skips_blank(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    voice = _NewVoice()
    _install_piper(monkeypatch, voice, tmp_path)
    out = await collect(_piper(tmp_path).stream(async_iter(["One.", "  ", "Two.", "Three."])))
    assert voice.calls == ["One.", "Two.", "Three."]
    assert len(out) == 12


async def test_piper_warmup_loads_and_synthesizes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    voice = _NewVoice()
    loads = _install_piper(monkeypatch, voice, tmp_path)
    p = _piper(tmp_path)
    await p.warmup()  # type: ignore[attr-defined]
    assert voice.calls == ["ok"] and len(loads) == 1
    await collect(p.stream(async_iter(["again"])))
    assert len(loads) == 1  # voice reused


async def test_piper_missing_model_is_clear(tmp_path: Path) -> None:
    p = build_tts(TTSConfig(provider="piper", model_path=str(tmp_path / "nope.onnx")))
    with pytest.raises((FileNotFoundError, ImportError)):
        await collect(p.stream(async_iter(["hi"])))


async def test_piper_voice_found_in_models_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from dana_agent.providers.tts.piper import PiperTTS

    loads = _install_piper(monkeypatch, _NewVoice(), tmp_path)
    (tmp_path / "vi_VN-vais1000-medium.onnx").write_bytes(b"x")
    p = PiperTTS(
        TTSConfig(provider="piper", voice="vi_VN-vais1000-medium"), models_dir=str(tmp_path)
    )
    await p.warmup()
    assert loads[0][0].endswith("vi_VN-vais1000-medium.onnx")


# --- openai ----------------------------------------------------------------


class _FakeResponse:
    def __init__(self, parts: list[bytes]) -> None:
        self._parts = parts

    async def __aenter__(self) -> _FakeResponse:
        return self

    async def __aexit__(self, *a: object) -> None:
        return None

    async def iter_bytes(self, chunk_size: int = 1024) -> Any:
        assert chunk_size == 4800
        for p in self._parts:
            yield p


def _install_openai(
    monkeypatch: pytest.MonkeyPatch, parts: list[bytes]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    init: dict[str, Any] = {}
    calls: list[dict[str, Any]] = []

    class _Streaming:
        def create(self, **kw: Any) -> _FakeResponse:
            calls.append(kw)
            return _FakeResponse(parts)

    class AsyncOpenAI:
        def __init__(self, **kw: Any) -> None:
            init.update(kw)
            self.audio = types.SimpleNamespace(
                speech=types.SimpleNamespace(with_streaming_response=_Streaming())
            )

        async def close(self) -> None:
            init["closed"] = True

    mod = types.ModuleType("openai")
    mod.AsyncOpenAI = AsyncOpenAI  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "openai", mod)
    return init, calls


async def test_openai_odd_byte_carry_and_params(monkeypatch: pytest.MonkeyPatch) -> None:
    data = bytes(range(10))
    init, calls = _install_openai(monkeypatch, [data[:3], data[3:4], data[4:9], data[9:]])
    p = build_tts(
        TTSConfig(provider="openai", api_key="sk-x", base_url="http://gw/v1")  # type: ignore[arg-type]
    )
    out = await collect(p.stream(async_iter(["Hi.", ""])))
    assert init["base_url"] == "http://gw/v1" and init["api_key"] == "sk-x"
    assert len(calls) == 1
    assert calls[0]["model"] == "gpt-4o-mini-tts" and calls[0]["voice"] == "alloy"
    assert calls[0]["response_format"] == "pcm" and calls[0]["input"] == "Hi."
    assert all(c.sample_rate == 24_000 and len(c.pcm) % 2 == 0 for c in out)
    assert b"".join(c.pcm for c in out) == data
    await p.aclose()
    assert init["closed"] is True


async def test_openai_custom_model_voice_and_warmup(monkeypatch: pytest.MonkeyPatch) -> None:
    init, calls = _install_openai(monkeypatch, [b"\x00\x00"])
    p = build_tts(TTSConfig(provider="openai", model="tts-1", voice="nova"))
    await p.warmup()  # type: ignore[attr-defined]
    assert "base_url" in init
    await collect(p.stream(async_iter(["a", "b"])))
    assert [(c["model"], c["voice"], c["input"]) for c in calls] == [
        ("tts-1", "nova", "a"),
        ("tts-1", "nova", "b"),
    ]


# --- litert (KittenTTS) ----------------------------------------------------


def test_kitten_tokenize_pads_and_maps() -> None:
    ids = litert_mod.tokenize("hə, a")
    assert ids.dtype == np.int32 and ids.shape[0] == 1
    assert ids[0, 0] == 0 and ids[0, -1] == 0
    assert ids.shape[1] > 4


def test_kitten_split_sentences_and_numbers() -> None:
    assert litert_mod.split_sentences("Hi there. How are you?") == ["Hi there.", "How are you?"]
    assert litert_mod.split_sentences("no end") == ["no end,"]
    assert litert_mod.normalize_text("I have 21 cats").split() == "I have twenty one cats".split()


class _G:
    """Fake graph stage; records calls, returns a vocoder-like waveform."""

    def __init__(self, samples: int = 0) -> None:
        self.samples = samples
        self.calls: list[dict[str, Any]] = []

    def __call__(self, feeds: dict[str, Any]) -> list[Any]:
        self.calls.append(feeds)
        if self.samples:
            return [np.linspace(-1.5, 1.5, self.samples, dtype=np.float32)[None, None, :]]
        if "input_ids" in feeds:
            n = feeds["input_ids"].shape[1]
            return [
                np.ones((1, n, 256), np.float32),
                np.ones((1, n, 128), np.float32),
                np.full((n,), 2, np.int32),
            ]
        if "en" in feeds:
            t = feeds["en"].shape[1]
            return [
                np.ones((1, t, 9), np.float32),
                np.ones((1, t), np.float32),
                np.ones((1, t), np.float32),
            ]
        raise AssertionError(feeds.keys())


def _ready_kitten(samples: int = 24_000) -> Any:
    p = litert_mod.LiteRTTTS(TTSConfig(provider="litert"))
    p._graphs = (_G(), _G(), _G(samples))  # type: ignore[assignment]
    p._voices = {"expr-voice-2-m": np.zeros((50, 256), np.float32)}
    p._phonemize = lambda s: "hə"
    p._ready = True
    return p


async def test_kitten_streams_int16_24k_chunks() -> None:
    p = _ready_kitten(samples=24_000)  # 1 s per sentence
    out = await collect(p.stream(async_iter(["One. Two."])))
    assert all(c.sample_rate == 24_000 and len(c.pcm) % 2 == 0 for c in out)
    assert max(len(c.pcm) for c in out) == 24_000 * 150 // 1000 * 2
    samples: Any = np.frombuffer(b"".join(c.pcm for c in out), dtype="<i2")
    assert len(samples) == 48_000  # two sentences, in order
    assert samples.max() == 32767 and samples.min() == -32767  # clipped
    pred = p._graphs[0].calls
    assert len(pred) == 2 and pred[0]["style"].shape == (1, 256)
    assert pred[0]["speed"][0] == pytest.approx(0.8)  # voice speed prior


async def test_kitten_webgpu_and_missing_model(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="webgpu"):
        await collect(
            build_tts(TTSConfig(provider="litert", backend="webgpu")).stream(async_iter(["x"]))
        )
    p = build_tts(TTSConfig(provider="litert", model_path=str(tmp_path)))
    with pytest.raises(FileNotFoundError):
        await collect(p.stream(async_iter(["hi"])))


def test_kitten_model_dir_layout(tmp_path: Path) -> None:
    d = litert_mod.resolve_model_dir(TTSConfig(provider="litert"), str(tmp_path))
    assert d == tmp_path / "litert-community__kitten-tts-nano-0.8"


def test_kitten_bundled_g2p_dictionary_path(tmp_path: Path) -> None:
    import gzip

    (tmp_path / "g2p_meta.json").write_text(
        json.dumps(
            {"char2idx": {"_": 0, "a": 3}, "idx2ph": {"0": "_", "3": "a"}, "start": 1, "end": 2}
        )
    )
    with gzip.open(tmp_path / "g2p_dict.txt.gz", "wt", encoding="utf-8") as fh:
        fh.write("hello\thəlˈoʊ\nworld\twˈɜːld\n")

    class It:
        def __init__(self, **kw: Any) -> None: ...
        def allocate_tensors(self) -> None: ...
        def get_input_details(self) -> list[dict[str, int]]:
            return [{"index": 0}]

        def get_output_details(self) -> list[dict[str, int]]:
            return [{"index": 1}]

    g = litert_mod.BundledG2P(tmp_path, It, 1)
    assert g("Hello, world!") == "həlˈoʊ, wˈɜːld!"


@pytest.mark.skipif(
    not os.environ.get("DANA_TEST_KITTEN_DIR"),
    reason="set DANA_TEST_KITTEN_DIR to run real KittenTTS",
)
async def test_kitten_real_synthesis() -> None:
    import time

    p = build_tts(
        TTSConfig(provider="litert", model_path=os.environ["DANA_TEST_KITTEN_DIR"], num_threads=4)
    )
    await p.warmup()  # type: ignore[attr-defined]
    t0 = time.perf_counter()
    first: float | None = None
    pcm = b""
    async for c in p.stream(async_iter(["Hello, I am DaNa."])):
        first = first or time.perf_counter() - t0
        assert c.sample_rate == 24_000
        pcm += c.pcm
    audio_s = len(pcm) / 2 / 24_000
    assert 1.0 < audio_s < 6.0
    assert first is not None and first < 3.0
    samples: Any = np.frombuffer(pcm, dtype="<i2")
    assert np.abs(samples).max() > 1000
    out = os.environ.get("DANA_TEST_KITTEN_WAV")
    if out:
        with wave.open(out, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(24_000)
            w.writeframes(pcm)


def test_kitten_tail_trim_keeps_short_pad() -> None:
    sr = 24_000
    loud: Any = (np.ones(sr // 10) * 12000).astype("<i2")
    pcm = np.concatenate([loud, np.zeros(sr, "<i2")]).tobytes()
    out = litert_mod._trim_tail(pcm)
    assert len(out) // 2 == sr // 10 + sr * 100 // 1000
    assert litert_mod._trim_tail(b"") == b""


async def test_kitten_clamps_leading_silence_frames() -> None:
    p = _ready_kitten()
    pred = p._graphs[0]
    seen: list[Any] = []
    orig = p._graphs[1]

    def spy(feeds: dict[str, Any]) -> list[Any]:
        seen.append(feeds["en"].shape[1])
        return list(orig(feeds))

    # fake predictor returns dur=2 for every token; make the first one long
    real = pred.__call__

    def long_first(feeds: dict[str, Any]) -> list[Any]:
        out = list(real(feeds))
        out[2] = out[2].copy()
        out[2][0] = 40
        return out

    p._graphs = (long_first, spy, p._graphs[2])
    await collect(p.stream(async_iter(["Hi."])))
    n_tokens = 3  # BOS/EOS + 1 phoneme ("hə" -> h, ə)... computed below
    ids = litert_mod.tokenize("hə")
    n_tokens = ids.shape[1]
    assert seen == [2 * (n_tokens - 1) + litert_mod.LEAD_FRAMES]


def test_kitten_graph_skips_realloc_on_same_shape() -> None:
    calls = {"alloc": 0}

    class It:
        def get_input_details(self) -> list[dict[str, Any]]:
            return [{"index": 0, "name": "serving_default_x:0"}]

        def get_output_details(self) -> list[dict[str, Any]]:
            return [{"index": 1}]

        def resize_tensor_input(self, i: int, shape: list[int]) -> None: ...
        def allocate_tensors(self) -> None:
            calls["alloc"] += 1

        def reset_all_variables(self) -> None: ...
        def set_tensor(self, i: int, v: Any) -> None: ...
        def invoke(self) -> None: ...
        def get_tensor(self, i: int) -> Any:
            return 0

    g = litert_mod._Graph(Path("m"), lambda **kw: It(), 1)
    g({"x": np.zeros((1, 4))})
    g({"x": np.zeros((1, 4))})
    g({"x": np.zeros((1, 5))})
    assert calls["alloc"] == 2

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from dana_agent.config import STTConfig
from dana_agent.providers.base import STTProvider
from dana_agent.providers.registry import build_stt
from dana_agent.providers.stt.litert import LiteRTWhisperSTT, _truncate_repeats
from dana_agent.runtime.litert import litert_available
from dana_agent.runtime.whisper_tokenizer import WhisperTokenizer, _bytes_to_unicode
from dana_agent.types import AudioFrame
from tests.conftest import async_iter, collect

B2U = _bytes_to_unicode()


def _tok(s: str) -> str:
    return "".join(B2U[b] for b in s.encode())


def _fixture_tokenizer() -> WhisperTokenizer:
    vocab = {_tok("Hello"): 0, _tok(" wor"): 1, _tok("ld"): 2, _tok(" xin"): 3, _tok(" chào"): 4}
    added = {
        "<|endoftext|>": 10,
        "<|startoftranscript|>": 11,
        "<|en|>": 12,
        "<|vi|>": 13,
        "<|transcribe|>": 14,
        "<|notimestamps|>": 15,
    }
    return WhisperTokenizer(vocab, added)


def test_tokenizer_decode_and_prefix(tmp_path: Path) -> None:
    t = _fixture_tokenizer()
    assert t.decode([0, 1, 2, 11, 10]) == "Hello world"
    assert t.decode([3, 4]) == "xin chào"
    assert t.prefix_ids("vi") == [11, 13, 14, 15]
    (tmp_path / "vocab.json").write_text(json.dumps({_tok("a"): 0}))
    (tmp_path / "added_tokens.json").write_text(json.dumps({"<|endoftext|>": 5}))
    assert WhisperTokenizer.from_dir(tmp_path).eot == 5
    with pytest.raises(FileNotFoundError):
        WhisperTokenizer.from_dir(tmp_path / "x")


def test_truncate_repeats() -> None:
    assert _truncate_repeats([1, 2] + [7] * 12) == [1, 2, 7]
    assert _truncate_repeats([1, 2, 3]) == [1, 2, 3]


class FakeModel:
    path = Path("/nonexistent/m.tflite")

    def __init__(self, sigs: dict[str, dict[str, list[str]]], runners: dict[str, Any]) -> None:
        self._s, self._r = sigs, runners

    def signatures(self) -> dict[str, dict[str, list[str]]]:
        return self._s

    def runner(self, sig: str | None = None) -> Any:
        return self._r[sig]  # type: ignore[index]

    def input_details(self) -> list[dict[str, Any]]:
        return [{"shape": [1, 80, 3000], "dtype": np.float32}]

    def invoke_default(self, inputs: list[np.ndarray]) -> list[np.ndarray]:
        assert inputs[0].shape == (1, 80, 3000)
        return [np.array([[0, 1, 2, 10]])]


class GenRunner:
    def get_input_details(self) -> dict[str, Any]:
        return {"mel": {"shape": [1, 80, 3000], "dtype": np.float32}}

    def __call__(self, **kw: Any) -> dict[str, Any]:
        assert kw["mel"].shape == (1, 80, 3000)
        return {"ids": np.array([[3, 4, 10]])}


class EncRunner:
    def get_input_details(self) -> dict[str, Any]:
        return {"args_0": {"shape": [1, 80, 3000], "dtype": np.float32}}

    def __call__(self, **kw: Any) -> dict[str, Any]:
        return {"o": np.zeros((1, 5, 4), np.float32)}


class DecRunner:
    """Emits Hello, ' wor', 'ld', EOT regardless of input, given the prefix length."""

    def __init__(self) -> None:
        self.calls = 0

    def get_input_details(self) -> dict[str, Any]:
        return {
            "args_0": {"shape": [1, 5, 4], "dtype": np.float32},
            "args_1": {"shape": [1, 16], "dtype": np.int32},
            "args_2": {"shape": [1, 1, 16, 16], "dtype": np.float32},
        }

    def __call__(self, **kw: Any) -> dict[str, Any]:
        self.calls += 1
        assert kw["args_1"].shape == (1, 16) and kw["args_2"].shape == (1, 1, 16, 16)
        n = int((kw["args_1"] != 10).sum())  # tokens so far (prefix 4 + generated)
        assert list(kw["args_1"][0, :4]) == [11, 13, 14, 15]
        out: Any = np.zeros((1, 16, 20), np.float32)
        out[0, n - 1, [0, 1, 2, 10][n - 4]] = 1.0
        return {"logits": out}


def _stt(model: Any, lang: str = "vi") -> LiteRTWhisperSTT:
    return LiteRTWhisperSTT(
        STTConfig(provider="litert", language_code=lang),
        tokenizer=_fixture_tokenizer(),
        model=model,
    )


async def test_generate_signatureless() -> None:
    stt = _stt(FakeModel({}, {}))
    out = await collect(stt.stream(async_iter([AudioFrame(b"\x00\x10" * 1600, 16000)])))
    assert len(out) == 1 and out[0].is_final and out[0].text == "Hello world"


async def test_generate_signature() -> None:
    stt = _stt(
        FakeModel({"serving": {"inputs": ["mel"], "outputs": ["ids"]}}, {"serving": GenRunner()})
    )
    out = await collect(stt.stream(async_iter([AudioFrame(b"\x00\x10" * 8000, 8000)])))
    assert out[0].text == "xin chào"


async def test_encoder_decoder() -> None:
    dec = DecRunner()
    sigs: dict[str, dict[str, list[str]]] = {
        "encode": {"inputs": ["args_0"], "outputs": []},
        "decode": {"inputs": [], "outputs": []},
    }
    stt = _stt(FakeModel(sigs, {"encode": EncRunner(), "decode": dec}))
    await stt.warmup()
    out = await collect(stt.stream(async_iter([AudioFrame(b"\x00\x10" * 3200, 16000)])))
    assert out[0].text == "Hello world" and out[0].language == "vi"
    assert dec.calls >= 4


async def test_empty_audio() -> None:
    out = await collect(_stt(FakeModel({}, {})).stream(async_iter([])))
    assert out[0].text == "" and out[0].is_final


def test_registered_and_offline_construct() -> None:
    stt = build_stt(STTConfig(provider="litert"))
    assert isinstance(stt, STTProvider) and isinstance(stt, LiteRTWhisperSTT)
    caps = stt.capabilities
    assert caps.self_hosted and caps.languages == {"en", "vi"} and caps.latency_class == "realtime"
    assert build_stt(STTConfig(provider="faster_whisper")).capabilities.self_hosted


async def test_missing_model_clear_error(tmp_path: Path) -> None:
    stt = LiteRTWhisperSTT(STTConfig(provider="litert", model="a/b"), models_dir=str(tmp_path))
    with pytest.raises(FileNotFoundError):
        await stt.warmup()


_REAL = os.environ.get("DANA_TEST_WHISPER_TFLITE")


@pytest.mark.skipif(not _REAL or not litert_available(), reason="DANA_TEST_WHISPER_TFLITE unset")
async def test_real_model_silence_and_speech() -> None:
    models_dir = os.environ.get("DANA_TEST_MODELS_DIR", str(Path(str(_REAL)).parents[1]))
    stt = LiteRTWhisperSTT(
        STTConfig(provider="litert", model_path=_REAL, num_threads=4), models_dir=models_dir
    )
    await stt.warmup()
    out: list[Any] = await collect(stt.stream(async_iter([AudioFrame(b"\x00\x00" * 32000, 16000)])))
    assert len(out[0].text) < 20
    wav = os.environ.get("DANA_TEST_SPEECH_WAV")
    if wav:
        import wave

        with wave.open(wav) as w:
            frame = AudioFrame(w.readframes(w.getnframes()), w.getframerate())
        res = await collect(stt.stream(async_iter([frame])))
        assert "weather" in res[0].text.lower()


async def test_energy_gate_skips_inference() -> None:
    class Boom(FakeModel):
        def invoke_default(self, inputs: list[np.ndarray]) -> list[np.ndarray]:
            raise AssertionError("must not run on silence")

    stt = _stt(Boom({}, {}))
    out = await collect(stt.stream(async_iter([AudioFrame(b"\x01\x00" * 16000, 16000)])))
    assert out[0].text == "" and out[0].is_final


class RawEncRunner:
    seen: list[tuple[int, ...]] = []

    def get_input_details(self) -> dict[str, Any]:
        return {"args_0": {"shape": [1, 8000], "dtype": np.float32}}

    def __call__(self, **kw: Any) -> dict[str, Any]:
        RawEncRunner.seen.append(kw["args_0"].shape)
        return {"o": np.zeros((1, 5, 4), np.float32)}


async def test_raw_waveform_windows_and_sentencepiece() -> None:
    from dana_agent.runtime.whisper_tokenizer import SentencePieceDecoder

    RawEncRunner.seen = []
    vocab = {"<unk>": 0, "<s>": 1, "</s>": 2, "<0x41>": 3, "\u2581hi": 4}
    tok = SentencePieceDecoder(vocab)
    assert tok.decode([1, 4, 3, 2]) == "hiA" or tok.decode([1, 4, 3, 2]) == "hi" + "A"
    assert tok.prefix_ids("en") == [1]

    class Dec(DecRunner):
        def get_input_details(self) -> dict[str, Any]:
            return {
                "args_0": {"shape": [1, 5, 4], "dtype": np.float32},
                "args_1": {"shape": [1, 8], "dtype": np.int32},
                "args_2": {"shape": [1, 1, 8, 8], "dtype": np.float32},
            }

        def __call__(self, **kw: Any) -> dict[str, Any]:
            n = int((kw["args_1"] != 2).sum())
            out: Any = np.zeros((1, 8, 5), np.float32)
            out[0, n - 1, 4 if n == 1 else 2] = 1.0
            return {"l": out}

    sigs: dict[str, dict[str, list[str]]] = {
        "encode": {"inputs": ["args_0"], "outputs": []},
        "decode": {"inputs": [], "outputs": []},
    }
    stt = LiteRTWhisperSTT(
        STTConfig(provider="litert"),
        tokenizer=tok,
        model=FakeModel(sigs, {"encode": RawEncRunner(), "decode": Dec()}),  # type: ignore[arg-type]
    )
    out = await collect(stt.stream(async_iter([AudioFrame(b"\x00\x10" * 12000, 16000)])))
    assert RawEncRunner.seen == [(1, 8000), (1, 8000)]  # 12000 samples -> 2 windows
    assert out[0].text == "hi hi"


_MOON = os.environ.get("DANA_TEST_MOONSHINE_TFLITE")


@pytest.mark.skipif(not _MOON or not litert_available(), reason="DANA_TEST_MOONSHINE_TFLITE unset")
async def test_real_moonshine() -> None:
    models_dir = os.environ.get("DANA_TEST_MODELS_DIR", str(Path(str(_MOON)).parents[1]))
    stt = LiteRTWhisperSTT(
        STTConfig(provider="litert", model="UsefulSensors/moonshine-tiny", model_path=_MOON),
        models_dir=models_dir,
    )
    await stt.warmup()
    wav = os.environ.get("DANA_TEST_SPEECH_WAV")
    if wav:
        import wave

        with wave.open(wav) as w:
            frame = AudioFrame(w.readframes(w.getnframes()), w.getframerate())
        res = await collect(stt.stream(async_iter([frame])))
        assert "weather" in res[0].text.lower()


def test_ctc_vocab_and_collapse(tmp_path: Path) -> None:
    from dana_agent.providers.stt.litert import _ctc_collapse, _load_ctc_vocab

    (tmp_path / "tokens.txt").write_text("<blk> 0\n<sos/eos> 1\n▁HI 2\nA 3\n", "utf-8")
    v = _load_ctc_vocab(tmp_path)
    assert v[2] == "▁HI"
    assert _ctc_collapse([0, 2, 2, 0, 3, 3, 1], v) == "HI A".replace(" A", "A")
    (tmp_path / "tokens.txt").write_text("<pad>\n<s>\n</s>\n<unk>\n|\nE\nT\n", "utf-8")
    v = _load_ctc_vocab(tmp_path)
    assert _ctc_collapse([5, 5, 4, 6, 0, 6], v) == "E TT"


def test_kaldi_fbank_shape_and_determinism() -> None:
    from dana_agent.runtime.audio import kaldi_fbank

    a = (0.3 * np.sin(np.arange(16000) / 7.0)).astype(np.float32)
    f = kaldi_fbank(a)
    assert f.shape == (100, 80) and np.isfinite(f).all()
    assert np.array_equal(f, kaldi_fbank(a))
    assert kaldi_fbank(np.zeros(0, np.float32)).shape == (0, 80)


async def test_ctc_fbank_strategy(tmp_path: Path) -> None:
    (tmp_path / "tokens.txt").write_text("<blk> 0\n<sos/eos> 1\n<unk> 2\n▁HI 3\n", "utf-8")

    class CtcModel(FakeModel):
        path = tmp_path / "m.tflite"

        def input_details(self) -> list[dict[str, Any]]:
            return [
                {"shape": [1, 160, 80], "dtype": np.float32},
                {"shape": [1, 78], "dtype": np.float32},
                {"shape": [1, 39], "dtype": np.float32},
            ]

        def output_details(self) -> list[dict[str, Any]]:
            return [{"shape": [1, 39, 4], "dtype": np.float32}]

        def invoke_default(self, inputs: list[np.ndarray]) -> list[np.ndarray]:
            assert inputs[0].shape == (1, 160, 80) and inputs[2].shape == (1, 39)
            out: Any = np.zeros((1, 39, 4), np.float32)
            out[0, :, 0] = 1.0
            out[0, 3:5, 3] = 5.0
            return [out]

    stt = LiteRTWhisperSTT(STTConfig(provider="litert"), model=CtcModel({}, {}))  # type: ignore[arg-type]
    t = np.arange(16000) / 7.0
    pcm = (0.3 * 32767 * np.sin(t)).astype("<i2").tobytes()
    out = await collect(stt.stream(async_iter([AudioFrame(pcm, 16000)])))
    assert out[0].text == "HI"

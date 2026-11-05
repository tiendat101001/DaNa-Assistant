"""LiteRT TTS — KittenTTS-nano (StyleTTS2/iSTFTNet) on ``ai-edge-litert``.

Model: ``litert-community/kitten-tts-nano-0.8`` (Apache-2.0, English only,
24 kHz, ~32 MB fp16). Layout under the model dir::

    kitten_predictor[_fp16].tflite   text ids + style -> durations, features
    kitten_prosody[_fp16].tflite     expanded features -> f0, noise, harmonics
    kitten_vocoder[_fp16].tflite     features -> waveform
    voices.npz                       per-voice style tables
    g2p/                             dictionary + neural G2P (no espeak-ng)

The inference logic is ported from the model card's reference ``say.py``.

Frontend (G2P): by default the *bundled* GPL-free frontend is used - the
OpenPhonemizer pronunciation dictionary (274k words) with a DeepPhonemizer
LiteRT graph (CTC decoding) for out-of-dictionary words. Its phoneme ids are not
espeak-ng's (stress on function words differs slightly), so audio differs a
little from the reference. If ``phonemizer`` + espeak-ng are installed and
``DANA_KITTEN_G2P=espeak`` is set, that exact reference frontend is used.

Not for Vietnamese: use ``tts=piper`` with a vi_VN voice. ``backend="webgpu"``
is browser-only and raises. The fused-LSTM graphs keep state in variable
tensors, so every graph call resets variables (required, per the model card).
Interpreters are not re-entrant: synthesis is serialized with a lock and runs
in a worker thread.
"""

from __future__ import annotations

import asyncio
import gzip
import json
import os
import re
import threading
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from dana_agent.config import ModelsConfig, TTSConfig
from dana_agent.observability import get_logger
from dana_agent.providers.base import Capabilities, TTSProvider
from dana_agent.providers.registry import register
from dana_agent.providers.tts._common import float_to_int16, slice_pcm
from dana_agent.types import AudioChunk

log = get_logger(__name__)

SAMPLE_RATE = 24_000
# The model emits ~0.8 s of silence for the leading pad token and 250-430 ms at
# the tail (measured). Clamp the lead to a few frames *before* the vocoder (also
# saves compute) and trim the tail after. 600 samples/frame at 24 kHz = 25 ms.
# Set either to None to disable (reference behaviour).
LEAD_FRAMES: int | None = 3
TAIL_KEEP_MS: int | None = 100
_SILENCE = 0.01  # |x| below this counts as silence for tail trimming
DEFAULT_REPO = "litert-community/kitten-tts-nano-0.8"
DEFAULT_VOICE = "expr-voice-2-m"

_PAD = "$"
_PUNCT = ';:,.!?¡¿—…"«»“” '
_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
_IPA = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"  # noqa: E501
SYMBOL_INDEX = {s: i for i, s in enumerate([_PAD, *_PUNCT, *_LETTERS, *_IPA])}

VOICE_ALIASES = {
    "Bella": "expr-voice-2-f",
    "Jasper": "expr-voice-2-m",
    "Luna": "expr-voice-3-f",
    "Bruno": "expr-voice-3-m",
    "Rosie": "expr-voice-4-f",
    "Hugo": "expr-voice-4-m",
    "Kiki": "expr-voice-5-f",
    "Leo": "expr-voice-5-m",
}
SPEED_PRIORS = {
    "expr-voice-2-f": 0.8,
    "expr-voice-2-m": 0.8,
    "expr-voice-3-m": 0.8,
    "expr-voice-3-f": 0.8,
    "expr-voice-4-m": 0.9,
    "expr-voice-4-f": 0.8,
    "expr-voice-5-m": 0.8,
    "expr-voice-5-f": 0.8,
}


# --------------------------------------------------------------- frontend


def tokenize(phonemes: str) -> Any:
    """IPA string -> ``[1, N]`` int32 ids (re-space like the official package)."""
    import numpy as np  # noqa: PLC0415

    spaced = " ".join(re.findall(r"\w+|[^\w\s]", phonemes))
    ids = [SYMBOL_INDEX[c] for c in spaced if c in SYMBOL_INDEX]
    return np.array([[0, *ids, 0]], dtype=np.int32)


_ONES = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split()  # noqa: E501
_TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()


def _int_to_words(n: int) -> str:
    if n < 20:
        return _ONES[n]
    if n < 100:
        return _TENS[n // 10] + ("" if n % 10 == 0 else " " + _ONES[n % 10])
    if n < 1000:
        rest = n % 100
        return _ONES[n // 100] + " hundred" + ("" if rest == 0 else " " + _int_to_words(rest))
    if n < 1_000_000:
        rest = n % 1000
        return (
            _int_to_words(n // 1000)
            + " thousand"
            + ("" if rest == 0 else " " + _int_to_words(rest))
        )
    return " ".join(_ONES[int(d)] for d in str(n))  # very large: digit by digit


def normalize_text(text: str) -> str:
    """Tiny English normalizer: expand integers; the G2P only knows letters."""
    return re.sub(r"\d+", lambda m: " " + _int_to_words(int(m.group())) + " ", text)


class BundledG2P:
    """Dictionary + DeepPhonemizer-LiteRT G2P (no espeak-ng)."""

    def __init__(self, g2p_dir: Path, interpreter_cls: Any, threads: int) -> None:
        self._meta = json.loads((g2p_dir / "g2p_meta.json").read_text(encoding="utf-8"))
        self._c2i: dict[str, int] = self._meta["char2idx"]
        self._i2p: dict[int, str] = {int(k): v for k, v in self._meta["idx2ph"].items()}
        self._repeats = int(self._meta.get("char_repeats", 3))
        self._maxt = int(self._meta.get("MAXT", 96))
        self._dict: dict[str, str] = {}
        self._oov: dict[str, str] = {}
        with gzip.open(g2p_dir / "g2p_dict.txt.gz", "rt", encoding="utf-8") as fh:
            for line in fh:
                word, _, ph = line.rstrip("\n").partition("\t")
                if word and word not in self._dict:
                    self._dict[word] = ph
        self._it = interpreter_cls(
            model_path=str(g2p_dir / "dp_g2p_matcha_fp16.tflite"), num_threads=threads
        )
        self._it.allocate_tensors()
        self._in = self._it.get_input_details()[0]["index"]
        self._out = self._it.get_output_details()[0]["index"]

    def word(self, word: str) -> str:
        w = word.lower()
        hit = self._dict.get(w)
        if hit is not None:
            return hit
        hit = self._oov.get(w)
        if hit is None:  # neural graph costs 15-40 ms; memoize out-of-dictionary words
            if len(self._oov) > 4096:
                self._oov.clear()
            hit = self._oov[w] = self._neural(w)
        return hit

    def _neural(self, word: str) -> str:
        import numpy as np  # noqa: PLC0415

        chars = [c for c in word if c in self._c2i]
        if not chars:
            return ""
        ids = [self._meta["start"]]
        for c in chars[: (self._maxt - 2) // self._repeats]:
            ids.extend([self._c2i[c]] * self._repeats)
        ids.append(self._meta["end"])
        ids += [0] * (self._maxt - len(ids))
        self._it.set_tensor(self._in, np.array([ids], dtype=np.float32))
        self._it.invoke()
        logits = self._it.get_tensor(self._out)[0]
        out: list[str] = []
        prev = -1
        end = int(self._meta["end"])
        for t in logits.argmax(axis=-1).tolist():
            if t == end:  # CTC output after <end> is padding garbage
                break
            if t != prev and t != 0:
                ph = self._i2p.get(t, "")
                if ph and not ph.startswith("<"):
                    out.append(ph)
            prev = t
        return "".join(out)

    def __call__(self, text: str) -> str:
        parts = re.findall(r"[A-Za-z']+|[^\sA-Za-z']", normalize_text(text))
        out: list[str] = []
        for tok in parts:
            if re.match(r"[A-Za-z']", tok):
                out.append(self.word(tok))
            elif out:
                out[-1] += tok  # punctuation hugs the previous word
            else:
                out.append(tok)
        return " ".join(p for p in out if p)


def _espeak_phonemizer() -> Any:
    import phonemizer  # type: ignore[import-not-found,unused-ignore]  # noqa: PLC0415

    backend = phonemizer.backend.EspeakBackend(
        language="en-us", preserve_punctuation=True, with_stress=True
    )
    return lambda text: str(backend.phonemize([text])[0])


def _trim_tail(pcm: bytes) -> bytes:
    """Drop trailing near-silence beyond ``TAIL_KEEP_MS`` (int16 mono PCM)."""
    if TAIL_KEEP_MS is None or not pcm:
        return pcm
    import numpy as np  # noqa: PLC0415

    a: Any = np.frombuffer(pcm, dtype="<i2")
    loud = np.nonzero(np.abs(a) > _SILENCE * 32767)[0]
    end = int(loud[-1]) + 1 if loud.size else 0
    keep = min(len(a), end + SAMPLE_RATE * TAIL_KEEP_MS // 1000)
    return pcm[: keep * 2]


def split_sentences(text: str, max_len: int = 400) -> list[str]:
    chunks: list[str] = []
    for sentence in re.split(r"(?<=[.!?])\s+", " ".join(text.split())):
        sentence = sentence.strip()
        if not sentence:
            continue
        while len(sentence) > max_len:
            cut = sentence.rfind(" ", 0, max_len)
            cut = cut if cut > max_len // 2 else max_len
            chunks.append(sentence[:cut])
            sentence = sentence[cut:].strip()
        chunks.append(sentence if sentence[-1] in ".!?,;:" else sentence + ",")
    return chunks


# ----------------------------------------------------------------- engine


def _import_interpreter_cls() -> Any:
    try:
        from dana_agent.runtime.litert import _import_interpreter_cls as imp  # noqa: PLC0415

        return imp()
    except ImportError:
        import importlib  # noqa: PLC0415

        return importlib.import_module("ai_edge_litert.interpreter").Interpreter


class _Graph:
    """One tflite stage with dynamic input shapes and per-call state reset."""

    def __init__(self, path: Path, interpreter_cls: Any, threads: int) -> None:
        self.it = interpreter_cls(model_path=str(path), num_threads=threads)
        self.inputs = self.it.get_input_details()
        self.outputs = self.it.get_output_details()
        self.last_ms = (0.0, 0.0)
        self._shapes: tuple[tuple[int, ...], ...] | None = None

    @staticmethod
    def _canon(name: str) -> str:
        name = name.split(":")[0]
        return name.removeprefix("serving_default_")

    def __call__(self, feeds: dict[str, Any]) -> list[Any]:
        t0 = time.perf_counter()
        shapes = tuple(tuple(feeds[self._canon(d["name"])].shape) for d in self.inputs)
        if shapes != self._shapes:  # re-allocating on an unchanged shape is pure waste
            for d, shape in zip(self.inputs, shapes, strict=True):
                self.it.resize_tensor_input(d["index"], list(shape))
            self.it.allocate_tensors()
            self._shapes = shapes
        t1 = time.perf_counter()
        self.it.reset_all_variables()  # fused-LSTM state persists otherwise
        for d in self.inputs:
            self.it.set_tensor(d["index"], feeds[self._canon(d["name"])])
        self.it.invoke()
        out = [self.it.get_tensor(o["index"]) for o in self.outputs]
        self.last_ms = ((t1 - t0) * 1000, (time.perf_counter() - t1) * 1000)  # (alloc, run)
        return out


def resolve_model_dir(cfg: TTSConfig, models_dir: str | None = None) -> Path:
    if cfg.model_path:
        p = Path(cfg.model_path)
        return p.parent if p.is_file() else p
    repo = cfg.model if "/" in cfg.model else DEFAULT_REPO
    root = Path(models_dir or ModelsConfig().models_dir)
    try:
        from dana_agent.models.store import model_dir  # noqa: PLC0415

        return Path(model_dir(root, repo))
    except ImportError:
        return root / repo.replace("/", "__")


@register("tts", "litert")
class LiteRTTTS(TTSProvider):
    def __init__(self, cfg: TTSConfig, *, models_dir: str | None = None) -> None:
        self._cfg = cfg
        self._models_dir = models_dir
        self._ready = False
        self._lock = threading.Lock()  # interpreters are not re-entrant
        self._aio_lock = asyncio.Lock()
        self._graphs: tuple[_Graph, _Graph, _Graph] | None = None
        self._voices: Any = None
        self._voice = DEFAULT_VOICE
        self._phonemize: Any = None

    @property
    def capabilities(self) -> Capabilities:
        # English only: KittenTTS has no Vietnamese voice. Use tts=piper for vi.
        return Capabilities(
            streaming=True,
            languages=frozenset({"en"}),
            self_hosted=True,
            latency_class="realtime",
        )

    # -- loading -----------------------------------------------------------

    def _load_sync(self) -> None:
        import numpy as np  # noqa: PLC0415

        if self._cfg.backend == "webgpu":
            raise ValueError(
                "tts backend 'webgpu' is a browser-side runtime and cannot run in the "
                "server process; use backend='cpu'."
            )
        root = resolve_model_dir(self._cfg, self._models_dir)
        if not (root / "voices.npz").is_file():
            raise FileNotFoundError(
                f"KittenTTS model not found in {root} (need kitten_*.tflite + voices.npz). "
                f"Pull {DEFAULT_REPO} with the model registry or set tts.model_path."
            )
        cls = _import_interpreter_cls()
        threads = self._cfg.num_threads

        def stage(name: str) -> _Graph:
            fp16 = root / f"kitten_{name}_fp16.tflite"
            return _Graph(fp16 if fp16.is_file() else root / f"kitten_{name}.tflite", cls, threads)

        self._graphs = (stage("predictor"), stage("prosody"), stage("vocoder"))
        self._voices = np.load(root / "voices.npz")
        v = self._cfg.voice
        v = DEFAULT_VOICE if v in ("", "default") else VOICE_ALIASES.get(v, v)
        if v not in self._voices.files:
            raise ValueError(f"unknown KittenTTS voice {v!r}; available: {self._voices.files}")
        self._voice = v
        if os.environ.get("DANA_KITTEN_G2P") == "espeak":
            self._phonemize = _espeak_phonemizer()
        else:
            self._phonemize = BundledG2P(root / "g2p", cls, threads)
        self._ready = True
        log.info("kitten_tts_loaded", dir=str(root), voice=self._voice, threads=threads)

    async def _ensure(self) -> None:
        if not self._ready:
            async with self._aio_lock:
                if not self._ready:
                    await asyncio.to_thread(self._load_sync)

    async def warmup(self) -> None:
        await self._ensure()
        # Two lengths: primes XNNPACK and the shape caches (first call ~300 ms).
        await asyncio.to_thread(self._synth_sync, "Ok.")
        await asyncio.to_thread(self._synth_sync, "Sure, I can do that for you now.")

    # -- inference ---------------------------------------------------------

    def _synth_sync(self, sentence: str, speed: float = 1.0) -> bytes:
        """One sentence -> int16 PCM @ 24 kHz (blocking; ported from say.py)."""
        import numpy as np  # noqa: PLC0415

        if self._graphs is None:
            raise RuntimeError("KittenTTS not loaded")
        predictor, prosody, vocoder = self._graphs
        with self._lock:
            ids = tokenize(self._phonemize(sentence))
            table = self._voices[self._voice]
            ref_id = min(len(sentence), table.shape[0] - 1)
            style = table[ref_id : ref_id + 1].astype(np.float32)
            eff = np.array([speed * SPEED_PRIORS.get(self._voice, 1.0)], dtype=np.float32)

            p_out = predictor({"input_ids": ids, "style": style, "speed": eff})
            d = next(o for o in p_out if o.ndim == 3 and o.shape[-1] == 256)
            t_en = next(o for o in p_out if o.ndim == 3 and o.shape[-1] == 128)
            dur = next(o for o in p_out if o.dtype in (np.int32, np.int64)).copy()
            if LEAD_FRAMES is not None:
                dur[0] = min(int(dur[0]), LEAD_FRAMES)
            en = np.repeat(d[0], dur, axis=0)[None].astype(np.float32)
            asr = np.repeat(t_en[0], dur, axis=0)[None].astype(np.float32)
            pr = prosody({"en": en, "style": style})
            har = next(o for o in pr if o.ndim == 3)
            f0, n = (o for o in pr if o.ndim == 2)
            v = vocoder({"asr": asr, "f0": f0, "n": n, "har": har, "style": style})
            return _trim_tail(float_to_int16(v[0][0]))

    async def stream(self, text: AsyncIterator[str]) -> AsyncIterator[AudioChunk]:
        await self._ensure()
        async for piece in text:
            for sentence in split_sentences(piece):
                pcm = await asyncio.to_thread(self._synth_sync, sentence)
                for chunk in slice_pcm(pcm, SAMPLE_RATE):
                    yield AudioChunk(pcm=chunk, sample_rate=SAMPLE_RATE)

    async def aclose(self) -> None:
        self._graphs = None
        self._ready = False

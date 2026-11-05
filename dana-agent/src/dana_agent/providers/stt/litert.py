"""Self-hosted CPU Whisper STT on LiteRT (utterance-level, low latency).

Two graph layouts are supported, chosen by signature introspection:

* ``generate``: one graph mapping a log-mel ``[1, n_mels, 3000]`` to token ids
  (usefulsensors/openai-whisper style, with or without signatures).
* ``encoder/decoder``: signatures whose names contain "encode"/"decode". The
  encoder runs once, then a greedy loop (forced prefix, max 224 tokens, stop at
  EOT) re-feeds the token sequence to the decoder (no KV-cache assumed).

Nothing is loaded at construction time; the model loads on first use / warmup.
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import numpy as np

from dana_agent.config import STTConfig
from dana_agent.observability import get_logger
from dana_agent.providers.base import Capabilities, STTProvider
from dana_agent.providers.registry import register
from dana_agent.runtime.audio import SAMPLE_RATE, pcm_to_mel, resample
from dana_agent.runtime.litert import MODEL_CACHE, LiteRTModel
from dana_agent.runtime.whisper_tokenizer import (
    SentencePieceDecoder,
    WhisperTokenizer,
    load_tokenizer,
)
from dana_agent.types import AudioFrame, Transcript

log = get_logger(__name__)

MAX_TOKENS = 224
SILENCE_DBFS = -45.0  # utterance RMS below this skips inference


def _models_dir() -> str:
    from dana_agent.config import get_settings

    return get_settings().models.models_dir


def _find_tflite(p: Path) -> Path | None:
    if p.is_file():
        return p
    if p.is_dir():
        files = sorted(p.rglob("*.tflite"), key=lambda f: f.stat().st_size, reverse=True)
        return files[0] if files else None
    return None


def resolve_model(cfg: STTConfig, models_dir: str | None = None) -> Path:
    """Explicit ``model_path`` (file or dir) else the registry store."""
    if cfg.model_path:
        found = _find_tflite(Path(cfg.model_path))
        if found is None:
            raise FileNotFoundError(f"no .tflite at model_path={cfg.model_path!r}")
        return found
    mdir = models_dir or _models_dir()
    try:
        from dana_agent.models.store import resolve_model_path

        found = resolve_model_path(mdir, cfg.model)
    except ImportError:
        d = Path(mdir) / cfg.model.replace("/", "__")
        found = _find_tflite(d)
    if found is None:
        raise FileNotFoundError(
            f"model {cfg.model!r} not found under {mdir}; pull it via the model registry"
        )
    return Path(found)


def _in_names(runner: Any, fallback: list[str]) -> dict[str, Any]:
    try:
        details = runner.get_input_details()
        return dict(details)
    except Exception:  # noqa: BLE001
        return {n: {} for n in fallback}


def _kind(name: str, det: dict[str, Any]) -> str:
    """Classify a decoder input by dtype/rank (names are often ``args_N``)."""
    shape = det.get("shape")
    dtype = np.dtype(det["dtype"]) if det.get("dtype") is not None else None
    if shape is not None and dtype is not None:
        rank = len(shape)
        if dtype.kind in "iu":
            return "pos" if rank <= 1 or int(np.prod(shape)) == 1 else "tok"
        if rank == 4 or "mask" in name.lower():
            return "mask"
        if rank == 3:
            return "enc"
    n = name.lower()
    if "mask" in n:
        return "mask"
    if any(k in n for k in ("encoder", "hidden", "audio", "memory", "feature", "cross")):
        return "enc"
    if any(k in n for k in ("pos", "offset")):
        return "pos"
    if any(k in n for k in ("token", "ids", "decoder_input", "input")):
        return "tok"
    return "other"


def _truncate_repeats(ids: list[int], max_run: int = 4) -> list[int]:
    """Collapse degenerate trailing repetition (n-gram loops) to one occurrence."""
    for n in (1, 2, 3, 4):
        reps = max_run if n == 1 else max_run // 2 + 1
        while len(ids) >= n * (reps + 1) and all(
            ids[-n * (k + 1) : len(ids) - n * k] == ids[-n:] for k in range(reps + 1)
        ):
            del ids[-n:]
        if len(ids) >= n * reps and all(
            ids[-n * (k + 1) : len(ids) - n * k] == ids[-n:] for k in range(reps)
        ):
            del ids[-n * (reps - 1) :]
    return ids


@register("stt", "litert")
class LiteRTWhisperSTT(STTProvider):
    def __init__(
        self,
        cfg: STTConfig,
        *,
        models_dir: str | None = None,
        tokenizer: WhisperTokenizer | SentencePieceDecoder | None = None,
        model: LiteRTModel | None = None,
    ) -> None:
        self._cfg = cfg
        self._models_dir = models_dir
        self._tok = tokenizer
        self._model = model
        self._lock = threading.Lock()

    @property
    def capabilities(self) -> Capabilities:
        return Capabilities(
            streaming=True,
            languages=frozenset({"en", "vi"}),
            self_hosted=True,
            latency_class="realtime",
        )

    # -- loading ---------------------------------------------------------
    def _ensure_loaded(self) -> tuple[LiteRTModel, WhisperTokenizer | SentencePieceDecoder]:
        with self._lock:
            if self._model is None:
                path = resolve_model(self._cfg, self._models_dir)
                self._model = MODEL_CACHE.get_or_load(
                    path, backend=self._cfg.backend, num_threads=self._cfg.num_threads
                )
            if self._tok is None:
                self._tok = self._load_tokenizer(self._model.path.parent)
            return self._model, self._tok

    def _load_tokenizer(self, model_dir: Path) -> WhisperTokenizer | SentencePieceDecoder:
        """model dir -> <models_dir>/openai__whisper-tiny -> clear error."""
        cands = [model_dir]
        try:
            base = Path(self._models_dir or _models_dir())
            name = self._cfg.model.split("/")[-1]
            cands += [
                base / f"openai__{name}",
                base / f"UsefulSensors__{name}",
                base / "openai__whisper-tiny",
            ]
        except Exception as exc:  # noqa: BLE001
            log.debug("stt_tokenizer_models_dir_unavailable", error=str(exc))
        for d in cands:
            if d.is_dir() and ((d / "vocab.json").is_file() or (d / "tokenizer.json").is_file()):
                return load_tokenizer(d)
        raise FileNotFoundError(
            "Whisper tokenizer files (vocab.json, added_tokens.json) not found in "
            f"{[str(c) for c in cands]}; pull openai/whisper-tiny via the model registry"
        )

    async def warmup(self) -> None:
        t0 = time.perf_counter()
        await asyncio.to_thread(self._ensure_loaded)
        await asyncio.to_thread(
            self._transcribe, np.zeros(SAMPLE_RATE, dtype=np.float32), gate=False
        )
        log.info("stt_litert_warm", ms=round((time.perf_counter() - t0) * 1000, 1))

    # -- inference (blocking; runs in a worker thread) ---------------------
    @staticmethod
    def _enc_input_shape(model: LiteRTModel) -> tuple[int, ...] | None:
        """Input shape of the first non-decode signature (or the sole graph)."""
        try:
            sigs = model.signatures()
            if sigs:
                for name in sigs:
                    if "decode" not in name.lower():
                        d = dict(model.runner(name).get_input_details())
                        return tuple(int(x) for x in next(iter(d.values()))["shape"])
                return None
            return tuple(int(x) for x in model.input_details()[0]["shape"])
        except Exception:  # noqa: BLE001
            return None

    def _transcribe(self, audio16k: np.ndarray, *, gate: bool = True) -> str:
        if gate and audio16k.size:
            rms = float(np.sqrt(np.mean(np.square(audio16k, dtype=np.float64))))
            if 20.0 * np.log10(max(rms, 1e-9)) < SILENCE_DBFS:
                log.debug("stt_litert_silence_skipped")
                return ""
        model, tok = self._ensure_loaded()
        shape = self._enc_input_shape(model)
        sigs = model.signatures()
        enc = [s for s in sigs if "encode" in s.lower()]
        dec = [s for s in sigs if "decode" in s.lower()]
        audio16k = np.clip(audio16k, -1.0, 1.0).astype(np.float32)
        if shape is not None and len(shape) == 2:
            # raw-waveform encoder (Moonshine): fixed window, chunk longer audio
            win = shape[1]
            texts: list[str] = []
            for start in range(0, max(audio16k.size, 1), win):
                chunk: np.ndarray[Any, Any] = np.zeros((1, win), dtype=np.float32)
                piece = audio16k[start : start + win]
                chunk[0, : piece.size] = piece
                if enc and dec:
                    ids = self._run_encdec(model, tok, chunk, enc[0], dec[0])
                else:
                    ids = self._run_generate(model, chunk, next(iter(sigs), None))
                texts.append(tok.decode(ids))
            return " ".join(t for t in texts if t).strip()
        n_mels = shape[1] if shape is not None and len(shape) == 3 and shape[1] in (80, 128) else 80
        pcm = (audio16k * 32767).astype("<i2").tobytes()
        mel: np.ndarray[Any, Any] = pcm_to_mel(pcm, SAMPLE_RATE, n_mels).astype(np.float32)
        if enc and dec:
            ids = self._run_encdec(model, tok, mel, enc[0], dec[0])
        else:
            ids = self._run_generate(model, mel, next(iter(sigs), None))
        return tok.decode(ids)

    @staticmethod
    def _run_generate(model: LiteRTModel, mel: np.ndarray, sig: str | None) -> list[int]:
        if sig is None:
            outs = model.invoke_default([mel])
            out = outs[0]
        else:
            runner = model.runner(sig)
            in_name = next(iter(_in_names(runner, model.signatures()[sig]["inputs"])))
            res = runner(**{in_name: mel})
            out = np.asarray(next(iter(res.values())))
        return [int(x) for x in np.asarray(out).reshape(-1)]

    def _run_encdec(
        self,
        model: LiteRTModel,
        tok: WhisperTokenizer | SentencePieceDecoder,
        mel: np.ndarray,
        enc_sig: str,
        dec_sig: str,
    ) -> list[int]:
        enc_runner = model.runner(enc_sig)
        enc_in = next(iter(_in_names(enc_runner, model.signatures()[enc_sig]["inputs"])))
        enc_out = dict(enc_runner(**{enc_in: mel}))

        dec_runner = model.runner(dec_sig)
        dec_inputs = _in_names(dec_runner, model.signatures()[dec_sig]["inputs"])
        prefix = tok.prefix_ids(self._cfg.language_code)
        eot = tok.eot
        tokens = list(prefix)
        tok_shape = next(
            (d.get("shape") for k, d in dec_inputs.items() if _kind(k, d) == "tok"), None
        )
        cap = int(tok_shape[-1]) if tok_shape is not None and len(tok_shape) >= 2 else MAX_TOKENS
        limit = min(MAX_TOKENS, cap) if cap else MAX_TOKENS
        for _ in range(limit - len(prefix)):
            feed: dict[str, Any] = {}
            n = len(tokens)
            for name, det in dec_inputs.items():
                kind = _kind(name, det)
                dtype = det.get("dtype", np.float32)
                if kind == "enc":
                    feed[name] = enc_out.get(name, next(iter(enc_out.values())))
                elif kind == "pos":
                    feed[name] = np.array([n - 1], dtype=dtype)
                elif kind == "tok":
                    shape = det.get("shape")
                    fixed = int(shape[-1]) if shape is not None and len(shape) >= 2 else 0
                    arr: np.ndarray[Any, Any] = np.full((1, max(fixed, n)), eot, dtype=np.int64)
                    arr[0, :n] = tokens
                    feed[name] = arr.astype(dtype)
                elif kind == "mask":
                    shape = tuple(int(x) for x in det.get("shape", (1, 1, n, n)))
                    t = shape[-1]
                    causal = np.triu(np.full((t, t), -1e9, dtype=np.float32), k=1)
                    feed[name] = np.broadcast_to(causal, shape).astype(dtype).copy()
                else:
                    feed[name] = np.zeros(det.get("shape", (1,)), dtype=dtype)
            logits = np.asarray(next(iter(dec_runner(**feed).values())))
            if logits.ndim == 3:
                row = logits[0, min(n, logits.shape[1]) - 1]
            else:
                row = logits.reshape(-1, logits.shape[-1])[-1]
            nxt = int(np.argmax(row))
            if nxt == eot:
                break
            tokens.append(nxt)
        return _truncate_repeats(tokens[len(prefix) :])

    # -- STTProvider -------------------------------------------------------
    async def stream(self, audio: AsyncIterator[AudioFrame]) -> AsyncIterator[Transcript]:
        chunks: list[np.ndarray] = []
        async for frame in audio:
            x: np.ndarray[Any, Any] = np.frombuffer(
                frame.pcm[: len(frame.pcm) // 2 * 2], dtype="<i2"
            )
            x = x.astype(np.float32) / 32768.0
            if frame.channels > 1:
                x = x[: x.size // frame.channels * frame.channels]
                x = x.reshape(-1, frame.channels).mean(axis=1)
            chunks.append(resample(x, frame.sample_rate))
        lang = self._cfg.language_code
        if not chunks:
            yield Transcript("", is_final=True, language=lang)
            return
        wav = np.concatenate(chunks)
        text = await asyncio.to_thread(self._transcribe, wav)
        yield Transcript(text, is_final=True, language=lang)

    async def aclose(self) -> None:
        return None

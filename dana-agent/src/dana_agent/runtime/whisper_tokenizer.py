"""Minimal byte-level BPE *decoder* for Whisper token ids (no encode side)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _bytes_to_unicode() -> dict[int, str]:
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs, strict=True)}


_UNICODE_TO_BYTE = {v: k for k, v in _bytes_to_unicode().items()}

# Multilingual Whisper defaults, used only if the model dir lacks the tokens.
_DEFAULT_SPECIALS = {
    "<|endoftext|>": 50257,
    "<|startoftranscript|>": 50258,
    "<|en|>": 50259,
    "<|vi|>": 50278,
    "<|transcribe|>": 50359,
    "<|notimestamps|>": 50363,
}


class WhisperTokenizer:
    def __init__(self, vocab: dict[str, int], added: dict[str, int] | None = None) -> None:
        self._added = dict(added or {})
        self._tok2id = dict(vocab)
        self._tok2id.update(self._added)
        self._id2tok = {i: t for t, i in vocab.items()}
        self._special_ids = set(self._added.values())
        self._special_ids.update(
            i for t, i in vocab.items() if t.startswith("<|") and t.endswith("|>")
        )

    @classmethod
    def from_dir(cls, model_dir: str | Path) -> WhisperTokenizer:
        d = Path(model_dir)
        vocab: dict[str, int] = {}
        added: dict[str, int] = {}
        vj = d / "vocab.json"
        if vj.is_file():
            vocab = {str(k): int(v) for k, v in json.loads(vj.read_text("utf-8")).items()}
        aj = d / "added_tokens.json"
        if aj.is_file():
            added.update({str(k): int(v) for k, v in json.loads(aj.read_text("utf-8")).items()})
        tj = d / "tokenizer.json"
        if tj.is_file():
            data: dict[str, Any] = json.loads(tj.read_text("utf-8"))
            model_vocab = (data.get("model") or {}).get("vocab")
            if not vocab and isinstance(model_vocab, dict):
                vocab = {str(k): int(v) for k, v in model_vocab.items()}
            for t in data.get("added_tokens") or []:
                added.setdefault(str(t["content"]), int(t["id"]))
        if not vocab:
            raise FileNotFoundError(f"no vocab.json / tokenizer.json in {d}")
        return cls(vocab, added)

    def token_id(self, token: str) -> int | None:
        return self._tok2id.get(token)

    @property
    def eot(self) -> int:
        tid = self.token_id("<|endoftext|>")
        return tid if tid is not None else _DEFAULT_SPECIALS["<|endoftext|>"]

    def prefix_ids(self, language: str, *, task: str = "transcribe") -> list[int]:
        """``<|startoftranscript|><|lang|><|task|><|notimestamps|>``."""
        names = ["<|startoftranscript|>", f"<|{language}|>", f"<|{task}|>", "<|notimestamps|>"]
        out: list[int] = []
        for n in names:
            tid = self.token_id(n)
            if tid is None:
                tid = _DEFAULT_SPECIALS.get(n)
            if tid is None:
                raise KeyError(f"token {n} not in vocabulary")
            out.append(tid)
        return out

    def decode(self, ids: list[int]) -> str:
        eot = self.eot
        buf = bytearray()
        for i in ids:
            i = int(i)
            if i >= eot or i in self._special_ids:
                continue
            tok = self._id2tok.get(i)
            if tok is None or (tok.startswith("<|") and tok.endswith("|>")):
                continue
            for ch in tok:
                b = _UNICODE_TO_BYTE.get(ch)
                if b is not None:
                    buf.append(b)
                else:
                    buf.extend(ch.encode("utf-8"))
        return buf.decode("utf-8", errors="replace").strip()


class SentencePieceDecoder:
    """Decode-only tokenizer for Llama/SentencePiece-style BPE ``tokenizer.json``
    (Moonshine): ``▁`` -> space, ``<0xNN>`` byte fallback, specials dropped."""

    def __init__(self, vocab: dict[str, int], *, bos: int = 1, eos: int = 2) -> None:
        self._id2tok = {i: t for t, i in vocab.items()}
        self.bos, self._eos = bos, eos

    @classmethod
    def from_dir(cls, model_dir: str | Path) -> SentencePieceDecoder:
        data: dict[str, Any] = json.loads((Path(model_dir) / "tokenizer.json").read_text("utf-8"))
        vocab = {str(k): int(v) for k, v in data["model"]["vocab"].items()}
        for t in data.get("added_tokens") or []:
            vocab.setdefault(str(t["content"]), int(t["id"]))
        return cls(vocab, bos=vocab.get("<s>", 1), eos=vocab.get("</s>", 2))

    @property
    def eot(self) -> int:
        return self._eos

    def prefix_ids(self, language: str, *, task: str = "transcribe") -> list[int]:
        return [self.bos]

    def decode(self, ids: list[int]) -> str:
        buf = bytearray()
        for i in ids:
            tok = self._id2tok.get(int(i))
            if tok is None or (
                tok.startswith("<") and tok.endswith(">") and not tok.startswith("<0x")
            ):
                continue
            if len(tok) == 6 and tok.startswith("<0x") and tok.endswith(">"):
                buf.append(int(tok[3:5], 16))
            else:
                buf.extend(tok.replace("▁", " ").encode("utf-8"))
        return buf.decode("utf-8", errors="replace").strip()


def load_tokenizer(model_dir: str | Path) -> WhisperTokenizer | SentencePieceDecoder:
    d = Path(model_dir)
    if (d / "vocab.json").is_file():
        return WhisperTokenizer.from_dir(d)
    tj = d / "tokenizer.json"
    if tj.is_file():
        vocab = (json.loads(tj.read_text("utf-8")).get("model") or {}).get("vocab") or {}
        if any(k.startswith("▁") for k in vocab):
            return SentencePieceDecoder.from_dir(d)
        return WhisperTokenizer.from_dir(d)
    raise FileNotFoundError(f"no tokenizer files in {d}")

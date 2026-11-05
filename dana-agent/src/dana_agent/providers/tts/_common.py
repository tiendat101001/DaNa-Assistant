"""Helpers shared by the local TTS providers (PCM slicing, model-file lookup)."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

from dana_agent.config import ModelsConfig, TTSConfig

# Small chunks so playback starts ~immediately and barge-in can cut audio fast.
CHUNK_MS = 150


def slice_pcm(pcm: bytes, sample_rate: int, chunk_ms: int = CHUNK_MS) -> Iterator[bytes]:
    """Split int16 mono PCM into ~``chunk_ms`` pieces, always sample-aligned."""
    step = max(2, (sample_rate * chunk_ms // 1000) * 2)
    for i in range(0, len(pcm), step):
        piece = pcm[i : i + step]
        if len(piece) % 2:  # defensive: never emit a half sample
            piece = piece[:-1]
        if piece:
            yield piece


def float_to_int16(audio: Any) -> bytes:
    """float32 waveform in [-1, 1] -> little-endian int16 PCM bytes."""
    import numpy as np  # noqa: PLC0415

    arr = np.asarray(audio, dtype=np.float32).reshape(-1)
    return bytes((np.clip(arr, -1.0, 1.0) * 32767.0).astype("<i2").tobytes())


def resolve_local_model(cfg: TTSConfig, suffix: str, *, models_dir: str | None = None) -> Path:
    """Locate the on-disk model file for a local TTS provider.

    Order: ``cfg.model_path`` -> ``<models_dir>/<voice><suffix>`` ->
    ``<models_dir>/<model><suffix>`` -> the model store's registry layout
    (``dana_agent.models.store.resolve_model_path``). Touches disk only when
    called (i.e. at warmup/first stream), never at construction.
    """
    if cfg.model_path:
        p = Path(cfg.model_path)
        if p.is_dir():
            found = sorted(p.glob(f"*{suffix}"), key=lambda f: f.stat().st_size, reverse=True)
            if found:
                return found[0]
            raise FileNotFoundError(f"no {suffix} file in model_path directory {p}")
        if not p.exists():
            raise FileNotFoundError(f"TTS model_path does not exist: {p}")
        return p

    root = Path(models_dir or ModelsConfig().models_dir)
    for name in (cfg.voice, cfg.model):
        if name and not name.startswith(("mock", "default")):
            cand = root / f"{name}{suffix}"
            if cand.is_file():
                return cand
    try:
        from dana_agent.models.store import resolve_model_path  # noqa: PLC0415

        found_p = resolve_model_path(root, cfg.model, suffixes=(suffix,))
        if found_p is not None:
            return Path(found_p)
    except ImportError:
        pass
    raise FileNotFoundError(
        f"No {suffix} TTS model found for voice={cfg.voice!r} model={cfg.model!r} "
        f"under {root}. Set tts.model_path, or place <voice>{suffix} in the models dir."
    )

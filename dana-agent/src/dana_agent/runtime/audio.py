"""Numpy-only Whisper audio front-end (matches openai/whisper's log-mel)."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np
from numpy.typing import NDArray

SAMPLE_RATE = 16_000
N_FFT = 400
HOP = 160
N_SAMPLES = 30 * SAMPLE_RATE
N_FRAMES = 3000


def pcm16_to_float(pcm: bytes) -> NDArray[np.float32]:
    return np.frombuffer(pcm[: len(pcm) // 2 * 2], dtype="<i2").astype(np.float32) / 32768.0


def resample(
    audio: NDArray[np.float32], src_rate: int, dst_rate: int = SAMPLE_RATE
) -> NDArray[np.float32]:
    """Linear-interpolation resample of mono float audio."""
    if src_rate == dst_rate or audio.size == 0:
        return audio.astype(np.float32, copy=False)
    n_out = max(1, int(round(audio.size * dst_rate / src_rate)))
    x = np.linspace(0.0, audio.size - 1, num=n_out)
    return np.interp(x, np.arange(audio.size), audio).astype(np.float32)


def _hz_to_mel(f: Any) -> Any:
    """Slaney mel scale (librosa htk=False)."""
    f = np.asarray(f, dtype=np.float64)
    f_sp = 200.0 / 3
    mels = f / f_sp
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = np.log(6.4) / 27.0
    return np.where(
        f >= min_log_hz,
        min_log_mel + np.log(np.maximum(f, 1e-10) / min_log_hz) / logstep,
        mels,
    )


def _mel_to_hz(m: Any) -> Any:
    f_sp = 200.0 / 3
    freqs = f_sp * m
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = np.log(6.4) / 27.0
    return np.where(m >= min_log_mel, min_log_hz * np.exp(logstep * (m - min_log_mel)), freqs)


@lru_cache(maxsize=4)
def mel_filters(n_mels: int = 80) -> NDArray[np.float32]:
    """Slaney-normalised mel filterbank, shape (n_mels, n_fft//2+1)."""
    n_freqs = N_FFT // 2 + 1
    fft_freqs = np.linspace(0, SAMPLE_RATE / 2, n_freqs)
    mel_pts = np.linspace(
        _hz_to_mel(np.array(0.0)), _hz_to_mel(np.array(SAMPLE_RATE / 2)), n_mels + 2
    )
    hz = _mel_to_hz(mel_pts)
    fdiff = np.diff(hz)
    ramps = hz[:, None] - fft_freqs[None, :]
    lower = -ramps[:-2] / fdiff[:-1, None]
    upper = ramps[2:] / fdiff[1:, None]
    weights = np.maximum(0.0, np.minimum(lower, upper))
    enorm = 2.0 / (hz[2 : n_mels + 2] - hz[:n_mels])
    weights *= enorm[:, None]
    return weights.astype(np.float32)


def log_mel_spectrogram(
    audio: NDArray[np.float32], n_mels: int = 80, *, pad_to_frames: int | None = N_FRAMES
) -> NDArray[np.float32]:
    """Whisper log-mel, shape (n_mels, frames). Audio is 16 kHz float32 mono.

    With ``pad_to_frames`` (default 3000 = 30 s) the *audio* is zero-padded /
    trimmed to ``pad_to_frames * HOP`` samples first, as Whisper does.
    """
    audio = np.asarray(audio, dtype=np.float32)
    if pad_to_frames is not None:
        n = pad_to_frames * HOP
        audio = audio[:n] if audio.size >= n else np.pad(audio, (0, n - audio.size))
    window: NDArray[np.float32] = np.hanning(N_FFT + 1)[:-1].astype(np.float32)  # periodic Hann
    padded = np.pad(audio, (N_FFT // 2, N_FFT // 2), mode="reflect")
    n_frames = 1 + (padded.size - N_FFT) // HOP
    idx = np.arange(N_FFT)[None, :] + HOP * np.arange(n_frames)[:, None]
    frames = padded[idx] * window
    spec = np.fft.rfft(frames, axis=1)
    magnitudes = (spec.real**2 + spec.imag**2)[:-1].T.astype(np.float32)  # drop last frame
    mel = mel_filters(n_mels) @ magnitudes
    log_spec = np.log10(np.maximum(mel, 1e-10))
    log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
    return ((log_spec + 4.0) / 4.0).astype(np.float32)


def pcm_to_mel(pcm: bytes, sample_rate: int, n_mels: int = 80) -> NDArray[np.float32]:
    """int16 PCM bytes at any rate -> (1, n_mels, 3000) float32 model input."""
    audio = resample(pcm16_to_float(pcm), sample_rate)
    return log_mel_spectrogram(audio, n_mels)[None, ...]

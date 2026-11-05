from __future__ import annotations

import numpy as np

from dana_agent.runtime.audio import (
    log_mel_spectrogram,
    mel_filters,
    pcm_to_mel,
    resample,
)


def test_mel_filters_shape_and_slaney_norm() -> None:
    for n in (80, 128):
        f = mel_filters(n)
        assert f.shape == (n, 201)
        assert (f >= 0).all() and (f.sum(axis=1) > 0).all()


def test_mel_shape_range_and_determinism() -> None:
    t = np.arange(16000 * 2) / 16000
    a = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    m1, m2 = log_mel_spectrogram(a), log_mel_spectrogram(a)
    assert m1.shape == (80, 3000)
    assert np.array_equal(m1, m2)
    assert m1.max() <= 2.0 and m1.min() >= m1.max() - 2.0 - 1e-5  # (x.max-8+4)/4 floor
    assert log_mel_spectrogram(a, 128).shape == (128, 3000)


def test_silence_is_constant_floor() -> None:
    m = log_mel_spectrogram(np.zeros(16000, dtype=np.float32))
    assert np.allclose(m, m.flat[0])
    assert abs(float(m.flat[0]) + 1.5) < 1e-5  # log10(1e-10)=-10 -> (-10+4)/4


def test_long_audio_trimmed_and_resample_and_pcm() -> None:
    assert log_mel_spectrogram(np.zeros(16000 * 40, dtype=np.float32)).shape == (80, 3000)
    assert resample(np.ones(100, dtype=np.float32), 16000).shape == (100,)
    assert resample(np.ones(48000, dtype=np.float32), 48000, 16000).shape == (16000,)
    pcm = np.zeros(8000, dtype="<i2").tobytes()
    assert pcm_to_mel(pcm, 8000).shape == (1, 80, 3000)

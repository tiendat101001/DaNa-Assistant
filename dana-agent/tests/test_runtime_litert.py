from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from dana_agent.runtime.litert import LiteRTModel, ModelCache, litert_available


class FakeInterp:
    created = 0

    def __init__(self, model_path: str, num_threads: int = 1) -> None:
        FakeInterp.created += 1
        self.shape = [1, 2]
        self.t: dict[int, np.ndarray] = {}

    def get_signature_list(self) -> dict[str, Any]:
        return {"main": {"inputs": ["a"], "outputs": ["b"]}}

    def get_signature_runner(self, sig: str) -> str:
        return f"runner:{sig}"

    def get_input_details(self) -> list[dict[str, Any]]:
        return [{"index": 0, "shape": np.array(self.shape), "dtype": np.float32, "name": "x"}]

    def get_output_details(self) -> list[dict[str, Any]]:
        return [{"index": 1, "shape": np.array([1]), "dtype": np.float32, "name": "y"}]

    def resize_tensor_input(self, idx: int, shape: list[int]) -> None:
        self.shape = shape

    def allocate_tensors(self) -> None: ...

    def set_tensor(self, idx: int, v: np.ndarray) -> None:
        self.t[idx] = v

    def invoke(self) -> None:
        self.t[1] = self.t[0].sum(keepdims=True)

    def get_tensor(self, idx: int) -> np.ndarray:
        return self.t[idx]


def _model_file(tmp_path: Path) -> Path:
    p = tmp_path / "m.tflite"
    p.write_bytes(b"x" * 10)
    return p


def test_cache_semantics(tmp_path: Path) -> None:
    FakeInterp.created = 0
    c = ModelCache(interpreter_factory=FakeInterp)
    p = _model_file(tmp_path)
    m1 = c.get_or_load(p, num_threads=2)
    assert c.get_or_load(str(p), num_threads=2) is m1
    assert c.get_or_load(p, num_threads=3) is not m1
    assert FakeInterp.created == 2
    assert len(c.loaded()) == 2
    assert c.unload(p) is True and c.loaded() == []
    assert c.unload(p) is False


def test_webgpu_rejected(tmp_path: Path) -> None:
    c = ModelCache(interpreter_factory=FakeInterp)
    with pytest.raises(ValueError, match="browser runtime"):
        c.get_or_load(_model_file(tmp_path), backend="webgpu")
    with pytest.raises(ValueError, match="browser runtime"):
        LiteRTModel(_model_file(tmp_path), backend="webgpu", interpreter_factory=FakeInterp)


def test_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        ModelCache(interpreter_factory=FakeInterp).get_or_load(tmp_path / "no.tflite")


def test_signatures_info_and_invoke_default(tmp_path: Path) -> None:
    import json

    m = LiteRTModel(_model_file(tmp_path), interpreter_factory=FakeInterp)
    assert m.signatures() == {"main": {"inputs": ["a"], "outputs": ["b"]}}
    assert m.runner() == "runner:main"
    with pytest.raises(KeyError):
        m.runner("nope")
    out = m.invoke_default([np.ones((1, 5), dtype=np.float32)])  # resize path
    assert out[0][0] == 5
    json.dumps(m.info())


def test_litert_available_returns_bool() -> None:
    assert isinstance(litert_available(), bool)


_REAL = os.environ.get("DANA_TEST_WHISPER_TFLITE")


@pytest.mark.skipif(not _REAL or not litert_available(), reason="DANA_TEST_WHISPER_TFLITE unset")
def test_real_model_signatures() -> None:
    m = ModelCache().get_or_load(Path(str(_REAL)), num_threads=4)
    assert {"encode", "decode"} <= set(m.signatures())

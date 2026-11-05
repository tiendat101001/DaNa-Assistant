"""Multi-stage LiteRT models (e.g. KittenTTS predictor/prosody/vocoder + G2P)
must load every stage, all-or-nothing."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from dana_agent.config import ModelsConfig
from dana_agent.models import ModelManager
from dana_agent.models.errors import ModelInputError
from dana_agent.models.store import ModelRecord, model_dir
from tests.test_models_pull import FakeCache

REPO = "litert-community/kitten-tts-nano-0.8"
STAGES = [
    "kitten_predictor_fp16.tflite",
    "kitten_prosody_fp16.tflite",
    "kitten_vocoder_fp16.tflite",
    "g2p/dp_g2p_matcha_fp16.tflite",
]


def _ready_model(tmp_path: Path, cache: Any) -> ModelManager:
    mgr = ModelManager(ModelsConfig(models_dir=str(tmp_path)), cache=cache)
    root = model_dir(tmp_path, REPO)
    files = []
    for rel in [*STAGES, "voices.npz"]:
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x")
        files.append({"path": rel, "size": 1})
    mgr.store.put(ModelRecord(id=REPO, task="tts", status="ready", files=files, size_bytes=5))
    return mgr


async def test_cpu_load_covers_every_stage(tmp_path: Path) -> None:
    cache = FakeCache()
    mgr = _ready_model(tmp_path, cache)
    out = await mgr.load(REPO, backend="cpu", num_threads=2)
    assert sorted(p.name for p in cache.loaded) == sorted(Path(s).name for s in STAGES)
    loaded = out["loaded"]
    assert loaded["backend"] == "cpu"
    assert len(loaded["info"]["stages"]) == len(STAGES)
    assert loaded["load_ms"] == pytest.approx(12.5 * len(STAGES))
    await mgr.unload(REPO)
    assert sorted(cache.unloaded) == sorted(cache.loaded)


async def test_stage_failure_rolls_back(tmp_path: Path) -> None:
    class Failing(FakeCache):
        def get_or_load(self, path: Path, *, backend: str = "cpu", num_threads: int = 4) -> Any:
            if "vocoder" in path.name:
                raise RuntimeError("bad graph")
            return super().get_or_load(path, backend=backend, num_threads=num_threads)

    cache = Failing()
    mgr = _ready_model(tmp_path, cache)
    with pytest.raises(ModelInputError, match="bad graph"):
        await mgr.load(REPO, backend="cpu")
    assert sorted(cache.unloaded) == sorted(cache.loaded)  # nothing left half-loaded
    assert mgr.get_local(REPO)["loaded"] is None

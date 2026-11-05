"""Live smoke test against huggingface.co (opt-in: ``pytest -m live``)."""

from __future__ import annotations

from pathlib import Path

import pytest

from dana_agent.config import ModelsConfig
from dana_agent.models import ModelManager

pytestmark = pytest.mark.live


async def test_live_list_detail_pull(tmp_path: Path) -> None:
    mgr = ModelManager(ModelsConfig(models_dir=str(tmp_path)))
    listing = await mgr.list_hub(limit=1000)
    assert len(listing["models"]) > 50
    detail = await mgr.hub_detail("litert-community/kitten-tts-nano-0.8")
    assert "kitten_predictor_fp16.tflite" in detail["recommended_files"]
    assert detail["task"] == "tts"
    job = await mgr.start_pull("litert-community/kitten-tts-nano-0.8", files=["README.md"])
    await mgr.wait(job.id)
    assert job.status == "done", job.error
    assert mgr.file_path("litert-community/kitten-tts-nano-0.8", "README.md").stat().st_size > 0

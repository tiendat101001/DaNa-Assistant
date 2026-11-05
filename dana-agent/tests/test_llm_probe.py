from __future__ import annotations

from typing import Any

import pytest

from dana_agent.brain import probe as probe_mod
from dana_agent.brain.probe import probe_llm
from dana_agent.config import LLMConfig


async def test_probe_mock() -> None:
    r = await probe_llm(LLMConfig(provider="mock"), "hello there")
    assert r["ok"] is True
    assert r["ttft_ms"] is not None
    assert r["total_ms"] >= r["ttft_ms"]
    assert "hello" in r["text"]
    assert r["error"] is None


async def test_probe_error_never_raises() -> None:
    r = await probe_llm(LLMConfig(provider="nope"))
    assert r["ok"] is False
    assert r["ttft_ms"] is None
    assert r["error"]


async def test_probe_stream_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    class Bad:
        def astream(self, prompt: str) -> Any:
            async def gen() -> Any:
                raise RuntimeError("conn refused")
                yield  # pragma: no cover

            return gen()

    monkeypatch.setattr(probe_mod, "build_chat_model", lambda cfg: Bad())
    r = await probe_llm(LLMConfig(provider="mock"))
    assert r["ok"] is False
    assert "conn refused" in r["error"]

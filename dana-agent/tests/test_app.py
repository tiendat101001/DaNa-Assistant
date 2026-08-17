"""Composition root: build_stack validates and assembles the right providers."""

from __future__ import annotations

import sys

import pytest

from dana_agent.app import build_stack
from dana_agent.config import AppConfig


async def test_build_cascade_stack() -> None:
    stack = await build_stack(AppConfig())
    assert stack.vad is not None
    assert stack.wake_gate is not None
    assert stack.wake_gate.state.value == "idle"
    assert stack.gate is not None
    assert stack.stt is not None
    assert stack.tts is not None
    assert stack.brain is not None
    assert stack.s2s is None
    await stack.aclose()


async def test_build_s2s_stack() -> None:
    stack = await build_stack(AppConfig.model_validate({"conversation": {"mode": "s2s"}}))
    assert stack.s2s is not None
    assert stack.stt is None
    assert stack.tts is None
    assert stack.brain is None
    await stack.aclose()


async def test_build_stack_with_sqlite_memory() -> None:
    cfg = AppConfig.model_validate(
        {"brain": {"memory": {"backend": "sqlite", "sqlite_path": ":memory:"}}}
    )
    stack = await build_stack(cfg)
    assert stack.brain is not None
    await stack.aclose()  # closes the sqlite connection


async def test_build_stack_defaults_to_shared_settings() -> None:
    # No cfg -> uses get_settings(); should assemble a cascade stack.
    stack = await build_stack()
    assert stack.brain is not None
    await stack.aclose()


def test_cli_main_validate_only_returns_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    from dana_agent.__main__ import main

    monkeypatch.setattr(sys, "argv", ["dana_agent", "--validate-only"])
    assert main() == 0

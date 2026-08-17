"""Shared test helpers for the control-plane suite."""

from __future__ import annotations

from typing import Any

from dana_agent.config import AppConfig


def clean_config(**overrides: Any) -> AppConfig:
    """Construct an ``AppConfig`` from defaults + overrides, ignoring any dev
    ``.env`` so control tests are deterministic (all-mock, English, no keys).

    ``_env_file`` is a real pydantic-settings ``__init__`` kwarg, but the pydantic
    mypy plugin synthesises ``__init__`` from the model fields and hides it, so the
    call-arg ignore lives here in one place instead of at every call site."""
    return AppConfig(_env_file=None, **overrides)  # type: ignore[call-arg]

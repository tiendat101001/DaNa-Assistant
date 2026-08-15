"""Config loading: defaults, env > file > defaults precedence, secret handling."""

from __future__ import annotations

import pytest
from pydantic import SecretStr, ValidationError

from dana_agent.config import AppConfig


def test_defaults_are_all_mock_cascade_english() -> None:
    # _env_file=None: bypass the local .env so we see pure code defaults, not
    # whatever this machine has configured.
    cfg = AppConfig(_env_file=None)  # type: ignore[call-arg]
    assert cfg.language == "en"
    assert cfg.conversation.mode == "cascade"
    assert cfg.conversation.stt.provider == "mock"
    assert cfg.conversation.llm.provider == "mock"
    assert cfg.conversation.tts.provider == "mock"
    assert cfg.actions.gate_policy == "standard"
    assert cfg.control.host == "127.0.0.1"
    assert cfg.brain.memory.backend == "memory"


def test_get_settings_is_cached() -> None:
    from dana_agent.config import get_settings

    assert get_settings() is get_settings()


def test_memory_backend_is_env_overridable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DANA_BRAIN__MEMORY__BACKEND", "sqlite")
    monkeypatch.setenv("DANA_BRAIN__MEMORY__SQLITE_PATH", ":memory:")
    cfg = AppConfig()
    assert cfg.brain.memory.backend == "sqlite"
    assert cfg.brain.memory.sqlite_path == ":memory:"


def test_env_overrides_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DANA_LANGUAGE", "vi")
    monkeypatch.setenv("DANA_CONVERSATION__LLM__PROVIDER", "mock")
    monkeypatch.setenv("DANA_CONVERSATION__LLM__TEMPERATURE", "0.1")
    cfg = AppConfig()
    assert cfg.language == "vi"
    assert cfg.conversation.llm.temperature == pytest.approx(0.1)


def test_init_kwargs_take_precedence_over_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DANA_LANGUAGE", "vi")
    cfg = AppConfig(language="en")
    assert cfg.language == "en"


def test_secrets_are_redacted() -> None:
    cfg = AppConfig.model_validate({"conversation": {"stt": {"api_key": "super-secret-key"}}})
    api_key = cfg.conversation.stt.api_key
    assert isinstance(api_key, SecretStr)
    assert "super-secret-key" not in repr(cfg)
    assert "super-secret-key" not in str(cfg.conversation.stt)
    assert api_key.get_secret_value() == "super-secret-key"


def test_unknown_field_is_rejected() -> None:
    with pytest.raises(ValidationError):
        AppConfig.model_validate({"not_a_real_field": True})


def test_out_of_range_sensitivity_rejected() -> None:
    with pytest.raises(ValidationError):
        AppConfig.model_validate({"wake": {"sensitivity": 5.0}})

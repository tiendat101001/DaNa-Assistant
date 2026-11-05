"""UI choices that need a restart (e.g. Models -> "Use as ASR") are persisted to
.dana/config-overrides.json and loaded at startup, above the .env file but
below the real process environment. Secrets are never written."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dana_agent.config import CONFIG_OVERRIDES_PATH, AppConfig
from dana_agent.control.controller import persist_overrides


def _assign_litert(base: AppConfig) -> AppConfig:
    data = base.model_dump()
    data["conversation"]["stt"].update(
        provider="litert", model="litert-community/moonshine-tiny", num_threads=2
    )
    data["conversation"]["llm"]["api_key"] = "sk-should-not-persist"
    return AppConfig.model_validate(data)


def test_persist_then_load_roundtrip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    for var in ("DANA_CONVERSATION__STT__PROVIDER", "DANA_CONVERSATION__STT__MODEL"):
        monkeypatch.delenv(var, raising=False)
    new = _assign_litert(AppConfig())
    persist_overrides(
        Path(CONFIG_OVERRIDES_PATH),
        new,
        [
            "conversation.stt.provider",
            "conversation.stt.model",
            "conversation.stt.num_threads",
            "conversation.llm.api_key",
        ],
    )
    saved = json.loads(Path(CONFIG_OVERRIDES_PATH).read_text(encoding="utf-8"))
    assert saved["conversation"]["stt"]["provider"] == "litert"
    assert "api_key" not in saved["conversation"].get("llm", {})  # never persist secrets

    reloaded = AppConfig()
    assert reloaded.conversation.stt.provider == "litert"
    assert reloaded.conversation.stt.model == "litert-community/moonshine-tiny"
    assert reloaded.conversation.stt.num_threads == 2


def test_overrides_beat_dotenv_but_not_process_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("DANA_CONVERSATION__STT__PROVIDER", raising=False)
    Path(".env").write_text("DANA_CONVERSATION__STT__PROVIDER=assemblyai\n", encoding="utf-8")
    Path(".dana").mkdir()
    Path(CONFIG_OVERRIDES_PATH).write_text(
        json.dumps({"conversation": {"stt": {"provider": "litert"}}}), encoding="utf-8"
    )
    assert AppConfig().conversation.stt.provider == "litert"  # UI choice beats .env file
    monkeypatch.setenv("DANA_CONVERSATION__STT__PROVIDER", "mock")
    assert AppConfig().conversation.stt.provider == "mock"  # real env still wins


def test_merge_keeps_earlier_choices(tmp_path: Path) -> None:
    path = tmp_path / "o.json"
    path.write_text(json.dumps({"conversation": {"tts": {"provider": "piper"}}}), encoding="utf-8")
    persist_overrides(path, _assign_litert(AppConfig()), ["conversation.stt.provider"])
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["conversation"]["tts"]["provider"] == "piper"
    assert saved["conversation"]["stt"]["provider"] == "litert"

"""Secret resolver: keyring first, env fallback, and CLI-facing mutators.

Everything runs offline: keyring is NOT installed in the test env, and where a
vault is needed it is monkeypatched with a fake, so nothing touches the real
Windows Credential Manager.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import Any

import pytest

from dana_agent import secrets as secrets_mod
from dana_agent.secrets import (
    KeyringUnavailableError,
    delete_secret,
    get_secret,
    resolve_secret,
    set_secret,
)
from tests.support import clean_config


class _FakeKeyring:
    """Minimal in-memory stand-in for the keyring module API."""

    class PasswordDeleteError(Exception):
        pass

    def __init__(self) -> None:
        self.store: dict[tuple[str, str], str] = {}

    def get_password(self, service: str, name: str) -> str | None:
        return self.store.get((service, name))

    def set_password(self, service: str, name: str, value: str) -> None:
        self.store[(service, name)] = value

    def delete_password(self, service: str, name: str) -> None:
        try:
            del self.store[(service, name)]
        except KeyError as exc:
            raise self.PasswordDeleteError(name) from exc


@pytest.fixture
def fake_keyring(monkeypatch: pytest.MonkeyPatch) -> _FakeKeyring:
    fake = _FakeKeyring()
    monkeypatch.setitem(sys.modules, "keyring", fake)
    return fake


# --- resolve_secret ---------------------------------------------------------


def test_resolve_secret_absent_keyring_falls_back_to_env(monkeypatch: pytest.MonkeyPatch) -> None:
    # Simulate keyring not installed: importing it raises ImportError.
    monkeypatch.setitem(sys.modules, "keyring", None)  # `import keyring` -> ImportError
    monkeypatch.setenv("SOME_KEY", "from-env")
    assert resolve_secret("SOME_KEY") == "from-env"


def test_resolve_secret_missing_everywhere_is_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "keyring", None)
    monkeypatch.delenv("NOPE_KEY", raising=False)
    assert resolve_secret("NOPE_KEY") is None


def test_resolve_secret_prefers_keyring_over_env(
    fake_keyring: _FakeKeyring, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_keyring.set_password("dana-agent", "K", "from-vault")
    monkeypatch.setenv("K", "from-env")
    assert resolve_secret("K") == "from-vault"


def test_resolve_secret_keyring_error_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    broken = SimpleNamespace(get_password=_raise)
    monkeypatch.setitem(sys.modules, "keyring", broken)
    monkeypatch.setenv("K2", "env-value")
    assert resolve_secret("K2") == "env-value"


def _raise(*_a: Any, **_k: Any) -> Any:
    raise RuntimeError("vault locked")


# --- CLI-facing mutators ----------------------------------------------------


def test_set_get_delete_roundtrip(fake_keyring: _FakeKeyring) -> None:
    set_secret("ASSEMBLYAI_API_KEY", "sk-123")
    assert get_secret("ASSEMBLYAI_API_KEY") == "sk-123"
    assert delete_secret("ASSEMBLYAI_API_KEY") is True
    assert get_secret("ASSEMBLYAI_API_KEY") is None


def test_delete_of_missing_is_false(fake_keyring: _FakeKeyring) -> None:
    assert delete_secret("NEVER_SET") is False


def test_mutators_without_keyring_raise_clean_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "keyring", None)
    with pytest.raises(KeyringUnavailableError):
        set_secret("X", "y")


def test_default_service_constant() -> None:
    assert secrets_mod.DEFAULT_SERVICE == "dana-agent"


# --- config wiring: keyring feeds the well-known API keys -------------------


def test_config_pulls_standard_key_from_vault(
    fake_keyring: _FakeKeyring, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("ASSEMBLYAI_API_KEY", raising=False)
    fake_keyring.set_password("dana-agent", "ASSEMBLYAI_API_KEY", "vault-key")
    cfg = clean_config(conversation={"stt": {"provider": "assemblyai"}})
    assert cfg.conversation.stt.api_key is not None
    assert cfg.conversation.stt.api_key.get_secret_value() == "vault-key"


def test_explicit_dana_key_beats_vault(
    fake_keyring: _FakeKeyring, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_keyring.set_password("dana-agent", "ASSEMBLYAI_API_KEY", "vault-key")
    monkeypatch.setenv("DANA_CONVERSATION__STT__API_KEY", "explicit-key")
    cfg = clean_config(conversation={"stt": {"provider": "assemblyai"}})
    assert cfg.conversation.stt.api_key is not None
    assert cfg.conversation.stt.api_key.get_secret_value() == "explicit-key"

"""Bearer-token generation/loading and header matching."""

from __future__ import annotations

from pathlib import Path

from dana_agent.control.auth import load_or_create_token, token_matches


def test_generates_and_persists_token(tmp_path: Path) -> None:
    path = tmp_path / "sub" / "control-token"
    token = load_or_create_token(path)
    assert token
    assert path.read_text(encoding="utf-8").strip() == token
    # A second load returns the same persisted token (agent and UI agree).
    assert load_or_create_token(path) == token


def test_reads_existing_token(tmp_path: Path) -> None:
    path = tmp_path / "control-token"
    path.write_text("preset-token\n", encoding="utf-8")
    assert load_or_create_token(path) == "preset-token"


def test_token_matches_accepts_correct_bearer() -> None:
    assert token_matches("abc", "Bearer abc") is True
    assert token_matches("abc", "bearer abc") is True  # scheme case-insensitive


def test_token_matches_rejects_bad_inputs() -> None:
    assert token_matches("abc", None) is False
    assert token_matches("abc", "") is False
    assert token_matches("abc", "abc") is False  # missing scheme
    assert token_matches("abc", "Bearer wrong") is False
    assert token_matches("abc", "Basic abc") is False

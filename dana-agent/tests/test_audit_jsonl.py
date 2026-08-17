"""Durable JSONL audit sink: valid append-only JSONL, survives reopen, and
redacts secret-looking arguments (safe-computer-use skill: you must be able to
explain and trust every action, without leaking secrets into the trail)."""

from __future__ import annotations

import json
from pathlib import Path

from dana_agent.actions.audit import AuditEntry, AuditLog, JsonlAuditSink, redact_arguments
from dana_agent.actions.gate import GateDecision
from dana_agent.types import Reversibility


def make_entry(tool: str, arguments: dict[str, object], *, at: float = 1.0) -> AuditEntry:
    return AuditEntry(
        at=at,
        tool=tool,
        arguments=arguments,
        reversibility=Reversibility.READ_ONLY,
        decision=GateDecision.ALLOW,
        executed=True,
        ok=True,
    )


def read_lines(path: Path) -> list[dict[str, object]]:
    text = path.read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def test_sink_appends_valid_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    log = AuditLog(sink=JsonlAuditSink(path))
    log.record(make_entry("browser_navigate", {"url": "https://example.com"}))
    log.record(make_entry("browser_read_text", {}))

    rows = read_lines(path)
    assert len(rows) == 2
    assert rows[0]["tool"] == "browser_navigate"
    assert rows[0]["arguments"] == {"url": "https://example.com"}
    assert rows[0]["decision"] == "allow"
    assert rows[0]["reversibility"] == "read_only"
    assert rows[0]["executed"] is True and rows[0]["ok"] is True


def test_sink_is_append_only(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    log = AuditLog(sink=JsonlAuditSink(path))
    log.record(make_entry("a", {}))
    first = path.read_text(encoding="utf-8")
    log.record(make_entry("b", {}))
    second = path.read_text(encoding="utf-8")
    # earlier content is preserved verbatim (never rewritten), only appended to
    assert second.startswith(first)
    assert len(read_lines(path)) == 2


def test_sink_survives_reopen(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    AuditLog(sink=JsonlAuditSink(path)).record(make_entry("a", {}))
    # a fresh log + fresh sink pointed at the same file keeps appending
    AuditLog(sink=JsonlAuditSink(path)).record(make_entry("b", {}))
    rows = read_lines(path)
    assert [r["tool"] for r in rows] == ["a", "b"]


def test_sink_creates_parent_dir(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "dir" / "audit.jsonl"
    AuditLog(sink=JsonlAuditSink(path)).record(make_entry("a", {}))
    assert path.exists()


def test_secret_arguments_are_redacted_in_the_trail(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    log = AuditLog(sink=JsonlAuditSink(path))
    log.record(
        make_entry(
            "browser_type",
            {"password": "hunter2", "api_key": "sk-abc", "into": "Login", "text": "user"},
        )
    )
    row = read_lines(path)[0]
    args = row["arguments"]
    assert isinstance(args, dict)
    assert args["password"] == "***redacted***"  # noqa: S105 — asserting redaction, not a secret
    assert args["api_key"] == "***redacted***"
    # non-secret args survive
    assert args["into"] == "Login" and args["text"] == "user"


def test_typed_password_value_is_redacted_by_field_label(tmp_path: Path) -> None:
    # The REAL browser_type shape: the secret lives under the generic "text" key,
    # with the field named by "into". Redaction must catch the value, not just
    # keys literally named 'password'/'api_key'. Otherwise the password lands in
    # .dana/audit.jsonl in cleartext.
    path = tmp_path / "audit.jsonl"
    log = AuditLog(sink=JsonlAuditSink(path))
    log.record(make_entry("browser_type", {"into": "Password", "text": "hunter2"}))
    args = read_lines(path)[0]["arguments"]
    assert isinstance(args, dict)
    assert args["text"] == "***redacted***"  # noqa: S105 — asserting redaction
    # a non-secret field's typed value is NOT redacted (audit stays useful)
    assert redact_arguments({"into": "Search", "text": "kittens"}) == {
        "into": "Search",
        "text": "kittens",
    }


def test_redact_arguments_is_pure() -> None:
    original: dict[str, object] = {"token": "abc", "url": "x"}
    redacted = redact_arguments(original)
    assert redacted == {"token": "***redacted***", "url": "x"}
    assert original == {"token": "abc", "url": "x"}  # input untouched


def test_in_memory_api_unaffected_without_a_sink() -> None:
    log = AuditLog()  # no sink → pure in-memory (Phase 4 behaviour)
    log.record(make_entry("a", {}))
    assert len(log.entries) == 1
    assert log.entries[0].tool == "a"

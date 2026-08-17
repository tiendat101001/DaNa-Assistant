"""Append-only audit trail of every action decision and execution.

Without this you cannot debug, explain, or trust the system (safe-computer-use
skill). Entries are kept in memory (a live snapshot for the UI) and, when a
durable sink is attached, mirrored to an append-only JSONL file so the trail
survives a restart.

The JSONL sink is *injectable and optional*: an ``AuditLog`` with no sink is
purely in-memory (the Phase 4 behaviour the existing suite relies on). Passing a
sink adds durability without changing the in-memory API.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from time import time
from typing import TYPE_CHECKING, Protocol

from dana_agent.observability import get_logger
from dana_agent.types import Reversibility

if TYPE_CHECKING:
    from dana_agent.actions.gate import GateDecision

log = get_logger(__name__)

# Argument keys whose values are secret-looking and must never hit the durable
# trail (or logs). Matched case-insensitively as a substring of the key.
_SECRET_KEY_HINTS: tuple[str, ...] = (
    "password",
    "passwd",
    "secret",
    "token",
    "api_key",
    "apikey",
    "credential",
    "auth",
    "cookie",
    "session",
    "otp",
    "pin",
)
_REDACTED = "***redacted***"

# A field-label argument (``browser_type(into="Password", text=...)``) that names
# a secret field means the *typed value* is itself a secret even though its own
# key ("text"/"value") is innocuous — redact the value too, so a password typed
# into a login form never lands in the durable trail in cleartext.
_FIELD_LABEL_KEYS: tuple[str, ...] = ("into", "label", "field", "placeholder")
_TYPED_VALUE_KEYS: tuple[str, ...] = ("text", "value")


def _looks_secret(name: str) -> bool:
    return any(hint in name.lower() for hint in _SECRET_KEY_HINTS)


def redact_arguments(arguments: dict[str, object]) -> dict[str, object]:
    """Copy ``arguments`` with secret-looking values replaced by a marker.

    Two rules, both fail-safe (over-redaction is harmless, a leak is not):

    1. **By key name** — an argument like ``password``/``api_key``/``token`` is
       redacted (substring match against known secret hints).
    2. **By field label** — the value typed into a secret-named field is redacted
       even when its own key is generic. ``browser_type(into="Password",
       text="hunter2")`` would otherwise write the password verbatim, because the
       secret lives under the innocuous ``text`` key.
    """
    typed_value_is_secret = any(
        key.lower() in _FIELD_LABEL_KEYS and isinstance(value, str) and _looks_secret(value)
        for key, value in arguments.items()
    )
    redacted: dict[str, object] = {}
    for key, value in arguments.items():
        lowered = key.lower()
        if _looks_secret(lowered):
            redacted[key] = _REDACTED
        elif typed_value_is_secret and lowered in _TYPED_VALUE_KEYS:
            redacted[key] = _REDACTED
        else:
            redacted[key] = value
    return redacted


@dataclass(frozen=True, slots=True)
class AuditEntry:
    at: float
    tool: str
    arguments: dict[str, object]
    reversibility: Reversibility
    decision: GateDecision
    executed: bool
    ok: bool | None

    def to_record(self) -> dict[str, object]:
        """A JSON-serialisable dict for the durable sink, with secrets redacted."""
        return {
            "at": self.at,
            "tool": self.tool,
            "arguments": redact_arguments(self.arguments),
            "reversibility": self.reversibility.value,
            "decision": self.decision.value,
            "executed": self.executed,
            "ok": self.ok,
        }


class AuditSink(Protocol):
    """A durable destination for audit entries. Implementations must be
    append-only and safe to call synchronously from the gate."""

    def append(self, entry: AuditEntry) -> None: ...


class JsonlAuditSink:
    """Append-only JSONL file sink: one JSON object per line, flushed per write.

    Append-only by construction — every ``append`` opens the file in ``"a"``
    mode and writes exactly one line, so existing lines are never rewritten and
    the trail survives a reopen. The parent directory is created on first use.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)

    @property
    def path(self) -> Path:
        return self._path

    def append(self, entry: AuditEntry) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(entry.to_record(), ensure_ascii=False)
        # Append mode + explicit flush: durable and never overwrites prior lines.
        with self._path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")


class AuditLog:
    """In-memory, append-only. Reads return an immutable snapshot. An optional
    ``sink`` mirrors every entry to durable storage (e.g. JSONL)."""

    def __init__(
        self,
        *,
        now: Callable[[], float] = time,
        sink: AuditSink | None = None,
    ) -> None:
        self._entries: list[AuditEntry] = []
        self._now = now
        self._sink = sink

    def record(self, entry: AuditEntry) -> None:
        self._entries.append(entry)
        if self._sink is not None:
            try:
                self._sink.append(entry)
            except OSError as exc:  # durability best-effort: never crash the gate
                log.warning("audit_sink_write_failed", error=str(exc))
        log.info(
            "action_audit",
            at=entry.at,
            tool=entry.tool,
            reversibility=entry.reversibility,
            decision=entry.decision,
            executed=entry.executed,
            ok=entry.ok,
        )

    @property
    def entries(self) -> tuple[AuditEntry, ...]:
        return tuple(self._entries)

    def now(self) -> float:
        return self._now()

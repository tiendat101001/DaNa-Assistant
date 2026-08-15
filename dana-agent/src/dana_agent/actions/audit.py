"""Append-only audit trail of every action decision and execution.

Without this you cannot debug, explain, or trust the system (safe-computer-use
skill). Phase 4 keeps entries in memory and mirrors them to structured logs; the
durable JSONL sink and live UI view arrive in Phase 5/6.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from time import time
from typing import TYPE_CHECKING

from dana_agent.observability import get_logger
from dana_agent.types import Reversibility

if TYPE_CHECKING:
    from dana_agent.actions.gate import GateDecision

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class AuditEntry:
    at: float
    tool: str
    arguments: dict[str, object]
    reversibility: Reversibility
    decision: GateDecision
    executed: bool
    ok: bool | None


class AuditLog:
    """In-memory, append-only. Reads return an immutable snapshot."""

    def __init__(self, *, now: Callable[[], float] = time) -> None:
        self._entries: list[AuditEntry] = []
        self._now = now

    def record(self, entry: AuditEntry) -> None:
        self._entries.append(entry)
        log.info(
            "action_audit",
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

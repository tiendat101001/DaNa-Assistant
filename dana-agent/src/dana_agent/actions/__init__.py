"""Action layer — the trusted, non-LLM code that validates and executes the
structured tool calls the brain proposes (safe-computer-use skill).

Phase 4 delivers the reversibility gate + audit log. The concrete browser
(Playwright) and OS action providers, allowlist enforcement on targets, and the
JSONL audit sink land in Phase 5 (TASK-006).
"""

from dana_agent.actions.audit import AuditEntry, AuditLog
from dana_agent.actions.gate import ActionGate, GateDecision

__all__ = ["ActionGate", "AuditEntry", "AuditLog", "GateDecision"]

"""Deterministic mock action provider.

Classifies by verb so the gate layer (Phase 5) can be tested against realistic
reversibility, and echoes rather than doing anything real. The gate/allowlist
and audit log are the trusted layer that will wrap any provider; this mock only
supplies ``classify`` + ``execute``.
"""

from __future__ import annotations

from dana_agent.config import ActionsConfig
from dana_agent.providers.base import ActionProvider, Capabilities
from dana_agent.providers.registry import register
from dana_agent.types import Action, ActionResult, Reversibility

_DESTRUCTIVE = ("delete", "send", "pay", "submit", "overwrite", "remove", "push")
_READ_ONLY = ("read", "get", "list", "screenshot", "view")


@register("actions", "mock")
class MockActions(ActionProvider):
    def __init__(self, cfg: ActionsConfig) -> None:
        self._cfg = cfg

    def classify(self, action: Action) -> Reversibility:
        name = action.tool.lower()
        if any(name.startswith(v) or v in name for v in _DESTRUCTIVE):
            return Reversibility.DESTRUCTIVE
        if any(name.startswith(v) for v in _READ_ONLY):
            return Reversibility.READ_ONLY
        return Reversibility.REVERSIBLE

    async def execute(self, action: Action) -> ActionResult:
        return ActionResult(ok=True, output=f"mock executed {action.tool}")

    @property
    def capabilities(self) -> Capabilities:
        return Capabilities(streaming=False, languages=frozenset({"en", "vi"}))

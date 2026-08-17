"""The action gate: classify by reversibility, decide allow/confirm/refuse per
policy, and audit everything. The LLM proposes; this layer disposes.

Gate by how bad a mistake would be, not how the request was phrased
(safe-computer-use skill):

- read-only            -> ALLOW  (run freely)
- reversible write     -> ALLOW, or CONFIRM under the conservative policy
- destructive          -> CONFIRM (all policies; this is the confirmation gate)
- out of scope         -> REFUSE
"""

from __future__ import annotations

from enum import StrEnum

from dana_agent.actions.audit import AuditEntry, AuditLog
from dana_agent.providers.base import ActionProvider
from dana_agent.types import Action, ActionResult, Reversibility


class GateDecision(StrEnum):
    ALLOW = "allow"
    CONFIRM = "confirm"
    REFUSE = "refuse"


# Which reversibility classes require confirmation under each policy.
_CONFIRM_UNDER: dict[str, frozenset[Reversibility]] = {
    "standard": frozenset({Reversibility.DESTRUCTIVE}),
    "wide": frozenset({Reversibility.DESTRUCTIVE}),
    "conservative": frozenset({Reversibility.REVERSIBLE, Reversibility.DESTRUCTIVE}),
}


class ActionGate:
    def __init__(
        self,
        provider: ActionProvider,
        *,
        policy: str = "standard",
        audit: AuditLog | None = None,
    ) -> None:
        if policy not in _CONFIRM_UNDER:
            raise ValueError(f"unknown gate policy {policy!r}")
        self._provider = provider
        self._policy = policy
        self._audit = audit or AuditLog()

    @property
    def audit(self) -> AuditLog:
        return self._audit

    @property
    def policy(self) -> str:
        return self._policy

    def set_policy(self, policy: str) -> None:
        """Change the gate policy on the live gate (hot-applied from PUT /config).

        ``decide`` re-reads ``self._policy`` on every action, so a change here
        takes effect on the very next gated action — no restart required."""
        if policy not in _CONFIRM_UNDER:
            raise ValueError(f"unknown gate policy {policy!r}")
        self._policy = policy

    def classify(self, action: Action) -> Reversibility:
        """Reversibility of an action, without recording an audit entry (for
        labelling UI/events). Use ``decide`` for the gated, audited decision."""
        return self._provider.classify(action)

    def decide(self, action: Action) -> tuple[GateDecision, Reversibility]:
        """Classify and decide; records a (non-executing) audit entry."""
        reversibility = self._provider.classify(action)
        if reversibility is Reversibility.OUT_OF_SCOPE:
            decision = GateDecision.REFUSE
        elif reversibility in _CONFIRM_UNDER[self._policy]:
            decision = GateDecision.CONFIRM
        else:
            decision = GateDecision.ALLOW
        self._record(action, reversibility, decision, executed=False, ok=None)
        return decision, reversibility

    async def execute(self, action: Action, reversibility: Reversibility) -> ActionResult:
        """Execute an already-authorized action; records an execution entry."""
        result = await self._provider.execute(action)
        self._record(action, reversibility, GateDecision.ALLOW, executed=True, ok=result.ok)
        return result

    def _record(
        self,
        action: Action,
        reversibility: Reversibility,
        decision: GateDecision,
        *,
        executed: bool,
        ok: bool | None,
    ) -> None:
        self._audit.record(
            AuditEntry(
                at=self._audit.now(),
                tool=action.tool,
                arguments=action.arguments,
                reversibility=reversibility,
                decision=decision,
                executed=executed,
                ok=ok,
            )
        )

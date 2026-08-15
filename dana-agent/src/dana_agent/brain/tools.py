"""DaNa tools as LangGraph/LangChain tools, each wrapped by the action gate.

The model proposes a tool call; the wrapped tool disposes: it classifies the
action, and for anything the policy flags destructive it calls LangGraph's
``interrupt`` to pause the graph for a human decision (resumable because state is
checkpointed). Read-only/reversible actions run and are audited. Nothing here
executes raw model output — every call goes through the ``ActionGate``.

Phase 4 tools are zero-argument actions (enough to prove the loop + gate +
confirmation end to end). Typed, parameterized browser/OS tools arrive with the
real action providers in Phase 5 (TASK-006).
"""

from __future__ import annotations

from dataclasses import dataclass

from langchain_core.tools import BaseTool, StructuredTool
from langgraph.types import interrupt

from dana_agent.actions.gate import ActionGate, GateDecision
from dana_agent.types import Action


@dataclass(frozen=True, slots=True)
class ToolSpec:
    name: str
    description: str


def build_gated_tool(spec: ToolSpec, gate: ActionGate) -> BaseTool:
    async def _run() -> str:
        action = Action(tool=spec.name, arguments={})
        decision, reversibility = gate.decide(action)

        if decision is GateDecision.REFUSE:
            return "refused: not allowed"
        if decision is GateDecision.CONFIRM:
            approved = interrupt(
                {
                    "type": "confirmation",
                    "tool": spec.name,
                    "arguments": {},
                    "reversibility": reversibility.value,
                }
            )
            if not approved:
                return "refused by user"

        result = await gate.execute(action, reversibility)
        return result.output or result.error or ""

    return StructuredTool.from_function(
        coroutine=_run,
        name=spec.name,
        description=spec.description,
    )


def build_gated_tools(specs: list[ToolSpec], gate: ActionGate) -> list[BaseTool]:
    return [build_gated_tool(spec, gate) for spec in specs]

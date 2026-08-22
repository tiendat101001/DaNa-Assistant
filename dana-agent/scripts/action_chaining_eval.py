"""Standalone multi-turn chaining eval (TASK-012 phase 2).

Phase 1 (``action_routing_eval.py``) proved single-utterance tool SELECTION.
This proves the whole loop with the real model driving the full ``Brain``
graph: tool sequences across turns, follow-ups that must reuse tool RESULTS
(summarize what was read), and destructive picks pausing at the confirmation
interrupt and honouring a denial.

Nothing real runs: actions execute against a ``RecordingActions`` provider —
classification delegates to the REAL browser/OS classifiers (so the gate and
interrupts behave exactly as production), execution is recorded and answered
with scenario-canned outputs.

Run (from dana-agent/):

    uv run python scripts/action_chaining_eval.py [--only youtube] [--verbose]
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dana_agent.actions.gate import ActionGate
from dana_agent.brain.agent import Brain
from dana_agent.brain.events import ConfirmationRequired, TurnComplete
from dana_agent.config import ActionsConfig, BrainConfig
from dana_agent.providers.actions.browser import BrowserActions
from dana_agent.providers.actions.os_win import WindowsOSActions
from dana_agent.providers.base import ActionProvider, Capabilities
from dana_agent.types import Action, ActionResult, Reversibility


def _load_phase1() -> Any:
    path = Path(__file__).resolve().parent / "action_routing_eval.py"
    spec = importlib.util.spec_from_file_location("action_routing_eval", path)
    if spec is None or spec.loader is None:  # pragma: no cover - path is fixed
        raise RuntimeError("cannot load action_routing_eval.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_P1 = _load_phase1()  # reuse build_eval_model + _SESSION_CONTEXT


class RecordingActions(ActionProvider):
    """Records executions; classification delegates to the real providers.

    Honest ``classify`` is what makes the confirmation scenario real: a
    destructive pick goes through the gate, raises LangGraph's ``interrupt``,
    and only executes if the human approves."""

    def __init__(self, cfg: ActionsConfig) -> None:
        self._browser = BrowserActions(cfg)
        self._os = WindowsOSActions(cfg)
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.canned: dict[str, str] = {}

    @property
    def capabilities(self) -> Capabilities:
        return Capabilities(
            streaming=False, languages=frozenset({"en", "vi"}), latency_class="fast"
        )

    def classify(self, action: Action) -> Reversibility:
        delegate = self._os if action.tool.startswith("os_") else self._browser
        return delegate.classify(action)

    async def execute(self, action: Action) -> ActionResult:
        self.calls.append((action.tool, dict(action.arguments)))
        return ActionResult(ok=True, output=self.canned.get(action.tool, "ok"))

    async def aclose(self) -> None:
        await self._browser.aclose()


@dataclass(slots=True)
class Scenario:
    id: str
    turns: tuple[str, ...]
    # Per-turn expectations: executed tools (name substring, arg predicate) and
    # a final-text predicate; confirm=(approve?) when a confirmation is due.
    expect_tools: tuple[tuple[str, dict[str, Any]], ...] = ()  # ALL must occur
    expect_any_tools: tuple[tuple[str, dict[str, Any]], ...] = ()  # ANY one must occur
    expect_no_tools: tuple[int, ...] = ()  # turn indices (0-based) with NO tool call
    confirm: bool | None = None  # None: no confirmation expected
    final_contains: tuple[str, ...] = ()
    canned: dict[str, str] = field(default_factory=dict)


_READ_PAGE_TEXT = (
    "DaNa is an always-on voice assistant built with Pipecat. It speaks English "
    "and Vietnamese, and its top priority is low latency."
)

SCENARIOS: tuple[Scenario, ...] = (
    Scenario(
        id="01-youtube-one-shot",
        turns=("Open YouTube and play lofi hip hop",),
        expect_tools=(("browser_search", {"engine": "youtube", "open_first": True}),),
    ),
    Scenario(
        id="02-read-then-summarize-vi",
        turns=("Read the page out loud", "Tóm tắt nội dung đó bằng một câu tiếng Việt"),
        expect_tools=(("browser_read_text", {}),),
        expect_no_tools=(1,),  # turn 2 must reuse the tool RESULT, not re-read
        canned={"browser_read_text": _READ_PAGE_TEXT},
        final_contains=("Việt", "độ trễ", "Pipecat", "tiếng Anh"),
    ),
    Scenario(
        id="03-confirm-deny-destructive",
        turns=("Click the Buy now button",),
        expect_tools=(),  # must NOT execute: confirmation denied below
        confirm=False,
    ),
    Scenario(
        id="04-seek-chain",
        turns=("Jump forward 30 seconds", "Now go back to the start"),
        expect_tools=(
            ("browser_media", {"action": "seek", "delta_s": 30}),
            ("browser_media", {"action": "seek_to", "position_s": 0}),
        ),
    ),
    Scenario(
        id="05-newtab-then-wikipedia",
        turns=("Open a new browser tab", "Now search Wikipedia for the Paracel Islands"),
        expect_tools=(
            ("browser_tab", {"action": "new"}),
            ("browser_search", {"engine": "wikipedia"}),
        ),
    ),
    Scenario(
        id="06-whats-playing-followup",
        turns=("What's playing right now?",),
        # Any status tool is a correct answer (same accept set as phase-1 #10).
        expect_any_tools=(
            ("browser_media", {"action": "state"}),
            ("browser_get_state", {}),
            ("os_media_state", {}),
        ),
        canned={
            "browser_media": "playing: 'lofi hip hop radio' at 3:12 / 10:40",
            "browser_get_state": (
                "page: 'lofi' (x) | tabs: 1 | media: playing: 'lofi hip hop radio'"
            ),
            "os_media_state": "Playing: 'lofi hip hop radio' by DaNa Radio",
        },
        final_contains=("lofi",),
    ),
)


def _match(call: tuple[str, dict[str, Any]], want_tool: str, want_args: dict[str, Any]) -> bool:
    name, args = call
    if want_tool not in name:
        return False
    for key, want in want_args.items():
        got = args.get(key)
        if isinstance(want, bool) or isinstance(want, str):
            if got != want:
                return False
        elif isinstance(want, (int, float)):
            if not isinstance(got, (int, float)) or abs(float(got) - float(want)) > 2:
                return False
    return True


async def run_scenario(model: Any, scenario: Scenario) -> tuple[bool, str]:
    cfg = ActionsConfig(
        provider="browser_os",
        os_actions_enabled=True,
        allowed_domains=["youtube.com", "google.com", "duckduckgo.com", "wikipedia.org"],
    )
    recorder = RecordingActions(cfg)
    recorder.canned.update(scenario.canned)
    gate = ActionGate(recorder)
    # Tools MUST wrap THIS gate (the recorder) — phase 1's helper builds its
    # own mock gate and executions would bypass the recording.
    from dana_agent.brain.tools import build_action_tools  # noqa: PLC0415

    tools = build_action_tools(cfg, gate)
    system = f"{BrainConfig().system_prompt}\n\n{_P1._SESSION_CONTEXT}"
    brain = Brain(model, gate, system_prompt=system, tools=tools)

    thread = f"eval-{scenario.id}-{uuid.uuid4().hex[:8]}"
    final_text = ""
    turn_tool_counts: list[int] = []
    saw_confirmation = False

    def _count_new(before: int) -> int:
        return len(recorder.calls) - before

    for turn_idx, utterance in enumerate(scenario.turns):
        before = len(recorder.calls)
        confirmation: ConfirmationRequired | None = None
        async for event in brain.run_turn(utterance, thread_id=thread):
            if isinstance(event, ConfirmationRequired):
                confirmation = event
            elif isinstance(event, TurnComplete):
                final_text = event.text
        if confirmation is not None:
            saw_confirmation = True
            if scenario.confirm is None:
                return False, f"turn {turn_idx}: unexpected confirmation for {confirmation.tool}"
            async for event in brain.resume_turn(thread_id=thread, approved=scenario.confirm):
                if isinstance(event, TurnComplete):
                    final_text = event.text
        turn_tool_counts.append(_count_new(before))

    # --- assertions -------------------------------------------------------
    if scenario.confirm is not None and not saw_confirmation:
        return False, "expected the destructive pick to pause for confirmation — none fired"
    if scenario.confirm is False:
        bought = [c for c in recorder.calls if "browser_click" in c[0]]
        if bought:
            return False, f"destructive click EXECUTED after denial: {bought}"
    for want_tool, want_args in scenario.expect_tools:
        if not any(_match(c, want_tool, want_args) for c in recorder.calls):
            return False, f"missing expected call {want_tool}({want_args}); got {recorder.calls}"
    if scenario.expect_any_tools and not any(
        _match(c, want_tool, want_args)
        for c in recorder.calls
        for want_tool, want_args in scenario.expect_any_tools
    ):
        return False, (
            f"none of the acceptable calls happened: {scenario.expect_any_tools}; "
            f"got {recorder.calls}"
        )
    for turn_idx in scenario.expect_no_tools:
        if turn_tool_counts[turn_idx] > 0:
            return False, (
                f"turn {turn_idx} re-called a tool instead of using the prior "
                f"result: {recorder.calls}"
            )
    for needle in scenario.final_contains:
        if needle.lower() not in final_text.lower():
            return False, f"final text missing {needle!r}: {final_text[:120]!r}"
    await recorder.aclose()
    return True, f"calls={[(c[0]) for c in recorder.calls]} final={final_text[:60]!r}"


async def run_all(model: Any, scenarios: tuple[Scenario, ...], verbose: bool) -> int:
    width = max(len(s.id) for s in scenarios) + 2
    failures = 0
    print(f"{'scenario':<{width}}result  detail")
    print("-" * (width + 70))
    for scenario in scenarios:
        ok, detail = await run_scenario(model, scenario)
        failures += 0 if ok else 1
        print(f"{scenario.id:<{width}}{'PASS ' if ok else 'FAIL '}  {detail}")
        if verbose:
            for i, turn in enumerate(scenario.turns):
                print(f"    turn {i}: {turn!r}")
    print("-" * (width + 70))
    total = len(scenarios)
    print(f"{total - failures}/{total} passed" + (f" — {failures} FAILED" if failures else ""))
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    import asyncio

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--only", default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    model = _P1.build_eval_model(args.model, args.temperature)
    scenarios = SCENARIOS if not args.only else tuple(s for s in SCENARIOS if args.only in s.id)
    if not scenarios:
        print(f"no scenario matches --only {args.only!r}")
        return 2
    return asyncio.run(run_all(model, scenarios, verbose=args.verbose))


if __name__ == "__main__":
    sys.exit(main())

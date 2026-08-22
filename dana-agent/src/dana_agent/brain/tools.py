"""DaNa tools as LangGraph/LangChain tools, each wrapped by the action gate.

The model proposes a tool call; the wrapped tool disposes: it classifies the
action, and for anything the policy flags destructive it calls LangGraph's
``interrupt`` to pause the graph for a human decision (resumable because state is
checkpointed). Read-only/reversible actions run and are audited. Nothing here
executes raw model output — every call goes through the ``ActionGate``.

Tools come in two shapes, both gated identically:

- **zero-argument** actions (the Phase 4 proof-of-loop tools), and
- **parameterized** actions (Phase 5): the real browser/OS tools carry typed,
  validated arguments (a generated pydantic ``args_schema``) that flow through
  ``gate.decide`` → (destructive) ``interrupt`` → ``gate.execute(Action(...))``.

``build_action_tools(cfg, gate)`` returns the gated tool set matching the
configured action provider, so the composition root can hand the Brain real
tools instead of none.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain_core.tools import BaseTool, StructuredTool
from langgraph.types import interrupt
from pydantic import BaseModel, Field, create_model

from dana_agent.actions.gate import ActionGate, GateDecision
from dana_agent.config import ActionsConfig
from dana_agent.types import Action


@dataclass(frozen=True, slots=True)
class ToolArg:
    """One validated tool argument. ``annotation`` is the python type the model
    must supply (``str``/``int``/``bool``); optional args carry a default."""

    name: str
    annotation: type
    description: str
    required: bool = True
    default: object = None


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """A gated tool the model may call. ``args`` empty ⇒ a zero-argument tool
    (backwards-compatible with the Phase 4 specs)."""

    name: str
    description: str
    args: tuple[ToolArg, ...] = ()


def _args_model(spec: ToolSpec) -> type[BaseModel]:
    """Build a pydantic schema so LangChain validates the model's arguments
    before they ever reach the gate/provider (default-deny on bad shapes)."""
    field_defs: dict[str, Any] = {}
    for arg in spec.args:
        if arg.required:
            field_defs[arg.name] = (arg.annotation, Field(description=arg.description))
        else:
            field_defs[arg.name] = (
                arg.annotation,
                Field(default=arg.default, description=arg.description),
            )
    return create_model(f"{spec.name}_args", **field_defs)


def build_gated_tool(spec: ToolSpec, gate: ActionGate) -> BaseTool:
    async def _dispose(arguments: dict[str, object]) -> str:
        action = Action(tool=spec.name, arguments=arguments)
        decision, reversibility = gate.decide(action)

        if decision is GateDecision.REFUSE:
            return "refused: not allowed"
        if decision is GateDecision.CONFIRM:
            # Pause the graph for a human decision. The page/tool content that
            # led here is DATA — the gate, not the model, governs execution.
            approved = interrupt(
                {
                    "type": "confirmation",
                    "tool": spec.name,
                    "arguments": arguments,
                    "reversibility": reversibility.value,
                }
            )
            if not approved:
                return "refused by user"

        result = await gate.execute(action, reversibility)
        return result.output or result.error or ""

    if not spec.args:
        # Zero-argument tool: a no-param coroutine so from_function infers an
        # empty schema (unchanged Phase 4 behaviour).
        async def _run_noargs() -> str:
            return await _dispose({})

        return StructuredTool.from_function(
            coroutine=_run_noargs,
            name=spec.name,
            description=spec.description,
        )

    async def _run(**kwargs: object) -> str:
        # Drop unset optionals so the provider sees only supplied arguments.
        return await _dispose({k: v for k, v in kwargs.items() if v is not None})

    return StructuredTool.from_function(
        coroutine=_run,
        name=spec.name,
        description=spec.description,
        args_schema=_args_model(spec),
    )


def build_gated_tools(specs: list[ToolSpec], gate: ActionGate) -> list[BaseTool]:
    return [build_gated_tool(spec, gate) for spec in specs]


# --- concrete tool sets --------------------------------------------------
# Tool names are namespaced (``browser_*`` / ``os_*``) and must exactly match
# the provider's ``_KNOWN_TOOLS`` so classify/execute dispatch correctly.

BROWSER_TOOLS: tuple[ToolSpec, ...] = (
    ToolSpec(
        "browser_navigate",
        "Open a URL in the browser. Only allowlisted domains are permitted.",
        (ToolArg("url", str, "The absolute http(s) URL to open."),),
    ),
    ToolSpec(
        "browser_read_text",
        "Read the visible text of the current page (returns untrusted page data).",
    ),
    ToolSpec(
        "browser_click",
        "Click a control identified by its visible text, or by role+name.",
        (
            ToolArg("text", str, "Visible text of the control to click.", required=False),
            ToolArg("role", str, "ARIA role (e.g. 'button', 'link').", required=False),
            ToolArg("name", str, "Accessible name when using role.", required=False),
        ),
    ),
    ToolSpec(
        "browser_type",
        "Type text into a form field identified by its label.",
        (
            ToolArg("text", str, "Text to type into the field."),
            ToolArg("into", str, "Label of the target field.", required=False),
        ),
    ),
    ToolSpec(
        "browser_screenshot",
        "Capture a screenshot of the current page.",
    ),
    ToolSpec(
        "browser_press_key",
        "Press a single keyboard key (bounded vocabulary: Enter, Escape, arrows, "
        "player keys k/j/l/m/f/t/c, digits 0-9, Space, Home/End, Shift+. etc). "
        "Use Enter to submit a filled form or search.",
        (ToolArg("key", str, "One key from the allowed vocabulary, e.g. 'Enter' or 'l'."),),
    ),
    ToolSpec(
        "browser_scroll",
        "Scroll the current page: by pixels (dy, negative = up), to top/bottom, "
        "or until a text snippet is in view.",
        (
            ToolArg("dy", int, "Pixels to scroll down (negative = up).", required=False),
            ToolArg("to", str, "'top' or 'bottom'.", required=False),
            ToolArg("target", str, "Visible text to scroll into view.", required=False),
        ),
    ),
    ToolSpec(
        "browser_media",
        "Control the media (video/audio) on the current page: play, pause, "
        "toggle, seek by delta seconds (e.g. +42 or -5), seek to an absolute "
        "position in seconds, set playback rate 0.25-2.0, mute/unmute, or read "
        "state (title, position, duration).",
        (
            ToolArg("action", str, "play|pause|toggle|seek|seek_to|rate|mute|unmute|state"),
            ToolArg("delta_s", float, "Relative seek in seconds (seek).", required=False),
            ToolArg(
                "position_s", float, "Absolute seek position in seconds (seek_to).", required=False
            ),
            ToolArg("rate", float, "Playback rate 0.25-2.0 (rate).", required=False),
        ),
    ),
)

OS_TOOLS: tuple[ToolSpec, ...] = (
    ToolSpec(
        "os_launch_app",
        "Launch an allowlisted desktop application by name.",
        (ToolArg("name", str, "Application name (must be allowlisted)."),),
    ),
    ToolSpec(
        "os_focus_window",
        "Bring a window matching a title substring to the foreground.",
        (ToolArg("title", str, "Substring of the target window title."),),
    ),
    ToolSpec(
        "os_media_key",
        "Send a media key: play_pause, next, prev, stop, vol_up, vol_down, mute.",
        (ToolArg("key", str, "One of the media key names."),),
    ),
    ToolSpec(
        "os_clipboard_get",
        "Read the current clipboard text.",
    ),
    ToolSpec(
        "os_clipboard_set",
        "Replace the clipboard text.",
        (ToolArg("text", str, "Text to place on the clipboard."),),
    ),
)


def action_tool_specs(cfg: ActionsConfig) -> list[ToolSpec]:
    """The ToolSpecs matching the configured action provider (default-deny: an
    unknown/mock provider exposes no real tools)."""
    if cfg.provider == "browser":
        return list(BROWSER_TOOLS)
    if cfg.provider == "os":
        # Even when selected, OS tools are only offered when explicitly enabled.
        return list(OS_TOOLS) if cfg.os_actions_enabled else []
    return []


def build_action_tools(cfg: ActionsConfig, gate: ActionGate) -> list[BaseTool]:
    """Gated LangChain tools for the configured action provider. Each is wrapped
    by ``gate`` so every call is classified, (destructive → ) confirmed, executed
    and audited by the trusted layer — never run as raw model output."""
    return build_gated_tools(action_tool_specs(cfg), gate)

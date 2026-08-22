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

import time
from dataclasses import dataclass
from typing import Any

from langchain_core.tools import BaseTool, StructuredTool
from langgraph.types import interrupt
from pydantic import BaseModel, Field, create_model

from dana_agent.actions.gate import ActionGate, GateDecision
from dana_agent.config import ActionsConfig
from dana_agent.observability import get_logger
from dana_agent.types import Action

log = get_logger(__name__)


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

        t0 = time.perf_counter()
        result = await gate.execute(action, reversibility)
        # Same "latency" event the conversation observer emits — the log bridge
        # turns it into a latency_span so the dashboard waterfall shows action
        # cost per turn.
        log.info(
            "latency",
            stage="action",
            tool=spec.name,
            ms=round((time.perf_counter() - t0) * 1000, 1),
        )
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
        "or until a text snippet is in view. Prefer target= when the user names "
        "a section (e.g. 'the comments', 'phần bình luận').",
        (
            ToolArg("dy", int, "Pixels to scroll down (negative = up).", required=False),
            ToolArg("to", str, "'top' or 'bottom'.", required=False),
            ToolArg("target", str, "Visible text to scroll into view.", required=False),
        ),
    ),
    ToolSpec(
        "browser_media",
        "Control the media (video/audio) on the current page. seek = RELATIVE "
        "move by delta seconds (+42 forward, -5 back, e.g. 'jump forward 40 "
        "seconds' or 'go back a bit'); seek_to = ABSOLUTE position in seconds "
        "(e.g. 'skip to 3:20' → position_s=200). Also: play, pause, toggle, "
        "rate 0.25-2.0, mute/unmute, next/prev (jumps via the site's player "
        "control where mapped, e.g. YouTube; otherwise use os_media_key), and "
        "state (title/position/duration).",
        (
            ToolArg(
                "action", str, "play|pause|toggle|seek|seek_to|rate|mute|unmute|next|prev|state"
            ),
            ToolArg(
                "delta_s",
                float,
                "Relative seek in seconds; negative = back. 'jump forward 40 "
                "seconds' → 40; 'tua lui 42 giây' / 'go back 42 seconds' → -42.",
                required=False,
            ),
            ToolArg(
                "position_s", float, "Absolute seek position in seconds (seek_to).", required=False
            ),
            ToolArg("rate", float, "Playback rate 0.25-2.0 (rate).", required=False),
        ),
    ),
    ToolSpec(
        "browser_tab",
        "Manage browser tabs: open a new one (optionally at a URL), list them, "
        "switch by title substring or 1-based index, or close one.",
        (
            ToolArg("action", str, "new|list|switch|close"),
            ToolArg("url", str, "URL for a new tab.", required=False),
            ToolArg("title", str, "Title/URL substring of the tab.", required=False),
            ToolArg("index", int, "1-based tab number.", required=False),
        ),
    ),
    ToolSpec(
        "browser_history",
        "Session/visit history: go back or forward, list recent visits "
        "(oldest→newest), or re-open one by its 1-based index.",
        (
            ToolArg("action", str, "back|forward|list|goto"),
            ToolArg("index", int, "1-based visit number (goto).", required=False),
        ),
    ),
    ToolSpec(
        "browser_search",
        "Search on an allowlisted engine (youtube, google, duckduckgo, wikipedia) "
        "in one step. With open_first=True, click the first result too — e.g. "
        "'play lofi on YouTube' is one call.",
        (
            ToolArg("query", str, "The search query."),
            ToolArg("engine", str, "youtube|google|duckduckgo|wikipedia"),
            ToolArg("open_first", bool, "Also click the first result.", required=False),
        ),
    ),
    ToolSpec(
        "browser_get_state",
        "Read the current page: title, URL, tab count, and media state — use it "
        "to answer 'what's playing' or verify a step.",
    ),
    ToolSpec(
        "browser_download",
        "Download a file from an allowlisted URL into DaNa's downloads folder. "
        "Executable files are refused. Report the saved path to the user.",
        (ToolArg("url", str, "Direct http(s) URL of the file (must be allowlisted)."),),
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
        "Send a media key: play_pause, next, prev, stop, vol_up, vol_down, mute. "
        "Use next/prev for 'next/previous video or song' — it hits whatever owns "
        "the system media session (browser tab or desktop app).",
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
    ToolSpec(
        "os_volume",
        "Get or set the system master volume (0-100), or mute/unmute it.",
        (
            ToolArg("action", str, "get|set|mute|unmute"),
            ToolArg("percent", int, "Volume level 0-100 (set).", required=False),
        ),
    ),
    ToolSpec(
        "os_window",
        "Manage a window (by title substring, else the foreground one): snap to "
        "screen halves, minimize, maximize, restore, or close it.",
        (
            ToolArg("action", str, "snap_left|snap_right|minimize|maximize|restore|close"),
            ToolArg("title", str, "Window title substring.", required=False),
        ),
    ),
    ToolSpec(
        "os_media_state",
        "What is playing system-wide right now (title/artist/status of the "
        "current media session, any app) — answers 'what song is this?'.",
    ),
)


def action_tool_specs(cfg: ActionsConfig) -> list[ToolSpec]:
    """The ToolSpecs matching the configured action provider (default-deny: an
    unknown/mock provider exposes no real tools)."""
    os_tools = list(OS_TOOLS) if cfg.os_actions_enabled else []
    if cfg.provider == "browser":
        return list(BROWSER_TOOLS)
    if cfg.provider == "os":
        return os_tools
    if cfg.provider == "browser_os":
        # Browser + OS together: both surfaces, OS still opt-in.
        return list(BROWSER_TOOLS) + os_tools
    return []


def build_action_tools(cfg: ActionsConfig, gate: ActionGate) -> list[BaseTool]:
    """Gated LangChain tools for the configured action provider. Each is wrapped
    by ``gate`` so every call is classified, (destructive → ) confirmed, executed
    and audited by the trusted layer — never run as raw model output."""
    return build_gated_tools(action_tool_specs(cfg), gate)

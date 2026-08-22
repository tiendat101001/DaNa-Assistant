"""Standalone transcript→action routing eval (TASK-012).

Verifies the *selection* half of the action pipeline with the REAL brain model:
an STT-style transcript goes in, and we assert the model picks the expected
gated tool (and arguments) for it. Nothing executes — the tools are bound for
their schemas only, so no browser launches and no OS call fires.

Run (from dana-agent/):

    uv run python scripts/action_routing_eval.py                 # all 20 cases
    uv run python scripts/action_routing_eval.py --only youtube  # subset
    uv run python scripts/action_routing_eval.py --model gemini-2.5-flash-lite

Model/key come from .env (DANA_CONVERSATION__LLM__*) exactly as in production;
--model/--temperature override for the eval (temperature 0 by default for
determinism). Exit code 0 only if every case passes — CI-safe to wire later as
`pytest -m live` (see tests/test_action_routing_live.py).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

# --- tiny predicate DSL --------------------------------------------------------

ArgCheck = Callable[[Any], bool]


def eq(expected: Any) -> ArgCheck:
    return lambda v: v == expected


def approx(expected: float, tol: float) -> ArgCheck:
    return lambda v: isinstance(v, (int, float)) and abs(float(v) - expected) <= tol


def contains_ci(*needles: str) -> ArgCheck:
    def _check(v: Any) -> bool:
        s = str(v).lower()
        return any(n in s for n in needles)

    return _check


def gt_(bound: float) -> ArgCheck:
    return lambda v: isinstance(v, (int, float)) and float(v) > bound


def one_of(*values: str) -> ArgCheck:
    return lambda v: v in values


TRUTHY: ArgCheck = lambda v: bool(v) is True  # noqa: E731


@dataclass(frozen=True, slots=True)
class Accepted:
    """One acceptable tool choice: name + per-arg predicates (missing arg ⇒ fail
    that predicate unless it is optional — declare Optional separately)."""

    tool: str
    args: dict[str, ArgCheck] = field(default_factory=dict)

    def matches(self, name: str, arguments: dict[str, Any]) -> tuple[bool, str]:
        if name != self.tool:
            return False, f"tool {name!r} != {self.tool!r}"
        for key, check in self.args.items():
            if key not in arguments:
                return False, f"missing arg {key!r}"
            if not check(arguments[key]):
                return False, f"arg {key}={arguments[key]!r} failed check"
        return True, ""


@dataclass(frozen=True, slots=True)
class Case:
    id: str
    utterance: str  # what STT would hand the brain
    lang: str
    accept: tuple[Accepted, ...]  # any single match passes

    def describe(self) -> str:
        return " | ".join(a.tool + (f"({','.join(a.args)})" if a.args else "") for a in self.accept)


# --- the 20 routing cases (EN + VI, from docs/ACTIONS-USE-CASES.md) ------------

SEARCH_LOFI = (
    Accepted(
        "browser_search",
        {
            "engine": eq("youtube"),
            "query": contains_ci("lofi"),
            "open_first": TRUTHY,
        },
    ),
)

CASES: tuple[Case, ...] = (
    Case("01-youtube-play-en", "Open YouTube and play lofi hip hop music", "en", SEARCH_LOFI),
    Case("02-youtube-play-vi", "Mở YouTube và phát nhạc lofi hip hop", "vi", SEARCH_LOFI),
    Case(
        "03-pause-en",
        "Pause the video",
        "en",
        (Accepted("browser_media", {"action": one_of("pause", "toggle")}),),
    ),
    Case(
        "04-pause-vi",
        "Tạm dừng video",
        "vi",
        (Accepted("browser_media", {"action": one_of("pause", "toggle")}),),
    ),
    Case(
        "05-seek-fwd-42-en",
        "Jump forward 42 seconds",
        "en",
        (Accepted("browser_media", {"action": eq("seek"), "delta_s": approx(42, 1)}),),
    ),
    Case(
        "06-seek-back-42-vi",
        "Tua lui 42 giây",
        "vi",
        (Accepted("browser_media", {"action": eq("seek"), "delta_s": approx(-42, 1)}),),
    ),
    Case(
        "07-seek-absolute-en",
        "Skip to 3 minutes 20 seconds",
        "en",
        (Accepted("browser_media", {"action": eq("seek_to"), "position_s": approx(200, 15)}),),
    ),
    Case(
        "08-rate-vi",
        "Phát với tốc độ 1.5 lần",
        "vi",
        (Accepted("browser_media", {"action": eq("rate"), "rate": approx(1.5, 0.05)}),),
    ),
    Case(
        "09-mute-en",
        "Mute the video",
        "en",
        (
            Accepted("browser_media", {"action": eq("mute")}),
            Accepted("os_volume", {"action": eq("mute")}),
        ),
    ),
    Case(
        "10-whats-playing-vi",
        "Đang phát gì vậy?",
        "vi",
        (
            Accepted("browser_media", {"action": eq("state")}),
            Accepted("browser_get_state"),
            # System-wide answer is equally correct for "what's playing".
            Accepted("os_media_state"),
        ),
    ),
    Case(
        "11-next-en",
        "Next video please",
        "en",
        (
            Accepted("browser_media", {"action": eq("next")}),
            Accepted("os_media_key", {"key": eq("next")}),
            Accepted("browser_click", {"text": contains_ci("next")}),
        ),
    ),
    Case(
        "12-wikipedia-en",
        "Search Wikipedia for the Paracel Islands",
        "en",
        (Accepted("browser_search", {"engine": eq("wikipedia"), "query": contains_ci("paracel")}),),
    ),
    Case(
        "13-wikipedia-vi",
        "Tra cứu trên Wikipedia về trường Sa",
        "vi",
        (
            Accepted(
                "browser_search",
                {"engine": eq("wikipedia"), "query": contains_ci("paracel", "trường")},
            ),
        ),
    ),
    Case(
        "14-google-en",
        "Google the weather in Hanoi",
        "en",
        (Accepted("browser_search", {"engine": eq("google"), "query": contains_ci("hanoi")}),),
    ),
    Case(
        "15-scroll-down-en",
        "Scroll down a bit",
        "en",
        (
            Accepted("browser_scroll", {"dy": gt_(0)}),
            Accepted("browser_scroll", {"to": eq("bottom")}),
        ),
    ),
    Case(
        "16-scroll-comments-vi",
        "Kéo xuống phần bình luận",
        "vi",
        (
            Accepted("browser_scroll", {"target": contains_ci("comment", "bình luận")}),
            # Comments sit at page bottom — 'to bottom' is a fair reading.
            Accepted("browser_scroll", {"to": eq("bottom")}),
        ),
    ),
    Case(
        "17-new-tab-en",
        "Open a new browser tab",
        "en",
        (Accepted("browser_tab", {"action": eq("new")}),),
    ),
    Case(
        "18-history-back-vi",
        "Quay lại trang trước",
        "vi",
        (Accepted("browser_history", {"action": eq("back")}),),
    ),
    Case(
        "19-volume-en",
        "Set the volume to 30 percent",
        "en",
        (Accepted("os_volume", {"action": eq("set"), "percent": approx(30, 2)}),),
    ),
    Case(
        "20-screenshot-vi",
        "Chụp màn hình trang này lại",
        "vi",
        (Accepted("browser_screenshot"),),
    ),
    Case(
        "21-prev-vi",
        "Video trước đó",
        "vi",
        (
            Accepted("browser_media", {"action": eq("prev")}),
            Accepted("os_media_key", {"key": eq("prev")}),
        ),
    ),
    Case(
        "22-read-page-vi",
        "Đọc nội dung trang này",
        "vi",
        (Accepted("browser_read_text"),),
    ),
    Case(
        "23-clipboard-set-en",
        "Copy this into my clipboard: meeting notes at 3pm",
        "en",
        (Accepted("os_clipboard_set", {"text": contains_ci("meeting")}),),
    ),
    Case(
        "24-clipboard-get-vi",
        "Trong clipboard có gì vậy?",
        "vi",
        (Accepted("os_clipboard_get"),),
    ),
    Case(
        "25-minimize-vi",
        "Thu nhỏ cửa sổ này lại",
        "vi",
        (Accepted("os_window", {"action": eq("minimize")}),),
    ),
    Case(
        "26-snap-en",
        "Snap the Discord window to the right half of the screen",
        "en",
        (Accepted("os_window", {"action": eq("snap_right"), "title": contains_ci("discord")}),),
    ),
    Case(
        "27-sysmute-vi",
        "Tắt tiếng toàn hệ thống",
        "vi",
        (
            Accepted("os_volume", {"action": eq("mute")}),
            Accepted("os_media_key", {"key": eq("mute")}),
        ),
    ),
    Case(
        "28-unmute-video-en",
        "Unmute the video",
        "en",
        (
            Accepted("browser_media", {"action": eq("unmute")}),
            Accepted("os_volume", {"action": eq("unmute")}),
        ),
    ),
    Case(
        "29-type-en",
        "Type 'hello world' into the search box",
        "en",
        (
            Accepted(
                "browser_type",
                {"text": contains_ci("hello"), "into": contains_ci("search")},
            ),
        ),
    ),
    Case(
        "30-enter-vi",
        "Nhấn phím Enter",
        "vi",
        (Accepted("browser_press_key", {"key": eq("Enter")}),),
    ),
    Case(
        "31-close-tab-vi",
        "Đóng tab này đi",
        "vi",
        (Accepted("browser_tab", {"action": eq("close")}),),
    ),
    Case(
        "32-switch-tab-en",
        "Switch to the YouTube tab",
        "en",
        (Accepted("browser_tab", {"action": eq("switch"), "title": contains_ci("youtube")}),),
    ),
    Case(
        "33-forward-en",
        "Go forward a page",
        "en",
        (Accepted("browser_history", {"action": eq("forward")}),),
    ),
    Case(
        "34-volume-50-vi",
        "Tăng âm lượng hệ thống lên 50",
        "vi",
        (Accepted("os_volume", {"action": eq("set"), "percent": approx(50, 2)}),),
    ),
    Case(
        "35-ddg-en",
        "Search DuckDuckGo for raspberry pi",
        "en",
        (
            Accepted(
                "browser_search", {"engine": eq("duckduckgo"), "query": contains_ci("raspberry")}
            ),
        ),
    ),
    Case(
        "36-scroll-top-vi",
        "Kéo lên đầu trang",
        "vi",
        (
            Accepted("browser_scroll", {"to": eq("top")}),
            Accepted(
                "browser_scroll", {"dy": lambda v: isinstance(v, (int, float)) and float(v) < 0}
            ),
        ),
    ),
    Case(
        "37-seek-7min-vi",
        "Tua nhanh đến phút thứ 7",
        "vi",
        (Accepted("browser_media", {"action": eq("seek_to"), "position_s": approx(420, 15)}),),
    ),
    Case(
        "38-screenshot-en",
        "Take a screenshot of this page",
        "en",
        (Accepted("browser_screenshot"),),
    ),
    Case(
        "39-time-left-en",
        "How much time is left in this video?",
        "en",
        (
            Accepted("browser_media", {"action": eq("state")}),
            Accepted("browser_get_state"),
        ),
    ),
    Case(
        "40-resume-vi",
        "Phát tiếp đi",
        "vi",
        (Accepted("browser_media", {"action": one_of("play", "toggle")}),),
    ),
    Case(
        "41-arxiv-search-en",
        "Find the attention is all you need paper on arxiv",
        "en",
        (
            Accepted(
                "browser_search",
                {"engine": eq("arxiv"), "query": contains_ci("attention")},
            ),
        ),
    ),
    Case(
        "42-translate-abstract-vi",
        "Dịch tóm tắt của bài báo này sang tiếng Việt",
        "vi",
        (Accepted("browser_read_text"),),
    ),
    Case(
        "43-zoom-en",
        "Zoom in a bit, this formula is tiny",
        "en",
        (Accepted("browser_zoom", {"action": eq("in")}),),
    ),
)


# --- runner ---------------------------------------------------------------------


def build_eval_model(model_override: str | None, temperature: float) -> Any:
    """The brain's chat model, exactly as production builds it (config from
    .env via get_settings), with eval-time overrides."""
    from dana_agent.brain.model import build_chat_model
    from dana_agent.config import LLMConfig, get_settings

    llm = get_settings().conversation.llm
    cfg = LLMConfig(
        provider=llm.provider,
        model=model_override or llm.model,
        api_key=llm.api_key,
        base_url=llm.base_url,
        temperature=temperature,
    )
    if cfg.provider == "mock":
        raise SystemExit(
            "no real LLM configured: set DANA_CONVERSATION__LLM__PROVIDER/MODEL/API_KEY "
            "in dana-agent/.env (this eval calls the real model)"
        )
    return build_chat_model(cfg)


def build_bound_tools() -> Any:
    """The real gated tool surface (schemas only — nothing here executes)."""
    from dana_agent.actions.gate import ActionGate
    from dana_agent.brain.tools import build_action_tools
    from dana_agent.config import ActionsConfig
    from dana_agent.providers.registry import build_actions

    cfg = ActionsConfig(
        provider="browser_os",
        os_actions_enabled=True,
        allowed_domains=["youtube.com", "google.com", "duckduckgo.com", "wikipedia.org"],
    )
    gate = ActionGate(build_actions(ActionsConfig(provider="mock")))
    return build_action_tools(cfg, gate)


# Standing context every case runs under — mirrors a live DaNa session where
# the browser is open with the previous turn's page/media still active (an STT
# transcript arrives without history, but the session state is real).
_SESSION_CONTEXT = (
    "Session context: DaNa is running with the user's dedicated browser open; "
    "the page and media from the previous turn are still active unless the "
    "utterance clearly implies otherwise (e.g. asking to open something new)."
)


def evaluate_case(model: Any, tools: Any, case: Case, retries: int = 2) -> tuple[bool, str]:
    """One transcript → one model turn → first tool call vs. accept set."""
    from langchain_core.messages import HumanMessage, SystemMessage

    from dana_agent.config import BrainConfig

    system = f"{BrainConfig().system_prompt}\n\n{_SESSION_CONTEXT}"
    bound = model.bind_tools(tools)
    last_error = ""
    for attempt in range(retries + 1):
        try:
            msg = bound.invoke(
                [SystemMessage(content=system), HumanMessage(content=case.utterance)]
            )
            calls = list(getattr(msg, "tool_calls", None) or [])
            if not calls:
                text = str(getattr(msg, "content", ""))[:100]
                return False, f"no tool call — replied: {text!r}"
            name = str(calls[0]["name"])
            args = dict(calls[0]["args"] or {})
            for accepted in case.accept:
                ok, why = accepted.matches(name, args)
                if ok:
                    return True, f"{name} {json.dumps(args, ensure_ascii=False, default=str)}"
            return (
                False,
                f"got {name} {json.dumps(args, ensure_ascii=False, default=str)}",
            )
        except Exception as exc:  # noqa: BLE001 — surface provider hiccups as retries
            last_error = f"{type(exc).__name__}: {exc}"
            time.sleep(1.5 * (attempt + 1))
    return False, f"model error after {retries + 1} attempts: {last_error}"


def run_eval(model: Any, tools: Any, cases: tuple[Case, ...] = CASES, verbose: bool = False) -> int:
    width = max(len(c.id) for c in cases) + 2
    failures = 0
    print(f"{'case':<{width}}lang  result  detail")
    print("-" * (width + 60))
    for case in cases:
        ok, detail = evaluate_case(model, tools, case)
        failures += 0 if ok else 1
        mark = "PASS " if ok else "FAIL "
        print(f"{case.id:<{width}}{case.lang:<6}{mark}  {detail}")
        if verbose:
            print(f"    utterance: {case.utterance!r}\n    accept:     {case.describe()}")
    total = len(cases)
    print("-" * (width + 60))
    print(f"{total - failures}/{total} passed" + (f" — {failures} FAILED" if failures else ""))
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", default=None, help="override the .env model (e.g. gemini-2.5-flash-lite)"
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--only", default=None, help="run cases whose id contains this substring")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    model = build_eval_model(args.model, args.temperature)
    tools = build_bound_tools()
    cases = CASES if not args.only else tuple(c for c in CASES if args.only in c.id)
    if not cases:
        print(f"no case matches --only {args.only!r}")
        return 2
    return run_eval(model, tools, cases, verbose=args.verbose)


if __name__ == "__main__":
    sys.exit(main())

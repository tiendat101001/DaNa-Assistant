"""Playwright browser actions — a bounded, allowlisted ``ActionProvider``.

The model proposes a structured tool call; this provider validates the target
against a default-deny domain allowlist and executes it with Playwright's
role/text locators (built-in auto-wait — never a sleep). It owns a *lazy*
persistent browser context on a DEDICATED profile dir under ``.dana/`` — never
the user's real Chrome profile — so cookies/state are isolated and disposable.

Trust boundary (safe-computer-use skill): this provider only ``classify()``s
honestly and ``execute()``s an already-authorised action. It does NOT gate or
confirm — ``ActionGate`` is the trusted layer. ``OUT_OF_SCOPE`` is how the
provider says "not allowlisted → refuse". Page text returned by ``read_text`` is
untrusted *data*, never instructions.

Heavy import guard (mirrors ``providers/tts/gemini.py``): ``playwright`` is
imported lazily inside a method, so the module imports offline with only stdlib +
dana types and the offline test suite stays green with no browser installed.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from dana_agent.config import ActionsConfig
from dana_agent.providers.base import ActionProvider, Capabilities
from dana_agent.providers.registry import register
from dana_agent.types import Action, ActionResult, Reversibility

_NAV_TIMEOUT_MS = 15_000
# Truncate page text: bounds context growth AND the prompt-injection surface.
_READ_TEXT_LIMIT = 8_000
# Click labels that commit an irreversible/consequential action → DESTRUCTIVE.
_DESTRUCTIVE_CLICK_WORDS = (
    "submit",
    "pay",
    "buy",
    "delete",
    "send",
    "confirm",
    "place order",
    "checkout",
    "remove",
)
# The bounded, known tool set. Anything else is OUT_OF_SCOPE.
_KNOWN_TOOLS = frozenset(
    {"browser_navigate", "browser_click", "browser_type", "browser_read_text", "browser_screenshot"}
)


@register("actions", "browser")
class BrowserActions(ActionProvider):
    """Bounded Playwright automation on a dedicated, disposable profile."""

    def __init__(self, cfg: ActionsConfig) -> None:
        self._cfg = cfg
        # Normalise allowlist entries to bare lower-case hosts (drop any leading
        # dot) so suffix matching is exact and case-insensitive.
        self._allowed_hosts = frozenset(
            h.lower().strip().lstrip(".") for h in cfg.allowed_domains if h.strip()
        )
        self._profile_dir = Path(cfg.browser_profile_dir)
        self._storage_state = Path(cfg.browser_storage_state_path)
        self._headless = cfg.browser_headless
        self._pw: Any = None  # playwright instance (lazy)
        self._ctx: Any = None  # persistent BrowserContext (lazy)
        self._page: Any = None  # active Page (lazy)
        self._page_lock: Any = None  # asyncio.Lock, created on first use

    @property
    def capabilities(self) -> Capabilities:
        return Capabilities(
            streaming=False,
            languages=frozenset({"en", "vi"}),
            latency_class="batch",
        )

    # --- allowlist -------------------------------------------------------

    def _host_allowed(self, url: str) -> bool:
        """Default-deny host-suffix check.

        Rejects non-http(s) schemes outright (``file:``/``javascript:``/``data:``)
        and matches the parsed host (never a substring of the raw URL, which
        would let ``evil.com/google.com`` through). Empty allowlist ⇒ deny all.
        """
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            return False
        host = (parsed.hostname or "").lower()
        if not host or not self._allowed_hosts:
            return False
        # Suffix match: "mail.google.com" is covered by allowlist entry "google.com".
        return any(host == d or host.endswith("." + d) for d in self._allowed_hosts)

    # --- classification (honest, so the gate can decide) -----------------

    def classify(self, action: Action) -> Reversibility:
        tool = action.tool
        args = action.arguments
        if tool == "browser_navigate":
            url = str(args.get("url", ""))
            return (
                Reversibility.READ_ONLY if self._host_allowed(url) else Reversibility.OUT_OF_SCOPE
            )
        if tool in ("browser_read_text", "browser_screenshot"):
            return Reversibility.READ_ONLY
        if tool == "browser_type":
            return Reversibility.REVERSIBLE  # fills a field; nothing committed yet
        if tool == "browser_click":
            label = f"{args.get('name', '')} {args.get('text', '')}".lower()
            if any(word in label for word in _DESTRUCTIVE_CLICK_WORDS):
                return Reversibility.DESTRUCTIVE  # submit/pay/delete/send → confirm
            return Reversibility.REVERSIBLE  # generic UI/navigation click
        return Reversibility.OUT_OF_SCOPE  # anything outside the bounded set

    # --- lazy browser lifecycle -----------------------------------------

    async def _ensure_page(self) -> Any:
        """Lazily start Playwright + a persistent context on the dedicated
        profile dir, returning the active page. Setup happens inside the running
        loop on first ``execute`` (never at construction), mirroring gemini.py.

        An asyncio Lock prevents a concurrent second call from racing through the
        ``self._page is None`` check and starting a duplicate browser process."""
        import asyncio  # noqa: PLC0415

        if self._page_lock is None:
            self._page_lock = asyncio.Lock()
        async with self._page_lock:
            if self._page is not None:
                return self._page
            try:
                from playwright.async_api import async_playwright  # noqa: PLC0415
            except ImportError as exc:
                raise ImportError(
                    "playwright is required for the 'browser' action provider: "
                    "install dana-agent[actions] then run `playwright install chromium`"
                ) from exc

            self._profile_dir.mkdir(parents=True, exist_ok=True)
            self._pw = await async_playwright().start()
            # Persistent context on a DEDICATED profile dir isolates cookies/storage
            # from the user's daily Chrome. It returns the context itself (no separate
            # browser object; closing the context closes the browser).
            self._ctx = await self._pw.chromium.launch_persistent_context(
                user_data_dir=str(self._profile_dir),
                headless=self._headless,
                args=["--no-first-run", "--no-default-browser-check"],
            )
            self._ctx.set_default_timeout(_NAV_TIMEOUT_MS)  # auto-wait cap; NO sleeps
            self._page = self._ctx.pages[0] if self._ctx.pages else await self._ctx.new_page()
        return self._page

    async def aclose(self) -> None:
        if self._ctx is not None:
            # Export a portable auth blob (cookies + localStorage) for next run.
            # It contains live session secrets → stored under .dana/ (gitignored).
            try:
                self._storage_state.parent.mkdir(parents=True, exist_ok=True)
                await self._ctx.storage_state(path=str(self._storage_state))
            finally:
                await self._ctx.close()
        if self._pw is not None:
            await self._pw.stop()
        self._pw = self._ctx = self._page = None

    # --- execution (already-authorised action) ---------------------------

    async def execute(self, action: Action) -> ActionResult:
        if action.tool not in _KNOWN_TOOLS:
            return ActionResult(ok=False, error=f"unknown browser tool {action.tool!r}")
        page = await self._ensure_page()
        args = action.arguments
        try:
            return await self._dispatch(page, action.tool, args)
        except Exception as exc:  # noqa: BLE001 — structured error, never a raw traceback
            return ActionResult(ok=False, error=f"{type(exc).__name__}: {exc}")

    def _current_page_refusal(self, page: Any) -> ActionResult | None:
        """Refuse if the *current* page is off-allowlist. A prior navigation may
        have redirected (30x / meta-refresh / client-side JS) to an off-allowlist
        host, so the allowlist must be re-asserted against ``page.url`` before any
        content/interaction tool acts — the initial-URL check is not enough."""
        current = str(getattr(page, "url", "") or "")
        if not self._host_allowed(current):
            host = (urlparse(current).hostname or current).lower()
            return ActionResult(ok=False, error=f"refused: current page is off-allowlist: {host}")
        return None

    async def _dispatch(self, page: Any, tool: str, args: dict[str, object]) -> ActionResult:
        if tool == "browser_navigate":
            url = str(args.get("url", ""))
            if not self._host_allowed(url):
                return ActionResult(ok=False, error=f"refused: domain not allowlisted: {url}")
            await page.goto(url, wait_until="domcontentloaded")
            # Re-assert against the LANDED url: an allowlisted target that
            # redirects off-allowlist must be treated as a refusal, not exposed.
            landed = str(page.url)
            if not self._host_allowed(landed):
                host = (urlparse(landed).hostname or landed).lower()
                return ActionResult(ok=False, error=f"refused: redirected off-allowlist to {host}")
            return ActionResult(ok=True, output=f"navigated to {landed}")

        if tool == "browser_read_text":
            if (refusal := self._current_page_refusal(page)) is not None:
                return refusal
            # Accessibility/DOM text is cheaper and more precise than a screenshot.
            # Truncated + treated as untrusted DATA (never instructions).
            text = await page.inner_text("body")
            return ActionResult(ok=True, output=str(text)[:_READ_TEXT_LIMIT])

        if tool == "browser_click":
            if (refusal := self._current_page_refusal(page)) is not None:
                return refusal
            if "role" in args:
                loc = page.get_by_role(str(args["role"]), name=str(args.get("name", "")))
            elif "text" in args:
                loc = page.get_by_text(str(args["text"]), exact=False)
            else:
                return ActionResult(ok=False, error="refused: click needs 'role' or 'text'")
            await loc.first.click()  # auto-waits until actionable; no sleeps
            return ActionResult(ok=True, output="clicked")

        if tool == "browser_type":
            if (refusal := self._current_page_refusal(page)) is not None:
                return refusal
            if "into" in args:
                field = page.get_by_label(str(args["into"]))
            else:
                field = page.get_by_role("textbox")
            await field.first.fill(str(args.get("text", "")))  # clears+types, auto-waits
            return ActionResult(ok=True, output="typed")

        # tool == "browser_screenshot" (guaranteed by _KNOWN_TOOLS)
        if (refusal := self._current_page_refusal(page)) is not None:
            return refusal
        import time  # noqa: PLC0415

        png = await page.screenshot()

        def _write_screenshot(blob: bytes) -> str:
            out_dir = Path(".dana/screenshots")
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{int(time.time())}.png"
            out_path.write_bytes(blob)
            return str(out_path)

        return ActionResult(ok=True, output=await asyncio.to_thread(_write_screenshot, png))

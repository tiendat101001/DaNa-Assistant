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
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from dana_agent.config import ActionsConfig
from dana_agent.observability import get_logger
from dana_agent.providers.base import ActionProvider, Capabilities
from dana_agent.providers.registry import register
from dana_agent.types import Action, ActionResult, Reversibility

log = get_logger(__name__)

_NAV_TIMEOUT_MS = 15_000
# Truncate page text: bounds context growth AND the prompt-injection surface.
_READ_TEXT_LIMIT = 8_000
# --- downloads (UC-26) -----------------------------------------------------
_DOWNLOAD_DIR = ".dana/downloads"
_DOWNLOAD_MAX_BYTES = 200 * 1024 * 1024
# Executable/script payloads a voice-requested download must never save, by
# URL extension or Content-Type hint. Belt and braces: both are checked.
_DOWNLOAD_DENY_EXTS = frozenset(
    {
        "exe",
        "msi",
        "msix",
        "msp",
        "bat",
        "cmd",
        "com",
        "scr",
        "vbs",
        "vbe",
        "js",
        "jse",
        "wsf",
        "ps1",
        "jar",
        "hta",
        "lnk",
        "reg",
        "dll",
        "msc",
    }
)
_DOWNLOAD_DENY_CT_HINTS = ("executable", "msdownload", "x-dosexec", "powershell", "batch")
# Purchase-committing labels — a strict subset of the destructive words. When
# ``forbid_purchases`` is set these are REFUSED outright, not confirmed.
_PURCHASE_WORDS = (
    "pay",
    "buy",
    "checkout",
    "place order",
    "thanh toán",  # pay / checkout
    "đặt hàng",  # place order
    "đặt mua",  # place order (alt)
)
# Click labels that commit an irreversible/consequential action → DESTRUCTIVE.
# English + Vietnamese (vi labels must survive .lower(), which keeps diacritics).
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
    "gửi",  # send
    "xóa",  # delete
    "thanh toán",  # pay / checkout
    "đặt hàng",  # place order
    "đặt mua",  # place order (alt)
    "xác nhận",  # confirm
)
# The bounded, known tool set. Anything else is OUT_OF_SCOPE.
_KNOWN_TOOLS = frozenset(
    {
        "browser_navigate",
        "browser_click",
        "browser_type",
        "browser_read_text",
        "browser_screenshot",
        "browser_press_key",
        "browser_scroll",
        "browser_media",
        "browser_tab",
        "browser_history",
        "browser_search",
        "browser_get_state",
        "browser_download",
    }
)

# Bounded key vocabulary — a frozenset, never an arbitrary model string. Covers
# form/navigation keys, HTML5-player shortcut keys (k/j/l/m/f/t/c and digits),
# and a few explicit combos (YouTube speed, Netflix ±10 s variants).
_PRESS_KEYS: frozenset[str] = frozenset(
    {
        "Enter",
        "Escape",
        "Tab",
        "Backspace",
        "Delete",
        "Space",
        "ArrowLeft",
        "ArrowRight",
        "ArrowUp",
        "ArrowDown",
        "Home",
        "End",
        "PageUp",
        "PageDown",
        *(chr(d) for d in range(ord("0"), ord("9") + 1)),
        "k",
        "j",
        "l",
        "m",
        "f",
        "t",
        "c",
        ".",
        ",",
        "Shift+.",
        "Shift+,",
        "Shift+ArrowLeft",
        "Shift+ArrowRight",
    }
)

_MEDIA_ACTIONS: frozenset[str] = frozenset(
    {
        "play",
        "pause",
        "toggle",
        "seek",
        "seek_to",
        "rate",
        "mute",
        "unmute",
        "state",
        "next",
        "prev",
    }
)
_RATE_MIN, _RATE_MAX = 0.25, 2.0
# A relative seek is clamped to ±6 h — generous for any real media, and it
# bounds runaway model arithmetic.
_SEEK_DELTA_LIMIT_S = 6 * 60 * 60
_SCROLL_MAX_PX = 3_000

# Provider-owned canned JavaScript for the media engine. The model NEVER
# authors JS — it picks an action name; the script below is the entire
# executable surface, and only whitelisted numeric/bool arguments flow in.
# Each script carries a ``/* dana:<name> */`` marker (contract-tested offline).
_MEDIA_SCRIPTS: dict[str, str] = {
    "state": (
        "/* dana:state */ () => {"
        " const v = document.querySelector('video, audio');"
        " if (!v) return 'no-media';"
        " const fmt = (s) => isFinite(s) ? Math.floor(s/60) + ':' + "
        "String(Math.floor(s%60)).padStart(2,'0') : '?';"
        " return {"
        " title: (navigator.mediaSession && navigator.mediaSession.metadata"
        " && navigator.mediaSession.metadata.title) || document.title,"
        " playing: !v.paused, position: v.currentTime,"
        " duration: isFinite(v.duration) ? v.duration : null,"
        " rate: v.playbackRate, muted: v.muted,"
        " at: fmt(v.currentTime) + (isFinite(v.duration) ? ' / ' + fmt(v.duration) : '')"
        "}; }"
    ),
    "play": (
        "/* dana:play */ () => {"
        " const v = document.querySelector('video, audio');"
        " if (!v) return 'no-media'; v.play(); return 'ok'; }"
    ),
    "pause": (
        "/* dana:pause */ () => {"
        " const v = document.querySelector('video, audio');"
        " if (!v) return 'no-media'; v.pause(); return 'ok'; }"
    ),
    "toggle": (
        "/* dana:toggle */ () => {"
        " const v = document.querySelector('video, audio');"
        " if (!v) return 'no-media'; if (v.paused) { v.play(); } else { v.pause(); }"
        " return 'ok'; }"
    ),
    "seek": (
        "/* dana:seek */ ([d]) => {"
        " const v = document.querySelector('video, audio');"
        " if (!v) return 'no-media';"
        " const max = isFinite(v.duration) ? v.duration : Infinity;"
        " v.currentTime = Math.max(0, Math.min(max - 0.05, v.currentTime + d));"
        " return v.currentTime; }"
    ),
    "seek_to": (
        "/* dana:seek_to */ ([t]) => {"
        " const v = document.querySelector('video, audio');"
        " if (!v) return 'no-media';"
        " const max = isFinite(v.duration) ? v.duration : Infinity;"
        " v.currentTime = Math.max(0, Math.min(max - 0.05, t));"
        " return v.currentTime; }"
    ),
    "rate": (
        "/* dana:rate */ ([r]) => {"
        " const v = document.querySelector('video, audio');"
        " if (!v) return 'no-media'; v.playbackRate = r; return v.playbackRate; }"
    ),
    "mute": (
        "/* dana:mute */ ([m]) => {"
        " const v = document.querySelector('video, audio');"
        " if (!v) return 'no-media'; v.muted = m; return v.muted; }"
    ),
    # scroll helper shared by browser_scroll(to=...)
    "scroll_to": (
        "/* dana:scroll_to */ ([w]) => {"
        " if (w === 'top') { window.scrollTo({top: 0}); }"
        " else { window.scrollTo({top: document.documentElement.scrollHeight}); }"
        " return window.scrollY; }"
    ),
}
# The YouTube-style player key used as the last-resort play/pause fallback.
_PLAY_PAUSE_KEY = "k"


def _fmt_seconds(s: float) -> str:
    m, sec = divmod(int(s), 60)
    return f"{m}:{sec:02d}"


def _safe_download_name(url: str) -> str:
    """Filename from the URL path, reduced to boring characters only."""
    base = urlparse(url).path.rsplit("/", 1)[-1] or "download.bin"
    name = "".join(c if c.isalnum() or c in "-_." else "_" for c in base)[:120]
    return name.lstrip(".") or "download.bin"


def _speakable_media(result: Any) -> str:
    """Turn the state-script dict into a short, TTS-friendly string."""
    if not isinstance(result, dict):
        return str(result)
    title = str(result.get("title", "?"))
    verb = "playing" if result.get("playing") else "paused"
    at = str(result.get("at", "?"))
    rate = result.get("rate")
    suffix = f" ({rate}x)" if rate not in (None, 1, 1.0) else ""
    muted = " [muted]" if result.get("muted") else ""
    return f"{verb}: '{title}' at {at}{suffix}{muted}"


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
        self._tabs: list[Any] = []  # tracked Pages (tab manager, M2)
        # Append-only visit log (url/title/ts) — the substrate for "go back to
        # that video from earlier". Same trust level as the audit log.
        self._visits: list[dict[str, object]] = []
        self._max_visits = 100

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
        if tool == "browser_download":
            # Local file write from an allowlisted host — reversible (delete),
            # audited; the conservative policy confirms it (design doc §5).
            url = str(args.get("url", ""))
            return (
                Reversibility.REVERSIBLE if self._host_allowed(url) else Reversibility.OUT_OF_SCOPE
            )
        if tool == "browser_search":
            from dana_agent.providers.actions.adapters import search_url as _su  # noqa: PLC0415

            search_target = _su(str(args.get("engine", "")), str(args.get("query", "")))
            if search_target is None or not self._host_allowed(search_target):
                return Reversibility.OUT_OF_SCOPE  # unknown engine / off-allowlist
            if args.get("open_first"):
                return Reversibility.REVERSIBLE  # includes a generic result click
            return Reversibility.READ_ONLY  # navigation only
        if tool == "browser_history":
            hist_act = str(args.get("action", ""))
            if hist_act == "goto":
                # Re-opening a logged visit is a navigate → same allowlist rule.
                idx = args.get("index")
                visits = self._recent_visits()
                if isinstance(idx, int) and not isinstance(idx, bool) and 1 <= idx <= len(visits):
                    revisit = str(visits[idx - 1]["url"])
                    return (
                        Reversibility.READ_ONLY
                        if self._host_allowed(revisit)
                        else Reversibility.OUT_OF_SCOPE
                    )
                return Reversibility.OUT_OF_SCOPE
            return Reversibility.READ_ONLY  # back/forward/list: transient navigation
        if tool == "browser_tab":
            tab_act = str(args.get("action", ""))
            if tab_act in ("new", "close"):
                return Reversibility.REVERSIBLE  # local, recoverable (history/reopen)
            if tab_act in ("list", "switch"):
                return Reversibility.READ_ONLY
            return Reversibility.OUT_OF_SCOPE
        if tool in ("browser_read_text", "browser_screenshot", "browser_get_state"):
            return Reversibility.READ_ONLY
        if tool in ("browser_media", "browser_scroll"):
            # Transport control and scrolling are transient (no lasting state,
            # same class as os_media_key); `state` is a pure read.
            return Reversibility.READ_ONLY
        if tool == "browser_press_key":
            # REVERSIBLE, not read-only: Enter can commit what browser_type
            # filled (a search submit is the common case, a message send the
            # rare one). Standard policy runs+audits it; conservative confirms.
            return Reversibility.REVERSIBLE
        if tool == "browser_type":
            return Reversibility.REVERSIBLE  # fills a field; nothing committed yet
        if tool == "browser_click":
            label = f"{args.get('name', '')} {args.get('text', '')}".lower()
            if self._cfg.forbid_purchases and any(word in label for word in _PURCHASE_WORDS):
                return Reversibility.OUT_OF_SCOPE  # purchases forbidden → refuse
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
                # Downloads only via the explicit browser_download tool (M-later);
                # never as a side effect of a click.
                accept_downloads=False,
                args=["--no-first-run", "--no-default-browser-check"],
            )

            def _on_popup(popup: Any) -> None:
                # An allowlisted page can still be coerced into opening windows:
                # every new page gets the allowlist re-asserted, and an
                # off-allowlist popup is closed immediately. An allowlisted
                # popup becomes a tracked tab.
                url = str(getattr(popup, "url", "") or "")
                if url and url != "about:blank" and not self._host_allowed(url):
                    log.warning("popup_blocked_off_allowlist", url=url)
                    try:
                        popup.close()
                    except Exception:  # noqa: BLE001 — best-effort guard, never fatal
                        log.debug("popup_close_failed")
                    return
                if popup not in self._tabs:
                    self._tabs.append(popup)

            self._ctx.on("page", _on_popup)
            self._ctx.set_default_timeout(_NAV_TIMEOUT_MS)  # auto-wait cap; NO sleeps
            self._page = self._ctx.pages[0] if self._ctx.pages else await self._ctx.new_page()
            self._tabs = [self._page]
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

    async def _goto(self, page: Any, url: str) -> ActionResult:
        """Allowlist-checked navigation (shared by navigate/search/history-goto).
        Records the landed visit in the provider's visit log."""
        if not self._host_allowed(url):
            return ActionResult(ok=False, error=f"refused: domain not allowlisted: {url}")
        await page.goto(url, wait_until="domcontentloaded")
        # Re-assert against the LANDED url: an allowlisted target that
        # redirects off-allowlist must be treated as a refusal, not exposed.
        landed = str(page.url)
        if not self._host_allowed(landed):
            host = (urlparse(landed).hostname or landed).lower()
            return ActionResult(ok=False, error=f"refused: redirected off-allowlist to {host}")
        await self._record_visit(page, landed)
        return ActionResult(ok=True, output=f"navigated to {landed}")

    async def _record_visit(self, page: Any, url: str) -> None:
        try:
            title = str(await page.title())
        except Exception:  # noqa: BLE001 — title is best-effort metadata
            title = ""
        self._visits.append({"url": url, "title": title, "at": time.time()})
        if len(self._visits) > self._max_visits:
            del self._visits[: len(self._visits) - self._max_visits]

    def _recent_visits(self) -> list[dict[str, object]]:
        """Visits oldest→newest (index 1 = oldest of the recent window)."""
        return self._visits

    def _tracked_tabs(self) -> list[Any]:
        return self._tabs if self._tabs else ([self._page] if self._page is not None else [])

    async def _download(self, args: dict[str, object]) -> ActionResult:
        """Save a file from an allowlisted URL into the fixed downloads dir.

        A direct API GET on the browser context (not a click), so the page's
        cookies make authenticated downloads work and the context itself stays
        ``accept_downloads=False`` — a download can only ever start through
        this tool, never as a side effect of clicking. Executable payloads are
        refused by URL extension AND Content-Type; size is capped."""
        url = str(args.get("url", ""))
        if not self._host_allowed(url):
            return ActionResult(ok=False, error=f"refused: domain not allowlisted: {url}")
        name = _safe_download_name(url)
        ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
        if ext in _DOWNLOAD_DENY_EXTS:
            return ActionResult(
                ok=False, error=f"refused: executable downloads are forbidden: {name}"
            )
        if self._ctx is None:
            return ActionResult(ok=False, error="no browser context for downloads")
        resp = await self._ctx.request.get(url)
        status = int(getattr(resp, "status", 0) or 0)
        if status >= 400:
            return ActionResult(ok=False, error=f"download failed: HTTP {status}")
        headers = {
            str(k).lower(): str(v) for k, v in dict(getattr(resp, "headers", {}) or {}).items()
        }
        ctype = headers.get("content-type", "")
        if any(hint in ctype.lower() for hint in _DOWNLOAD_DENY_CT_HINTS):
            return ActionResult(ok=False, error=f"refused: executable content-type: {ctype!r}")
        body = await resp.body()
        if len(body) > _DOWNLOAD_MAX_BYTES:
            return ActionResult(ok=False, error=f"refused: too large ({len(body)} bytes)")

        def _write(blob: bytes) -> str:
            out_dir = Path(_DOWNLOAD_DIR)
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / name
            out_path.write_bytes(blob)
            return str(out_path)

        return ActionResult(ok=True, output=f"saved to {await asyncio.to_thread(_write, body)}")

    async def _dispatch(self, page: Any, tool: str, args: dict[str, object]) -> ActionResult:
        if tool == "browser_navigate":
            return await self._goto(page, str(args.get("url", "")))

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

        if tool == "browser_press_key":
            if (refusal := self._current_page_refusal(page)) is not None:
                return refusal
            key = str(args.get("key", ""))
            if key not in _PRESS_KEYS:
                return ActionResult(
                    ok=False, error=f"refused: key not in bounded vocabulary: {key!r}"
                )
            await page.keyboard.press(key)
            return ActionResult(ok=True, output=f"pressed {key}")

        if tool == "browser_scroll":
            if (refusal := self._current_page_refusal(page)) is not None:
                return refusal
            to = args.get("to")
            dy = args.get("dy")
            target = args.get("target")
            supplied = sum(v is not None for v in (to, dy, target))
            if supplied != 1:
                return ActionResult(
                    ok=False, error="refused: exactly one of 'to', 'dy', 'target' is required"
                )
            if to is not None:
                where = str(to)
                if where not in ("top", "bottom"):
                    return ActionResult(
                        ok=False, error=f"refused: 'to' must be top|bottom: {where!r}"
                    )
                y = await page.evaluate(_MEDIA_SCRIPTS["scroll_to"], [where])
                return ActionResult(ok=True, output=f"scrolled to {where} (scrollY={y})")
            if dy is not None:
                if not isinstance(dy, int) or isinstance(dy, bool):
                    return ActionResult(ok=False, error=f"refused: 'dy' must be an integer: {dy!r}")
                pixels = max(-_SCROLL_MAX_PX, min(_SCROLL_MAX_PX, dy))
                await page.mouse.wheel(0, pixels)
                return ActionResult(ok=True, output=f"scrolled {pixels:+d}px")
            # target is not None — scroll the first text match into view.
            locator = page.get_by_text(str(target))
            await locator.first.scroll_into_view_if_needed()
            return ActionResult(ok=True, output=f"scrolled to {str(target)!r}")

        if tool == "browser_media":
            if (refusal := self._current_page_refusal(page)) is not None:
                return refusal
            return await self._media(page, args)

        if tool == "browser_tab":
            return await self._tab(page, args)

        if tool == "browser_history":
            action = str(args.get("action", ""))
            if action in ("back", "forward"):
                if (refusal := self._current_page_refusal(page)) is not None:
                    return refusal
                await (page.go_back() if action == "back" else page.go_forward())
                return ActionResult(ok=True, output=f"went {action}")
            if action == "list":
                visits = self._recent_visits()
                if not visits:
                    return ActionResult(ok=True, output="no visited pages yet")
                lines = [
                    f"{i}. {v['title'] or v['url']} — {v['url']}"
                    for i, v in enumerate(reversed(visits[-10:]), start=1)
                ]
                return ActionResult(ok=True, output="\n".join(lines))
            if action == "goto":
                idx = args.get("index")
                visits = self._recent_visits()
                if (
                    not isinstance(idx, int)
                    or isinstance(idx, bool)
                    or not (1 <= idx <= len(visits))
                ):
                    return ActionResult(ok=False, error=f"refused: bad visit index: {idx!r}")
                url = str(visits[idx - 1]["url"])  # oldest→newest, 1-based
                return await self._goto(page, url)
            return ActionResult(ok=False, error=f"refused: unknown history action {action!r}")

        if tool == "browser_search":
            from dana_agent.providers.actions.adapters import ADAPTERS, search_url  # noqa: PLC0415

            engine = str(args.get("engine", ""))
            query = str(args.get("query", ""))
            search_target = search_url(engine, query)
            if search_target is None:
                known = ", ".join(sorted(ADAPTERS))
                return ActionResult(
                    ok=False, error=f"refused: unknown search engine {engine!r} (known: {known})"
                )
            result = await self._goto(page, search_target)
            if not result.ok:
                return result
            if args.get("open_first"):
                selector = ADAPTERS[engine].first_result_selector
                if selector is None:
                    return ActionResult(
                        ok=True, output=result.output + "; no first-result selector"
                    )
                await page.locator(selector).first.click()  # auto-waits; no sleeps
                await self._record_visit(page, str(page.url))
                return ActionResult(ok=True, output=f"searched {engine}, opened first result")
            return result

        if tool == "browser_get_state":
            if (refusal := self._current_page_refusal(page)) is not None:
                return refusal
            tabs = self._tracked_tabs()
            titles = []
            for tab in tabs:
                try:
                    titles.append(str(await tab.title()))
                except Exception:  # noqa: BLE001 — best-effort metadata
                    titles.append("")
            active = titles[0] if titles else ""
            if tabs and self._page is not None and tabs.index(self._page) < len(titles):
                active = titles[tabs.index(self._page)]
            media_state = await self._eval_media(page, "state", None)
            media_str = _speakable_media(media_state) if isinstance(media_state, dict) else "none"
            return ActionResult(
                ok=True,
                output=f"page: {active!r} ({page.url}) | tabs: {len(tabs)} | media: {media_str}",
            )

        if tool == "browser_download":
            return await self._download(args)

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

    # --- tab manager (M2) ----------------------------------------------------

    async def _tab(self, page: Any, args: dict[str, object]) -> ActionResult:
        action = str(args.get("action", ""))

        if action == "new":
            if self._ctx is None:
                return ActionResult(ok=False, error="no browser context for a new tab")
            new_page = await self._ctx.new_page()
            self._tabs.append(new_page)
            self._page = new_page  # a voice-opened tab becomes the active one
            await new_page.bring_to_front()
            url = args.get("url")
            if url is not None:
                result = await self._goto(new_page, str(url))
                if not result.ok:
                    return result
            return ActionResult(ok=True, output=f"new tab ({len(self._tabs)} open)")

        if action == "list":
            tabs = self._tracked_tabs()
            if not tabs:
                return ActionResult(ok=True, output="no tabs")
            lines = []
            for i, tab in enumerate(tabs, start=1):
                title = ""
                try:
                    title = str(await tab.title())
                except Exception:  # noqa: BLE001 — best-effort metadata
                    log.debug("tab_title_unavailable")
                marker = "*" if tab is self._page else " "
                lines.append(f"{marker}{i}. {title or tab.url}")
            return ActionResult(ok=True, output="\n".join(lines))

        if action == "switch":
            target = await self._match_tab(args.get("title"), args.get("index"))
            if target is None:
                return ActionResult(ok=False, error="refused: no tab matches")
            self._page = target
            await target.bring_to_front()
            return ActionResult(ok=True, output=f"switched to tab ({target.url})")

        if action == "close":
            target = await self._match_tab(args.get("title"), args.get("index")) or page
            await target.close()
            if target in self._tabs:
                self._tabs.remove(target)
            if target is self._page:
                # Fall back to another tab, or a fresh page on the live context.
                if self._tabs:
                    self._page = self._tabs[0]
                    await self._page.bring_to_front()
                elif self._ctx is not None:
                    self._page = await self._ctx.new_page()
                    self._tabs = [self._page]
                else:
                    self._page = None
            return ActionResult(ok=True, output=f"closed tab ({len(self._tracked_tabs())} open)")

        return ActionResult(ok=False, error=f"refused: unknown tab action {action!r}")

    async def _match_tab(self, title: object, index: object) -> Any | None:
        """Find a tab by 1-based index, URL substring, or (awaited) page title
        substring; None if no match. The active page is a candidate too (close/
        switch current tab)."""
        tabs = self._tracked_tabs()
        if isinstance(index, int) and not isinstance(index, bool) and 1 <= index <= len(tabs):
            return tabs[index - 1]
        if isinstance(title, str) and title:
            needle = title.lower()
            for tab in tabs:  # URL match first — cheap, no round trip
                if needle in str(getattr(tab, "url", "")).lower():
                    return tab
            for tab in tabs:  # then the rendered page title
                try:
                    if needle in str(await tab.title()).lower():
                        return tab
                except Exception:  # noqa: BLE001 — best-effort metadata
                    log.debug("tab_title_unavailable")
                    continue
        return None

    # --- media engine (canned JS only; the model never authors scripts) ---

    async def _media(self, page: Any, args: dict[str, object]) -> ActionResult:
        action = str(args.get("action", ""))
        if action not in _MEDIA_ACTIONS:
            return ActionResult(ok=False, error=f"refused: unknown media action {action!r}")

        if action == "seek":
            delta = self._num(args, "delta_s")
            if delta is None:
                return ActionResult(ok=False, error="refused: seek requires a numeric 'delta_s'")
            result = await self._eval_media(page, "seek", delta)
            if isinstance(result, (int, float)):
                return ActionResult(ok=True, output=f"position {_fmt_seconds(float(result))}")
            return self._no_media_result(result)

        if action == "seek_to":
            pos = self._num(args, "position_s")
            if pos is None:
                return ActionResult(
                    ok=False, error="refused: seek_to requires a numeric 'position_s'"
                )
            if pos < 0:
                return ActionResult(ok=False, error=f"refused: position_s must be >= 0: {pos}")
            result = await self._eval_media(page, "seek_to", pos)
            if isinstance(result, (int, float)):
                return ActionResult(ok=True, output=f"position {_fmt_seconds(float(result))}")
            return self._no_media_result(result)

        if action == "rate":
            rate = self._num(args, "rate")
            if rate is None or not (_RATE_MIN <= rate <= _RATE_MAX):
                return ActionResult(
                    ok=False,
                    error=f"refused: rate must be within {_RATE_MIN}-{_RATE_MAX}: {rate!r}",
                )
            result = await self._eval_media(page, "rate", rate)
            return ActionResult(ok=True, output=f"playback rate {result}x")

        if action in ("mute", "unmute"):
            result = await self._eval_media(page, "mute", action == "mute")
            return ActionResult(ok=True, output="muted" if result else "unmuted")

        if action == "state":
            result = await self._eval_media(page, "state", None)
            if isinstance(result, dict):
                return ActionResult(ok=True, output=_speakable_media(result))
            return self._no_media_result(result)

        if action in ("next", "prev"):
            # No generic HTML5 API for track-skipping — adapter territory:
            # click the site's player control if one is mapped for the CURRENT
            # page; otherwise point at the system media session.
            from dana_agent.providers.actions.adapters import adapter_for_url  # noqa: PLC0415

            adapter = adapter_for_url(str(getattr(page, "url", "") or ""))
            selector = getattr(adapter, f"{action}_selector", None) if adapter else None
            if selector is None:
                return ActionResult(
                    ok=False,
                    error=f"no in-page {action} control for this site — use os_media_key",
                )
            await page.locator(selector).first.click()
            return ActionResult(ok=True, output=action)

        # play / pause / toggle — with the player-key fallback when the page
        # has no <video>/<audio> element the script can reach.
        result = await self._eval_media(page, action, None)
        if result == "no-media":
            await page.keyboard.press(_PLAY_PAUSE_KEY)
            return ActionResult(
                ok=True, output=f"no media element; pressed {_PLAY_PAUSE_KEY!r} (player key)"
            )
        return ActionResult(ok=True, output=action)

    async def _eval_media(self, page: Any, script_name: str, arg: object) -> Any:
        if arg is None:
            return await page.evaluate(_MEDIA_SCRIPTS[script_name])
        return await page.evaluate(_MEDIA_SCRIPTS[script_name], [arg])

    @staticmethod
    def _num(args: dict[str, object], name: str) -> float | None:
        """Pull a validated, range-clamped numeric arg; None ⇒ missing/invalid."""
        raw = args.get(name)
        if not isinstance(raw, (int, float)) or isinstance(raw, bool):
            return None  # missing, prose, or bool — refuse upstream
        return max(-_SEEK_DELTA_LIMIT_S, min(_SEEK_DELTA_LIMIT_S, float(raw)))

    @staticmethod
    def _no_media_result(result: object) -> ActionResult:
        if result == "no-media":
            return ActionResult(ok=False, error="no media element on this page")
        return ActionResult(ok=False, error=f"media error: {result!r}")

"""A fake Playwright page so browser-action tests run fully offline.

``BrowserActions`` lazily creates its real page in ``_ensure_page`` and returns
early when ``self._page`` is already set — so a test injects a ``FakePage`` and
NO real browser (and no ``playwright`` import) is ever touched.

``evaluate`` simulates the provider's canned media scripts (keyed by their
``/* dana:<name> */`` markers) against a tiny in-memory media element, so the
media engine's contract — marker + numeric arg → state change — is tested
without Chromium.
"""

from __future__ import annotations

from typing import Any


class FakeKeyboard:
    def __init__(self, page: FakePage) -> None:
        self._page = page

    async def press(self, key: str) -> None:
        self._page.key_presses.append(key)


class FakeMouse:
    def __init__(self, page: FakePage) -> None:
        self._page = page

    async def wheel(self, dx: int, dy: int) -> None:
        self._page.wheel_calls.append((dx, dy))


class FakeLocator:
    """A stand-in for a Playwright Locator supporting ``.first.click/fill``."""

    def __init__(self, page: FakePage, meta: dict[str, object]) -> None:
        self._page = page
        self._meta = meta

    @property
    def first(self) -> FakeLocator:
        return self

    async def click(self) -> None:
        self._page.clicks.append(self._meta)

    async def fill(self, text: str) -> None:
        self._page.fills.append((self._meta, text))

    async def scroll_into_view_if_needed(self) -> None:
        self._page.scroll_targets.append(self._meta)


class FakeResponse:
    """Stand-in for a Playwright APIResponse."""

    def __init__(
        self,
        status: int = 200,
        headers: dict[str, str] | None = None,
        body: bytes = b"file-bytes",
    ) -> None:
        self.status = status
        self.headers = headers or {}
        self._body = body

    async def body(self) -> bytes:
        return self._body


class FakeAPIRequest:
    """Stand-in for context.request: records GETs, returns a canned response."""

    def __init__(self, response: FakeResponse) -> None:
        self.response = response
        self.gets: list[str] = []

    async def get(self, url: str, **_kw: object) -> FakeResponse:
        self.gets.append(url)
        return self.response


class FakeContext:
    """Stand-in for a Playwright BrowserContext: hands out new FakePages."""

    def __init__(self) -> None:
        self.created = 0
        self.request = FakeAPIRequest(FakeResponse())

    async def new_page(self) -> FakePage:
        self.created += 1
        return FakePage(url="about:blank")


class FakePage:
    """Records interactions; returns canned page text. Async where Playwright is."""

    def __init__(
        self,
        *,
        body_text: str = "hello world",
        url: str = "about:blank",
        redirect_to: str | None = None,
        doc_title: str = "Example Page",
    ) -> None:
        self.url = url
        self._body_text = body_text
        # When set, goto() lands here instead of its argument (simulates a 30x /
        # meta-refresh / JS redirect to a possibly off-allowlist host).
        self._redirect_to = redirect_to
        self.doc_title = doc_title
        self.goto_urls: list[str] = []
        self.clicks: list[dict[str, object]] = []
        self.fills: list[tuple[dict[str, object], str]] = []
        self.screenshots = 0
        self.key_presses: list[str] = []
        self.wheel_calls: list[tuple[int, int]] = []
        self.scroll_targets: list[dict[str, object]] = []
        # Simulated <video>/<audio> element state; None ⇒ page has no media
        # element (the canned scripts return 'no-media').
        self.media: dict[str, Any] | None = {
            "paused": True,
            "currentTime": 0.0,
            "duration": 640.0,  # 10:40 — long enough for any seek test
            "rate": 1.0,
            "muted": False,
        }
        self.media_title: str | None = "lofi hip hop radio"
        self.scroll_y = 0.0
        # Tab/history interaction recording.
        self.history_calls: list[str] = []
        self.brought_to_front = 0
        self.closed = False

    async def title(self) -> str:
        return self.doc_title

    async def go_back(self) -> None:
        self.history_calls.append("back")

    async def go_forward(self) -> None:
        self.history_calls.append("forward")

    async def bring_to_front(self) -> None:
        self.brought_to_front += 1

    async def close(self) -> None:
        self.closed = True

    def locator(self, css: str) -> FakeLocator:
        return FakeLocator(self, {"css": css})

    @property
    def keyboard(self) -> FakeKeyboard:
        return FakeKeyboard(self)

    @property
    def mouse(self) -> FakeMouse:
        return FakeMouse(self)

    async def goto(self, url: str, wait_until: str | None = None) -> None:
        self.goto_urls.append(url)
        self.url = self._redirect_to if self._redirect_to is not None else url

    async def inner_text(self, selector: str) -> str:
        return self._body_text

    def get_by_role(self, role: str, name: str = "") -> FakeLocator:
        return FakeLocator(self, {"role": role, "name": name})

    def get_by_text(self, text: str, exact: bool = False) -> FakeLocator:
        return FakeLocator(self, {"text": text})

    def get_by_label(self, label: str) -> FakeLocator:
        return FakeLocator(self, {"label": label})

    async def screenshot(self) -> bytes:
        self.screenshots += 1
        return b"\x89PNG\r\n" * 8

    # --- canned-script simulation ------------------------------------------

    async def evaluate(self, script: str, arg: Any = None) -> Any:
        """Dispatch a ``/* dana:<name> */`` canned script against ``self.media``.

        Mirrors the semantics of the JS in ``browser._MEDIA_SCRIPTS`` closely
        enough to pin the provider↔page contract offline."""
        name = script.split("*/", 1)[0].replace("/*", "").strip().removeprefix("dana:")
        m = self.media
        if name == "scroll_to":
            self.scroll_y = 0.0 if arg[0] == "top" else 8_000.0
            return self.scroll_y
        if m is None:
            return "no-media"
        if name == "state":
            cur, dur = m["currentTime"], m["duration"]

            def fmt(s: float) -> str:
                return f"{int(s // 60)}:{int(s % 60):02d}"

            return {
                "title": self.media_title or self.doc_title,
                "playing": m["paused"] is False,
                "position": cur,
                "duration": dur,
                "rate": m["rate"],
                "muted": m["muted"],
                "at": f"{fmt(cur)} / {fmt(dur)}",
            }
        if name == "play":
            m["paused"] = False
            return "ok"
        if name == "pause":
            m["paused"] = True
            return "ok"
        if name == "toggle":
            m["paused"] = not m["paused"]
            return "ok"
        if name == "seek":
            m["currentTime"] = max(0.0, min(m["duration"] - 0.05, m["currentTime"] + arg[0]))
            return m["currentTime"]
        if name == "seek_to":
            m["currentTime"] = max(0.0, min(m["duration"] - 0.05, arg[0]))
            return m["currentTime"]
        if name == "rate":
            m["rate"] = arg[0]
            return m["rate"]
        if name == "mute":
            m["muted"] = bool(arg[0])
            return m["muted"]
        raise AssertionError(f"FakePage.evaluate: unknown canned script {name!r}")


def browser_with_fake_page(provider: Any, page: FakePage | None = None) -> FakePage:
    """Inject a FakePage into a BrowserActions instance and return it."""
    fake = page or FakePage()
    provider._page = fake
    return fake

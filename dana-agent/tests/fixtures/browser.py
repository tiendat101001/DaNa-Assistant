"""A fake Playwright page so browser-action tests run fully offline.

``BrowserActions`` lazily creates its real page in ``_ensure_page`` and returns
early when ``self._page`` is already set — so a test injects a ``FakePage`` and
NO real browser (and no ``playwright`` import) is ever touched.
"""

from __future__ import annotations

from typing import Any


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


class FakePage:
    """Records interactions; returns canned page text. Async where Playwright is."""

    def __init__(
        self,
        *,
        body_text: str = "hello world",
        url: str = "about:blank",
        redirect_to: str | None = None,
    ) -> None:
        self.url = url
        self._body_text = body_text
        # When set, goto() lands here instead of its argument (simulates a 30x /
        # meta-refresh / JS redirect to a possibly off-allowlist host).
        self._redirect_to = redirect_to
        self.goto_urls: list[str] = []
        self.clicks: list[dict[str, object]] = []
        self.fills: list[tuple[dict[str, object], str]] = []
        self.screenshots = 0

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


def browser_with_fake_page(provider: Any, page: FakePage | None = None) -> FakePage:
    """Inject a FakePage into a BrowserActions instance and return it."""
    fake = page or FakePage()
    provider._page = fake
    return fake

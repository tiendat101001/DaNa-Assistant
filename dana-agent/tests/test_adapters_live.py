"""TASK-011 M6 live probes — adapter facts verified against the real world.

Opt-in ONLY (``pytest -m live``): these launch real Chromium via the production
``BrowserActions`` code path. Two groups:

1. Network probes (real sites) — the anti-rot check for the site adapters:
   every engine's ``first_result_selector`` must still match ≥1 element on its
   live results page, and YouTube's ``open_first`` must reach a /watch page
   with a ``<video>`` element and a readable media state.
2. Local probe (NO network — real Chromium, 127.0.0.1 fixture page) — the
   canned media scripts (seek/state/rate) proven against a real HTML5 media
   element, end to end through the provider.

Requires: ``uv sync --extra actions && playwright install chromium`` plus a
network connection for group 1. Headless unless DANA_LIVE_HEADED=1.
"""

from __future__ import annotations

import wave
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.live

pytest.importorskip("playwright.async_api", reason="playwright not installed (extra 'actions')")

from dana_agent.config import ActionsConfig  # noqa: E402
from dana_agent.providers.actions.adapters import ADAPTERS  # noqa: E402
from dana_agent.providers.actions.browser import BrowserActions  # noqa: E402
from dana_agent.types import Action  # noqa: E402

_ENGINE_HOSTS = [h for a in ADAPTERS.values() for h in a.host_suffixes]


def _browser(tmp_path: Path) -> BrowserActions:
    import os

    cfg = ActionsConfig(
        provider="browser",
        allowed_domains=_ENGINE_HOSTS + ["127.0.0.1"],
        browser_profile_dir=str(tmp_path / "profile"),
        browser_headless=os.environ.get("DANA_LIVE_HEADED") != "1",
    )
    return BrowserActions(cfg)


# --- group 1: network probes (anti-rot for adapter selectors) ------------------


@pytest.mark.parametrize("engine", sorted(ADAPTERS))
async def test_live_search_first_result_selector_matches(engine: str, tmp_path: Path) -> None:
    provider = _browser(tmp_path)
    try:
        result = await provider.execute(
            Action("browser_search", {"query": "open source voice assistant", "engine": engine})
        )
        assert result.ok is True, result.error
        page = provider._page
        selector = ADAPTERS[engine].first_result_selector
        assert selector, engine
        count = await page.locator(selector).count()
        assert count >= 1, (
            f"selector drift: {engine} first_result_selector {selector!r} matched {count} "
            f"elements on {page.url} — update the adapter and re-stamp 'verified'"
        )
    finally:
        await provider.aclose()


async def test_live_youtube_open_first_reaches_watch_page(tmp_path: Path) -> None:
    provider = _browser(tmp_path)
    try:
        result = await provider.execute(
            Action(
                "browser_search",
                {"query": "lofi hip hop radio", "engine": "youtube", "open_first": True},
            )
        )
        assert result.ok is True, result.error
        page = provider._page
        assert "/watch" in str(page.url), f"landed on {page.url}"
        assert await page.locator("video").count() >= 1
        state = await provider.execute(Action("browser_media", {"action": "state"}))
        assert state.ok is True, state.error
        assert "paused" in state.output or "playing" in state.output
    finally:
        await provider.aclose()


# --- group 2: local probe (real Chromium, no network) ---------------------------


def _write_fixture_site(site_dir: Path) -> None:
    """A 3-second 8 kHz tone + a minimal player page shaped like a media site."""
    wav = site_dir / "tone.wav"
    with wave.open(str(wav), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(8000)
        import math
        import struct

        frames = bytearray()
        for i in range(3 * 8000):
            frames += struct.pack("<h", int(12000 * math.sin(2 * math.pi * 440 * i / 8000)))
        handle.writeframes(bytes(frames))
    (site_dir / "media.html").write_text(
        "<html><body><h1>Local Fixture Player</h1>"
        '<audio controls src="tone.wav"></audio>'
        "</body></html>",
        encoding="utf-8",
    )


class _SiteServer:
    """Threaded http.server on an ephemeral 127.0.0.1 port."""

    def __init__(self, root: Path) -> None:
        import functools
        import http.server

        handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(root))
        self._httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.port = self._httpd.server_address[1]

    def __enter__(self) -> _SiteServer:
        import threading

        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self._httpd.shutdown()


async def test_live_media_scripts_against_real_chromium_local(tmp_path: Path) -> None:
    site = tmp_path / "site"
    site.mkdir()
    _write_fixture_site(site)
    provider = _browser(tmp_path)
    with _SiteServer(site) as server:
        try:
            url = f"http://127.0.0.1:{server.port}/media.html"
            nav = await provider.execute(Action("browser_navigate", {"url": url}))
            assert nav.ok is True, nav.error

            state0: Any = await provider.execute(Action("browser_media", {"action": "state"}))
            assert state0.ok is True, state0.error
            assert "0:00" in state0.output and "0:03" in state0.output  # duration metadata

            seek = await provider.execute(
                Action("browser_media", {"action": "seek_to", "position_s": 2.0})
            )
            assert seek.ok is True, seek.error
            assert seek.output == "position 0:02"

            rate = await provider.execute(Action("browser_media", {"action": "rate", "rate": 1.5}))
            assert rate.ok is True and "1.5x" in rate.output
        finally:
            await provider.aclose()

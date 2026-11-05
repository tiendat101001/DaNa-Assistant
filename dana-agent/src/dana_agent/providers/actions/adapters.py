"""Site adapters — per-site *data* (URL templates, selectors), not code ladders.

An adapter is a frozen record of facts about a site: how to build a search URL
and which CSS selector picks the first organic result. The browser provider
uses them for ``browser_search`` (and its ``open_first`` shortcut); everything
else stays on the generic ARIA/HTML5 path.

Facts about the world rot. Every adapter carries a ``verified`` stamp — the
date its selectors were last confirmed against real Chromium. The offline suite
pins URL building; selector drift is caught by the opt-in live probes
(``-m live``, TASK-011 M6), never by a user's voice command failing.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from urllib.parse import quote_plus


@dataclass(frozen=True, slots=True)
class SiteAdapter:
    name: str
    host_suffixes: tuple[str, ...]
    search_template: str  # "{q}" is the URL-encoded query slot
    first_result_selector: str | None = None
    verified: str = ""
    # Player next/prev controls for browser_media(next|prev); None ⇒ the site
    # has no in-page control (playlists aside) — route via os_media_key.
    next_selector: str | None = None
    prev_selector: str | None = None


ADAPTERS: dict[str, SiteAdapter] = {
    "youtube": SiteAdapter(
        name="youtube",
        host_suffixes=("youtube.com", "youtube-nocookie.com"),
        search_template="https://www.youtube.com/results?search_query={q}",
        # First organic video tile on the results page.
        first_result_selector="ytd-video-renderer ytd-thumbnail a",
        # Player next-track button (present on watch pages; disabled at the end
        # of a plain video — the click auto-wait handles that as an error).
        next_selector=".ytp-next-button",
        verified="2026-08-22 live probe ok (selector, open_first->/watch, media state)",
    ),
    "google": SiteAdapter(
        name="google",
        host_suffixes=("google.com",),
        search_template="https://www.google.com/search?q={q}",
        first_result_selector="a h3",
        verified="2026-08-22",
    ),
    "duckduckgo": SiteAdapter(
        name="duckduckgo",
        host_suffixes=("duckduckgo.com",),
        search_template="https://duckduckgo.com/?q={q}",
        first_result_selector="a[data-testid='result-title-a']",
        verified="2026-08-22",
    ),
    "wikipedia": SiteAdapter(
        name="wikipedia",
        host_suffixes=("wikipedia.org",),
        search_template="https://en.wikipedia.org/w/index.php?search={q}",
        # Note: an exact-title match lands directly on the article page.
        first_result_selector=".mw-search-result a",
        verified="2026-08-22 live probe ok",
    ),
    "arxiv": SiteAdapter(
        name="arxiv",
        host_suffixes=("arxiv.org",),
        search_template="https://arxiv.org/search/?query={q}&searchtype=all",
        # Search results page: one li per paper, PDF link inside the
        # result item → opens the PDF directly by default.
        first_result_selector="li.arxiv-result a[href*='/pdf/']",
        verified="2026-08-22 live probe ok",
    ),
}


_DOMAIN_RE = re.compile(r"^(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,24}$")


def merged_templates(extra: Mapping[str, str] | None = None) -> dict[str, str]:
    """Built-in engine templates overlaid with the user's ``cfg.search_engines``.

    User entries must be http(s) templates containing ``{q}``; anything else is
    dropped (default-deny: a malformed template never becomes a navigation)."""
    out = {name: a.search_template for name, a in ADAPTERS.items()}
    for name, tpl in (extra or {}).items():
        key = name.strip().lower()
        if key and "{q}" in tpl and tpl.lower().startswith(("http://", "https://")):
            out[key] = tpl
    return out


def known_engines(extra: Mapping[str, str] | None = None) -> list[str]:
    return sorted(merged_templates(extra))


def search_url(engine: str, query: str, extra: Mapping[str, str] | None = None) -> str | None:
    """Build a search URL, or None for an unknown engine (default-deny).

    ``extra`` is the user's configured ``{name: template}`` map; the query is
    URL-encoded into the ``{q}`` slot."""
    template = merged_templates(extra).get(engine.strip().lower())
    if template is None:
        return None
    return template.replace("{q}", quote_plus(query))


def bare_domain(engine: str) -> str | None:
    """``"shopee.vn"`` / ``"https://www.shopee.vn/x"`` -> ``"www.shopee.vn"``, or
    None when ``engine`` is not a domain-looking string."""
    text = engine.strip().lower()
    for prefix in ("https://", "http://"):
        text = text.removeprefix(prefix)
    host = text.split("/", 1)[0].split("?", 1)[0]
    return host if _DOMAIN_RE.match(host) else None


def adapter_for_url(url: str) -> SiteAdapter | None:
    """The adapter whose host suffixes cover ``url``, if any."""
    from urllib.parse import urlparse  # noqa: PLC0415 — keep module import light

    host = (urlparse(url).hostname or "").lower()
    if not host:
        return None
    for adapter in ADAPTERS.values():
        if any(host == s or host.endswith("." + s) for s in adapter.host_suffixes):
            return adapter
    return None

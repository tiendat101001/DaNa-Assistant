"""Browser action provider: the allowlist (default-deny), scheme rejection, the
classify() table, and bounded execution against a fake page — all offline, no
real browser ever launches (safe-computer-use skill)."""

from __future__ import annotations

import pytest

from dana_agent.config import ActionsConfig
from dana_agent.providers.actions.browser import BrowserActions
from dana_agent.types import Action, Reversibility
from tests.fixtures.browser import FakePage, browser_with_fake_page


def make_provider(domains: list[str] | None = None) -> BrowserActions:
    return BrowserActions(ActionsConfig(provider="browser", allowed_domains=domains or []))


# --- allowlist: default-deny + host-suffix match -------------------------


def test_empty_allowlist_denies_everything() -> None:
    provider = make_provider([])
    assert provider._host_allowed("https://example.com") is False


def test_host_suffix_match_allows_subdomains() -> None:
    provider = make_provider(["example.com"])
    assert provider._host_allowed("https://example.com/x") is True
    assert provider._host_allowed("https://mail.example.com/x") is True


def test_lookalike_and_foreign_hosts_denied() -> None:
    provider = make_provider(["example.com"])
    # substring/lookalike tricks must not slip past a parsed-host check
    assert provider._host_allowed("https://example.com.evil.com") is False
    assert provider._host_allowed("https://notexample.com") is False
    assert provider._host_allowed("https://evil.com/example.com") is False


def test_userinfo_at_trick_is_denied() -> None:
    # The classic bypass: an allowlisted token in the userinfo before '@'. The
    # real host is what follows '@' — urlparse().hostname must be used, not a
    # substring of the raw URL.
    provider = make_provider(["example.com"])
    assert provider._host_allowed("http://example.com@evil.com") is False
    assert provider._host_allowed("https://user:pass@evil.com") is False
    # a real subdomain behind userinfo is still correctly allowed
    assert provider._host_allowed("https://user:pass@mail.example.com") is True


def test_trailing_dot_fqdn_is_denied() -> None:
    # A trailing-dot FQDN ("example.com.") is a distinct host string; default-deny
    # rejects it rather than treating it as equivalent (the safe direction).
    provider = make_provider(["example.com"])
    assert provider._host_allowed("https://example.com.") is False


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "javascript:alert(1)",
        "data:text/html,<h1>x</h1>",
        "ftp://example.com/x",
    ],
)
def test_non_http_schemes_rejected_even_if_host_allowlisted(url: str) -> None:
    provider = make_provider(["example.com"])
    assert provider._host_allowed(url) is False


# --- classify() table ----------------------------------------------------


def test_classify_navigate_allowlisted_is_read_only() -> None:
    provider = make_provider(["example.com"])
    action = Action("browser_navigate", {"url": "https://example.com"})
    assert provider.classify(action) is Reversibility.READ_ONLY


def test_classify_navigate_off_allowlist_is_out_of_scope() -> None:
    provider = make_provider(["example.com"])
    action = Action("browser_navigate", {"url": "https://evil.com"})
    assert provider.classify(action) is Reversibility.OUT_OF_SCOPE


def test_classify_reads_are_read_only() -> None:
    provider = make_provider()
    assert provider.classify(Action("browser_read_text", {})) is Reversibility.READ_ONLY
    assert provider.classify(Action("browser_screenshot", {})) is Reversibility.READ_ONLY


def test_classify_type_is_reversible() -> None:
    provider = make_provider()
    assert provider.classify(Action("browser_type", {"text": "hi"})) is Reversibility.REVERSIBLE


def test_classify_generic_click_is_reversible() -> None:
    provider = make_provider()
    action = Action("browser_click", {"text": "Read more"})
    assert provider.classify(action) is Reversibility.REVERSIBLE


@pytest.mark.parametrize("label", ["Submit", "Pay now", "Delete account", "Send", "Place order"])
def test_classify_committing_click_is_destructive(label: str) -> None:
    provider = make_provider()
    action = Action("browser_click", {"text": label})
    assert provider.classify(action) is Reversibility.DESTRUCTIVE


def test_classify_unknown_tool_is_out_of_scope() -> None:
    provider = make_provider(["example.com"])
    assert provider.classify(Action("browser_eval", {})) is Reversibility.OUT_OF_SCOPE


# --- execute() against a fake page ---------------------------------------


async def test_execute_unknown_tool_is_structured_error() -> None:
    provider = make_provider(["example.com"])
    browser_with_fake_page(provider)
    result = await provider.execute(Action("browser_eval", {"code": "x"}))
    assert result.ok is False
    assert result.error is not None and "unknown browser tool" in result.error


async def test_execute_navigate_off_allowlist_refuses_without_io() -> None:
    provider = make_provider(["example.com"])
    page = browser_with_fake_page(provider)
    result = await provider.execute(Action("browser_navigate", {"url": "https://evil.com"}))
    assert result.ok is False
    assert result.error is not None and "not allowlisted" in result.error
    assert page.goto_urls == []  # never navigated


async def test_execute_navigate_allowlisted_goes() -> None:
    url = "https://example.com/page"
    provider = make_provider(["example.com"])
    page = browser_with_fake_page(provider)
    result = await provider.execute(Action("browser_navigate", {"url": url}))
    assert result.ok is True
    assert page.goto_urls == [url]


async def test_execute_read_text_truncates_and_returns_data() -> None:
    provider = make_provider(["example.com"])
    browser_with_fake_page(provider, FakePage(body_text="Z" * 20_000, url="https://example.com"))
    result = await provider.execute(Action("browser_read_text", {}))
    assert result.ok is True
    assert len(result.output) == 8_000  # bounded read (injection-surface control)


async def test_execute_click_requires_a_selector() -> None:
    provider = make_provider(["example.com"])
    browser_with_fake_page(provider, FakePage(url="https://example.com"))
    result = await provider.execute(Action("browser_click", {}))
    assert result.ok is False
    assert result.error is not None and "role" in result.error


async def test_execute_type_fills_via_label() -> None:
    provider = make_provider(["example.com"])
    page = browser_with_fake_page(provider, FakePage(url="https://example.com"))
    result = await provider.execute(Action("browser_type", {"into": "Search", "text": "kittens"}))
    assert result.ok is True
    assert page.fills == [({"label": "Search"}, "kittens")]


async def test_execute_screenshot_saves_file() -> None:
    provider = make_provider(["example.com"])
    page = browser_with_fake_page(provider, FakePage(url="https://example.com"))
    result = await provider.execute(Action("browser_screenshot", {}))
    assert result.ok is True
    assert result.output.endswith(".png")
    assert page.screenshots == 1


# --- redirect / current-page allowlist re-assertion ----------------------


async def test_navigate_refuses_on_off_allowlist_redirect() -> None:
    # goto an allowlisted target that redirects to an off-allowlist host.
    provider = make_provider(["example.com"])
    page = browser_with_fake_page(
        provider, FakePage(url="about:blank", redirect_to="https://evil.com/landing")
    )
    result = await provider.execute(
        Action("browser_navigate", {"url": "https://example.com/start"})
    )
    assert result.ok is False
    assert result.error is not None and "redirected off-allowlist" in result.error
    assert "evil.com" in result.error
    assert page.goto_urls == ["https://example.com/start"]  # the goto did happen
    assert page.url == "https://evil.com/landing"  # landed off-allowlist


async def test_read_text_refuses_when_current_page_off_allowlist() -> None:
    # A prior client-side redirect left page.url off-allowlist.
    provider = make_provider(["example.com"])
    browser_with_fake_page(provider, FakePage(body_text="secret bank data", url="https://evil.com"))
    result = await provider.execute(Action("browser_read_text", {}))
    assert result.ok is False
    assert result.error is not None and "off-allowlist" in result.error
    assert "secret" not in result.error  # content never exposed
    assert "evil.com" in result.error


async def test_click_refuses_when_current_page_off_allowlist() -> None:
    provider = make_provider(["example.com"])
    page = browser_with_fake_page(provider, FakePage(url="https://evil.com"))
    result = await provider.execute(Action("browser_click", {"text": "OK"}))
    assert result.ok is False
    assert result.error is not None and "off-allowlist" in result.error
    assert page.clicks == []  # never interacted with off-allowlist page


async def test_type_and_screenshot_refuse_when_current_page_off_allowlist() -> None:
    provider = make_provider(["example.com"])
    page = browser_with_fake_page(provider, FakePage(url="https://evil.com"))
    typed = await provider.execute(Action("browser_type", {"into": "pw", "text": "hunter2"}))
    shot = await provider.execute(Action("browser_screenshot", {}))
    assert typed.ok is False and shot.ok is False
    assert page.fills == [] and page.screenshots == 0

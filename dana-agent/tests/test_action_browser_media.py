"""M1 media-core tools (TASK-011): press_key, scroll, media engine.

All offline via the FakePage canned-script simulation — the provider↔page
contract (script marker + numeric arg → state change) is pinned without
Chromium. Real-browser fixture probes are the M6 live layer.
"""

from __future__ import annotations

import pytest

from dana_agent.config import ActionsConfig
from dana_agent.providers.actions.browser import (
    _MEDIA_ACTIONS,
    _MEDIA_SCRIPTS,
    _PRESS_KEYS,
    BrowserActions,
)
from dana_agent.types import Action, Reversibility
from tests.fixtures.browser import FakePage, browser_with_fake_page


def make_provider(domains: list[str] | None = None) -> BrowserActions:
    return BrowserActions(
        ActionsConfig(provider="browser", allowed_domains=domains or ["example.com"])
    )


def media(action: str, **kwargs: object) -> Action:
    return Action("browser_media", {"action": action, **kwargs})


# --- classify() table for the new tools -----------------------------------


def test_classify_media_and_scroll_are_read_only() -> None:
    provider = make_provider()
    assert provider.classify(media("seek", delta_s=42)) is Reversibility.READ_ONLY
    assert provider.classify(media("state")) is Reversibility.READ_ONLY
    assert provider.classify(Action("browser_scroll", {"dy": 500})) is Reversibility.READ_ONLY


def test_classify_press_key_is_reversible() -> None:
    # Enter can commit a filled form → audited reversible write, and the
    # conservative policy confirms it (same reasoning as browser_type).
    provider = make_provider()
    assert provider.classify(Action("browser_press_key", {"key": "Enter"})) is (
        Reversibility.REVERSIBLE
    )


# --- press_key: bounded vocabulary -----------------------------------------


async def test_press_key_allowed_key_reaches_keyboard() -> None:
    provider = make_provider()
    page = browser_with_fake_page(provider, FakePage(url="https://example.com/watch"))
    result = await provider.execute(Action("browser_press_key", {"key": "l"}))
    assert result.ok is True
    assert page.key_presses == ["l"]


@pytest.mark.parametrize("key", ["Control+A", "meta", "x", "", "ENTER"])
async def test_press_key_outside_vocabulary_is_refused(key: str) -> None:
    provider = make_provider()
    page = browser_with_fake_page(provider, FakePage(url="https://example.com"))
    result = await provider.execute(Action("browser_press_key", {"key": key}))
    assert result.ok is False
    assert result.error is not None and "vocabulary" in result.error
    assert page.key_presses == []  # nothing ever pressed


def test_press_key_vocabulary_is_bounded_and_lowercase_exact() -> None:
    assert len(_PRESS_KEYS) < 50  # small, enumerable surface
    assert "Enter" in _PRESS_KEYS and "enter" not in _PRESS_KEYS


# --- scroll -----------------------------------------------------------------


async def test_scroll_by_pixels_uses_mouse_wheel() -> None:
    provider = make_provider()
    page = browser_with_fake_page(provider, FakePage(url="https://example.com/article"))
    result = await provider.execute(Action("browser_scroll", {"dy": 800}))
    assert result.ok is True
    assert page.wheel_calls == [(0, 800)]


async def test_scroll_up_negative_and_clamped() -> None:
    provider = make_provider()
    page = browser_with_fake_page(provider, FakePage(url="https://example.com"))
    up = await provider.execute(Action("browser_scroll", {"dy": -400}))
    runaway = await provider.execute(Action("browser_scroll", {"dy": 999_999}))
    assert up.ok and runaway.ok
    assert page.wheel_calls == [(0, -400), (0, 3_000)]  # clamped, never runaway


async def test_scroll_to_top_bottom_via_canned_script() -> None:
    provider = make_provider()
    page = browser_with_fake_page(provider, FakePage(url="https://example.com"))
    bottom = await provider.execute(Action("browser_scroll", {"to": "bottom"}))
    top = await provider.execute(Action("browser_scroll", {"to": "top"}))
    assert bottom.ok and top.ok
    assert page.scroll_y == 0.0  # ended at top


async def test_scroll_to_text_uses_locator() -> None:
    provider = make_provider()
    page = browser_with_fake_page(provider, FakePage(url="https://example.com"))
    result = await provider.execute(Action("browser_scroll", {"target": "Comments"}))
    assert result.ok is True
    assert page.scroll_targets == [{"text": "Comments"}]


async def test_scroll_rejects_zero_or_multiple_modes() -> None:
    provider = make_provider()
    browser_with_fake_page(provider, FakePage(url="https://example.com"))
    none_supplied = await provider.execute(Action("browser_scroll", {}))
    both = await provider.execute(Action("browser_scroll", {"dy": 10, "to": "top"}))
    assert none_supplied.ok is False and both.ok is False
    assert "exactly one" in (none_supplied.error or "")


# --- media engine -----------------------------------------------------------


async def test_media_seek_forward_42s_is_precise() -> None:
    provider = make_provider()
    page = browser_with_fake_page(provider, FakePage(url="https://example.com/watch?v=x"))
    result = await provider.execute(media("seek", delta_s=42))
    assert result.ok is True
    assert page.media is not None and page.media["currentTime"] == pytest.approx(42.0)
    assert result.output == "position 0:42"


async def test_media_seek_back_clamps_at_zero() -> None:
    provider = make_provider()
    page = browser_with_fake_page(provider, FakePage(url="https://example.com/watch"))
    result = await provider.execute(media("seek", delta_s=-5))
    assert result.ok is True
    assert page.media is not None and page.media["currentTime"] == 0.0


async def test_media_seek_to_absolute_position() -> None:
    provider = make_provider()
    page = browser_with_fake_page(provider, FakePage(url="https://example.com/watch"))
    result = await provider.execute(media("seek_to", position_s=200))
    assert result.ok is True
    assert page.media is not None and page.media["currentTime"] == pytest.approx(200.0)
    assert result.output == "position 3:20"  # "jump to 3:20"


async def test_media_seek_requires_numeric_delta() -> None:
    provider = make_provider()
    page = browser_with_fake_page(provider, FakePage(url="https://example.com"))
    missing = await provider.execute(media("seek"))
    prose = await provider.execute(media("seek", delta_s="forty two"))
    assert missing.ok is False and prose.ok is False
    assert page.media is not None and page.media["currentTime"] == 0.0


async def test_media_play_pause_toggle() -> None:
    provider = make_provider()
    page = browser_with_fake_page(provider, FakePage(url="https://example.com/watch"))
    assert (await provider.execute(media("play"))).ok is True
    assert page.media is not None and page.media["paused"] is False
    assert (await provider.execute(media("toggle"))).ok is True
    assert page.media is not None and page.media["paused"] is True
    assert page.key_presses == []  # JS path worked; no keyboard fallback fired


async def test_media_rate_bounds() -> None:
    provider = make_provider()
    page = browser_with_fake_page(provider, FakePage(url="https://example.com/watch"))
    ok = await provider.execute(media("rate", rate=1.5))
    too_fast = await provider.execute(media("rate", rate=3.0))
    assert ok.ok is True and too_fast.ok is False
    assert page.media is not None and page.media["rate"] == 1.5
    assert "0.25-2.0" in (too_fast.error or "")


async def test_media_mute_unmute() -> None:
    provider = make_provider()
    page = browser_with_fake_page(provider, FakePage(url="https://example.com/watch"))
    assert (await provider.execute(media("mute"))).ok is True
    assert page.media is not None and page.media["muted"] is True
    assert (await provider.execute(media("unmute"))).ok is True
    assert page.media is not None and page.media["muted"] is False


async def test_media_state_is_speakable() -> None:
    provider = make_provider()
    browser_with_fake_page(provider, FakePage(url="https://example.com/watch"))
    await provider.execute(media("seek_to", position_s=192))
    result = await provider.execute(media("state"))
    assert result.ok is True
    # "what's playing?" → title + position, ready for TTS
    assert "lofi hip hop radio" in result.output
    assert "3:12 / 10:40" in result.output
    assert "paused" in result.output


async def test_media_no_media_element_play_falls_back_to_player_key() -> None:
    provider = make_provider()
    page = FakePage(url="https://example.com/watch")
    page.media = None  # page has no <video>/<audio>
    browser_with_fake_page(provider, page)
    result = await provider.execute(media("play"))
    assert result.ok is True
    assert page.key_presses == ["k"]
    assert "pressed 'k'" in result.output


async def test_media_no_media_element_seek_is_honest_error() -> None:
    provider = make_provider()
    page = FakePage(url="https://example.com")
    page.media = None
    browser_with_fake_page(provider, page)
    result = await provider.execute(media("seek", delta_s=42))
    assert result.ok is False
    assert result.error is not None and "no media element" in result.error


async def test_media_unknown_action_refused() -> None:
    provider = make_provider()
    browser_with_fake_page(provider, FakePage(url="https://example.com"))
    result = await provider.execute(media("download"))
    assert result.ok is False
    assert "unknown media action" in (result.error or "")


# --- allowlist re-assertion for the new tools -------------------------------


@pytest.mark.parametrize(
    ("tool", "arguments"),
    [
        ("browser_press_key", {"key": "Enter"}),
        ("browser_scroll", {"dy": 100}),
        ("browser_media", {"action": "state"}),
    ],
)
async def test_new_tools_refuse_off_allowlist_current_page(
    tool: str, arguments: dict[str, object]
) -> None:
    provider = make_provider()
    page = browser_with_fake_page(provider, FakePage(url="https://evil.com"))
    result = await provider.execute(Action(tool, arguments))
    assert result.ok is False
    assert "off-allowlist" in (result.error or "")
    assert page.key_presses == [] and page.wheel_calls == []


# --- canned-script registry invariants ---------------------------------------


def test_every_media_script_has_dana_marker_and_no_wildcard_input() -> None:
    # The registry is the entire executable JS surface: every entry is marked,
    # and only the provider's own numeric/bool args ever flow in.
    for name, script in _MEDIA_SCRIPTS.items():
        assert script.startswith(f"/* dana:{name} */"), name
        assert "fetch(" not in script and "XMLHttpRequest" not in script, name


def test_media_actions_registry_is_bounded() -> None:
    assert _MEDIA_ACTIONS == frozenset(
        {"play", "pause", "toggle", "seek", "seek_to", "rate", "mute", "unmute", "state"}
    )

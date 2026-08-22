"""TASK-011 M2-M4: tabs & history, semantic search + adapters, OS extensions.

Browser side runs against the FakePage (offline contract tests); OS side uses
injectable factories / monkeypatched user32 so nothing real fires (no window
is ever snapped on the dev machine, mirroring test_action_os.py).
"""

from __future__ import annotations

from typing import Any

import pytest

from dana_agent.config import ActionsConfig
from dana_agent.providers.actions.adapters import ADAPTERS, adapter_for_url, search_url
from dana_agent.providers.actions.browser import BrowserActions
from dana_agent.providers.actions.composite import BrowserOSActions
from dana_agent.providers.actions.os_win import WindowsOSActions
from dana_agent.types import Action, Reversibility
from tests.fixtures.browser import FakeContext, FakePage, browser_with_fake_page


def make_provider(domains: list[str] | None = None) -> BrowserActions:
    return BrowserActions(
        ActionsConfig(provider="browser", allowed_domains=domains or ["example.com"])
    )


def make_os(**kw: Any) -> WindowsOSActions:
    cfg = ActionsConfig(provider="os", os_actions_enabled=True, allowed_apps=[], **kw)
    return WindowsOSActions(cfg)


# --- adapters (M3) -----------------------------------------------------------


def test_search_url_builds_and_encodes() -> None:
    assert (
        search_url("youtube", "lofi hip hop")
        == "https://www.youtube.com/results?search_query=lofi+hip+hop"
    )
    assert search_url("google", "a&b") == "https://www.google.com/search?q=a%26b"
    assert search_url("wikpedia", "x") is None  # typo'd engine → default-deny


def test_adapter_for_url_matches_host_suffixes() -> None:
    yt = adapter_for_url("https://music.youtube.com/watch?v=1")
    assert yt is not None and yt.name == "youtube"
    assert adapter_for_url("https://example.com") is None


def test_every_adapter_is_verified_stamped() -> None:
    for adapter in ADAPTERS.values():
        assert adapter.verified, adapter.name  # unstamped facts are untested facts


# --- browser_search -----------------------------------------------------------


async def test_search_navigates_to_engine_url_and_logs_visit() -> None:
    provider = make_provider(["youtube.com"])
    page = browser_with_fake_page(provider, FakePage(url="https://youtube.com"))
    result = await provider.execute(
        Action("browser_search", {"query": "lofi hip hop", "engine": "youtube"})
    )
    assert result.ok is True
    assert page.goto_urls == ["https://www.youtube.com/results?search_query=lofi+hip+hop"]
    assert provider._visits and str(provider._visits[0]["url"]).startswith(
        "https://www.youtube.com"
    )


async def test_search_open_first_clicks_adapter_selector() -> None:
    provider = make_provider(["duckduckgo.com"])
    page = browser_with_fake_page(provider, FakePage(url="https://duckduckgo.com"))
    result = await provider.execute(
        Action("browser_search", {"query": "dana", "engine": "duckduckgo", "open_first": True})
    )
    assert result.ok is True
    css_clicked = [c["css"] for c in page.clicks if "css" in c]
    assert css_clicked == [ADAPTERS["duckduckgo"].first_result_selector]


async def test_search_unknown_engine_refused_without_navigation() -> None:
    provider = make_provider(["example.com"])
    page = browser_with_fake_page(provider, FakePage(url="https://example.com"))
    result = await provider.execute(Action("browser_search", {"query": "x", "engine": "altavista"}))
    assert result.ok is False
    assert "unknown search engine" in (result.error or "")
    assert page.goto_urls == []


async def test_search_off_allowlist_engine_refused() -> None:
    # The engine exists but its host was never opted in → default-deny.
    provider = make_provider(["example.com"])
    browser_with_fake_page(provider, FakePage(url="https://example.com"))
    result = await provider.execute(Action("browser_search", {"query": "x", "engine": "youtube"}))
    assert result.ok is False
    assert "not allowlisted" in (result.error or "")


def test_classify_search_table() -> None:
    provider = make_provider(["youtube.com"])
    plain = Action("browser_search", {"query": "x", "engine": "youtube"})
    opened = Action("browser_search", {"query": "x", "engine": "youtube", "open_first": True})
    off = Action("browser_search", {"query": "x", "engine": "google"})
    unknown = Action("browser_search", {"query": "x", "engine": "lycos"})
    assert provider.classify(plain) is Reversibility.READ_ONLY
    assert provider.classify(opened) is Reversibility.REVERSIBLE
    assert provider.classify(off) is Reversibility.OUT_OF_SCOPE
    assert provider.classify(unknown) is Reversibility.OUT_OF_SCOPE


# --- browser_tab (M2) ----------------------------------------------------------


async def test_tab_new_switch_list_close_cycle() -> None:
    provider = make_provider(["example.com"])
    first = FakePage(url="https://example.com/a", doc_title="Alpha")
    browser_with_fake_page(provider, first)
    provider._ctx = FakeContext()
    provider._tabs = [first]

    new = await provider.execute(
        Action("browser_tab", {"action": "new", "url": "https://example.com/b"})
    )
    assert new.ok is True
    assert provider._page is not first  # new tab became active
    assert len(provider._tracked_tabs()) == 2

    switch = await provider.execute(Action("browser_tab", {"action": "switch", "title": "Alpha"}))
    assert switch.ok is True
    assert provider._page is first

    listing = await provider.execute(Action("browser_tab", {"action": "list"}))
    assert listing.ok is True
    assert "Alpha" in listing.output and "2" in listing.output

    closed = await provider.execute(Action("browser_tab", {"action": "close"}))
    assert closed.ok is True
    assert first.closed is True
    assert provider._page is not first  # fell back to the remaining tab


async def test_tab_switch_by_url_substring() -> None:
    provider = make_provider(["example.com"])
    first = FakePage(url="https://example.com/watch", doc_title="Video")
    browser_with_fake_page(provider, first)
    provider._tabs = [first]
    result = await provider.execute(Action("browser_tab", {"action": "switch", "title": "watch"}))
    assert result.ok is True
    assert provider._page is first


async def test_tab_close_no_match_is_error_not_crash() -> None:
    provider = make_provider(["example.com"])
    page = browser_with_fake_page(provider, FakePage(url="https://example.com"))
    provider._tabs = [page]
    result = await provider.execute(
        Action("browser_tab", {"action": "switch", "title": "nonexistent-thing"})
    )
    assert result.ok is False
    assert "no tab matches" in (result.error or "")


def test_classify_tab_table() -> None:
    provider = make_provider()
    assert provider.classify(Action("browser_tab", {"action": "new"})) is Reversibility.REVERSIBLE
    assert provider.classify(Action("browser_tab", {"action": "close"})) is Reversibility.REVERSIBLE
    assert provider.classify(Action("browser_tab", {"action": "list"})) is Reversibility.READ_ONLY
    assert provider.classify(Action("browser_tab", {"action": "teleport"})) is (
        Reversibility.OUT_OF_SCOPE
    )


# --- browser_history (M2) -------------------------------------------------------


async def test_history_back_forward_and_visit_log() -> None:
    provider = make_provider(["example.com"])
    page = browser_with_fake_page(provider, FakePage(url="https://example.com"))
    await provider.execute(Action("browser_navigate", {"url": "https://example.com/one"}))
    await provider.execute(Action("browser_navigate", {"url": "https://example.com/two"}))

    back = await provider.execute(Action("browser_history", {"action": "back"}))
    fwd = await provider.execute(Action("browser_history", {"action": "forward"}))
    assert back.ok and fwd.ok
    assert page.history_calls == ["back", "forward"]

    listing = await provider.execute(Action("browser_history", {"action": "list"}))
    assert listing.ok is True
    assert "/one" in listing.output and "/two" in listing.output
    # most-recent first
    assert listing.output.index("/two") < listing.output.index("/one")


async def test_history_goto_reopens_earlier_visit() -> None:
    provider = make_provider(["example.com"])
    page = browser_with_fake_page(provider, FakePage(url="https://example.com"))
    await provider.execute(Action("browser_navigate", {"url": "https://example.com/first"}))
    await provider.execute(Action("browser_navigate", {"url": "https://example.com/second"}))
    page.goto_urls.clear()

    result = await provider.execute(Action("browser_history", {"action": "goto", "index": 1}))
    assert result.ok is True
    assert page.goto_urls == ["https://example.com/first"]  # oldest of the window


async def test_history_goto_bad_index_refused() -> None:
    provider = make_provider(["example.com"])
    browser_with_fake_page(provider, FakePage(url="https://example.com"))
    result = await provider.execute(Action("browser_history", {"action": "goto", "index": 99}))
    assert result.ok is False
    assert "bad visit index" in (result.error or "")


def test_classify_history_table() -> None:
    provider = make_provider(["example.com"])
    provider._visits = [{"url": "https://example.com/x", "title": "x", "at": 0.0}]
    assert provider.classify(Action("browser_history", {"action": "back"})) is (
        Reversibility.READ_ONLY
    )
    goto_ok = Action("browser_history", {"action": "goto", "index": 1})
    goto_bad = Action("browser_history", {"action": "goto", "index": 42})
    assert provider.classify(goto_ok) is Reversibility.READ_ONLY
    assert provider.classify(goto_bad) is Reversibility.OUT_OF_SCOPE


# --- browser_get_state (M3) -------------------------------------------------------


async def test_get_state_summarizes_page_tabs_media() -> None:
    provider = make_provider(["example.com"])
    page = FakePage(url="https://example.com/watch", doc_title="Cat Video")
    browser_with_fake_page(provider, page)
    provider._tabs = [page]
    result = await provider.execute(Action("browser_get_state", {}))
    assert result.ok is True
    assert "Cat Video" in result.output
    assert "tabs: 1" in result.output
    assert "lofi hip hop radio" in result.output  # media summary via state script


async def test_get_state_refuses_off_allowlist() -> None:
    provider = make_provider(["example.com"])
    browser_with_fake_page(provider, FakePage(url="https://evil.com"))
    result = await provider.execute(Action("browser_get_state", {}))
    assert result.ok is False
    assert "off-allowlist" in (result.error or "")


# --- composite provider (browser_os) ------------------------------------------------


def test_composite_classifies_by_prefix() -> None:
    provider = BrowserOSActions(ActionsConfig(provider="browser_os", os_actions_enabled=True))
    assert provider.classify(Action("browser_media", {"action": "state"})) is (
        Reversibility.READ_ONLY
    )
    assert provider.classify(Action("os_media_key", {"key": "play_pause"})) is (
        Reversibility.READ_ONLY
    )


def test_composite_os_disabled_still_refuses_os_tools() -> None:
    provider = BrowserOSActions(ActionsConfig(provider="browser_os", os_actions_enabled=False))
    assert provider.classify(Action("os_media_key", {"key": "next"})) is (
        Reversibility.OUT_OF_SCOPE
    )


def test_composite_registered_and_tool_specs_merge() -> None:
    from dana_agent.brain.tools import action_tool_specs
    from dana_agent.providers.registry import get_class

    assert get_class("actions", "browser_os") is BrowserOSActions
    cfg = ActionsConfig(provider="browser_os", os_actions_enabled=True)
    names = {s.name for s in action_tool_specs(cfg)}
    assert "browser_media" in names and "os_volume" in names
    # OS half stays opt-in even on the composite provider.
    assert "os_volume" not in {
        s.name
        for s in action_tool_specs(ActionsConfig(provider="browser_os", os_actions_enabled=False))
    }


# --- vi destructive click words (M5 partial) -----------------------------------------


@pytest.mark.parametrize(
    "label", ["Gửi", "Xóa tài khoản", "Thanh toán", "Đặt hàng ngay", "Xác nhận"]
)
def test_vietnamese_committing_clicks_are_destructive(label: str) -> None:
    provider = make_provider()
    assert provider.classify(Action("browser_click", {"text": label})) is (
        Reversibility.DESTRUCTIVE
    )


def test_vietnamese_benign_click_stays_reversible() -> None:
    provider = make_provider()
    assert provider.classify(Action("browser_click", {"text": "Phát video"})) is (
        Reversibility.REVERSIBLE
    )


# --- os_volume (M4) -------------------------------------------------------------------


class FakeEndpointVolume:
    def __init__(self, level: float = 0.42) -> None:
        self.level = level
        self.muted = False
        self.set_calls: list[float] = []

    def GetMasterVolumeLevelScalar(self) -> float:
        return self.level

    def SetMasterVolumeLevelScalar(self, v: float, _ctx: object) -> None:
        self.level = v
        self.set_calls.append(v)

    def SetMute(self, m: int, _ctx: object) -> None:
        self.muted = bool(m)


async def test_os_volume_get_set_mute_with_injected_endpoint() -> None:
    provider = make_os()
    fake = FakeEndpointVolume()
    provider._volume_factory = lambda: fake

    got = await provider.execute(Action("os_volume", {"action": "get"}))
    assert got.ok is True and "42%" in got.output

    set_ = await provider.execute(Action("os_volume", {"action": "set", "percent": 30}))
    assert set_.ok is True
    assert fake.level == pytest.approx(0.30)

    await provider.execute(Action("os_volume", {"action": "mute"}))
    assert fake.muted is True


async def test_os_volume_rejects_bad_percent_and_action() -> None:
    provider = make_os()
    provider._volume_factory = lambda: FakeEndpointVolume()
    loud = await provider.execute(Action("os_volume", {"action": "set", "percent": 150}))
    prose = await provider.execute(Action("os_volume", {"action": "set", "percent": "loud"}))
    weird = await provider.execute(Action("os_volume", {"action": "crank"}))
    assert loud.ok is False and prose.ok is False and weird.ok is False


def test_classify_volume_table() -> None:
    provider = make_os()
    assert provider.classify(Action("os_volume", {"action": "get"})) is Reversibility.READ_ONLY
    assert provider.classify(Action("os_volume", {"action": "set", "percent": 30})) is (
        Reversibility.REVERSIBLE
    )
    assert provider.classify(Action("os_volume", {"action": "boost"})) is (
        Reversibility.OUT_OF_SCOPE
    )


# --- os_window (M4) --------------------------------------------------------------------


class FakeUser32Module:
    """Stands in for ctypes.windll.user32: records window operations."""

    def __init__(self) -> None:
        self.show_calls: list[tuple[int, int]] = []
        self.posts: list[tuple[int, int]] = []
        self.key_chords: list[int] = []
        self.foreground = 101
        self.procedures: dict[str, Any] = {}

    def __getattr__(self, name: str) -> Any:
        def _rec(*a: Any) -> Any:
            if name == "ShowWindow":
                self.show_calls.append((int(a[0]), int(a[1])))
                return 1
            if name == "PostMessageW":
                self.posts.append((int(a[0]), int(a[1])))
                return 1
            if name == "keybd_event":
                self.key_chords.append(int(a[0]))
                return None
            if name == "SetForegroundWindow":
                self.foreground = int(a[0])
                return 1
            if name == "EnumWindows":
                return 0  # no windows → title match path finds nothing
            if name == "GetForegroundWindow":
                return self.foreground
            return 1

        return _rec


def patch_user32(monkeypatch: pytest.MonkeyPatch, fake: FakeUser32Module) -> None:
    import ctypes

    class _Windll:
        user32 = fake

    monkeypatch.setattr(ctypes, "windll", _Windll(), raising=False)


def test_os_window_minimize_targets_foreground(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = make_os()
    fake = FakeUser32Module()
    patch_user32(monkeypatch, fake)
    result = provider._window("minimize", None)
    assert result.ok is True
    assert fake.show_calls == [(101, 6)]  # SW_MINIMIZE on the foreground hwnd


def test_os_window_snap_sends_win_arrow_chord(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = make_os()
    fake = FakeUser32Module()
    patch_user32(monkeypatch, fake)
    result = provider._window("snap_right", None)
    assert result.ok is True
    assert fake.key_chords == [0x5B, 0x27, 0x27 | 0x0002, 0x5B | 0x0002]  # Win+Right chord


def test_os_window_close_posts_wm_close(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = make_os()
    fake = FakeUser32Module()
    patch_user32(monkeypatch, fake)
    result = provider._window("close", None)
    assert result.ok is True
    assert fake.posts == [(101, 0x0010)]


def test_classify_window_table() -> None:
    provider = make_os()
    assert provider.classify(Action("os_window", {"action": "snap_left"})) is (
        Reversibility.REVERSIBLE
    )
    assert provider.classify(Action("os_window", {"action": "close"})) is Reversibility.DESTRUCTIVE
    assert provider.classify(Action("os_window", {"action": "destroy"})) is (
        Reversibility.OUT_OF_SCOPE
    )


# --- os_media_state (M4) ----------------------------------------------------------------


async def test_os_media_state_with_injected_session() -> None:
    provider = make_os()

    async def fake_factory() -> Any:
        return {"title": "Blinding Lights", "artist": "The Weeknd", "status": "Playing"}

    provider._smtc_factory = fake_factory
    result = await provider.execute(Action("os_media_state", {}))
    assert result.ok is True
    assert "Blinding Lights" in result.output and "Playing" in result.output


async def test_os_media_state_no_session() -> None:
    provider = make_os()

    async def empty() -> Any:
        return None

    provider._smtc_factory = empty
    result = await provider.execute(Action("os_media_state", {}))
    assert result.ok is True
    assert "no active media session" in result.output


# --- M5: forbid_purchases -------------------------------------------------------


def test_forbid_purchases_refuses_purchase_clicks_outright() -> None:
    provider = BrowserActions(
        ActionsConfig(provider="browser", allowed_domains=["shop.example"], forbid_purchases=True)
    )
    pay = provider.classify(Action("browser_click", {"text": "Buy now"}))
    vi = provider.classify(Action("browser_click", {"text": "Thanh toán"}))
    submit = provider.classify(Action("browser_click", {"text": "Submit"}))
    assert pay is Reversibility.OUT_OF_SCOPE  # refused, not confirmed
    assert vi is Reversibility.OUT_OF_SCOPE
    assert submit is Reversibility.DESTRUCTIVE  # non-purchase destructive still confirms


def test_purchases_confirm_when_not_forbidden() -> None:
    provider = make_provider(["shop.example"])
    assert provider.classify(Action("browser_click", {"text": "Buy now"})) is (
        Reversibility.DESTRUCTIVE
    )


# --- M5: human-unit confirmation previews ---------------------------------------


def test_human_preview_phrases_in_human_units() -> None:
    from dana_agent.actions.audit import human_preview

    assert human_preview("browser_media", {"action": "seek", "delta_s": 42}) == "seek 0:42"
    assert human_preview("browser_media", {"action": "seek", "delta_s": -5}) == "seek -0:05"
    assert (
        human_preview("browser_media", {"action": "seek_to", "position_s": 200}) == "seek to 3:20"
    )
    assert human_preview("browser_media", {"action": "rate", "rate": 1.5}) == "set speed 1.5x"
    assert (
        human_preview("browser_search", {"query": "lofi", "engine": "youtube", "open_first": True})
        == "search youtube for 'lofi' and open the first result"
    )
    assert (
        human_preview("os_volume", {"action": "set", "percent": 30}) == "set system volume to 30%"
    )


def test_human_preview_redacts_and_falls_back() -> None:
    from dana_agent.actions.audit import human_preview

    preview = human_preview("browser_type", {"into": "Password", "text": "hunter2"})
    assert "hunter2" not in preview  # never speak a secret back
    assert "browser_type" in preview

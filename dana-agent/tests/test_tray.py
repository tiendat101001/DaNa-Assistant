"""System tray logic — exercised with a fake icon/controller (no real tray).

pystray/pillow are the ``control`` extra and are not installed in CI, so the
import guard and the menu-callback logic are tested without ever creating a real
tray icon or a message-pump thread.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from dana_agent.control.tray import Tray
from tests.fakes.control import FakeController


class _FakeIcon:
    def __init__(self) -> None:
        self.update_menu_calls = 0
        self.stopped = False

    def update_menu(self) -> None:
        self.update_menu_calls += 1

    def stop(self) -> None:
        self.stopped = True


class _FakeLoop:
    def __init__(self) -> None:
        self.soon_calls: list[Any] = []

    def call_soon_threadsafe(self, fn: Any, *args: Any) -> None:
        self.soon_calls.append((fn, args))

    def stop(self) -> None:  # referenced as self._loop.stop by Tray._quit
        pass


@pytest.fixture
def scheduled(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    """Capture coroutines handed to run_coroutine_threadsafe (and close them so
    no 'coroutine was never awaited' warning fires)."""
    captured: list[Any] = []

    def fake_schedule(coro: Any, _loop: Any) -> None:
        captured.append(coro)
        coro.close()

    monkeypatch.setattr("asyncio.run_coroutine_threadsafe", fake_schedule)
    return captured


def _make_tray() -> tuple[Tray, _FakeIcon, FakeController, _FakeLoop]:
    loop = _FakeLoop()
    controller = FakeController()
    t = Tray(loop, controller, "http://127.0.0.1:8760")  # type: ignore[arg-type]
    icon = _FakeIcon()
    t._icon = icon
    return t, icon, controller, loop


def test_import_guard_raises_clear_error() -> None:
    # pystray/pillow absent → start() surfaces an actionable ImportError.
    loop = asyncio.new_event_loop()
    try:
        t = Tray(loop, FakeController(), "http://127.0.0.1:8760")
        with pytest.raises(ImportError, match="pystray"):
            t.start()
    finally:
        loop.close()


def test_mute_label_reflects_state() -> None:
    t, _icon, controller, _loop = _make_tray()
    assert t._mute_label() == "Mute mic"
    controller._muted = True
    assert t._mute_label() == "Unmute mic"


def test_toggle_mute_schedules_mute_without_eager_menu_refresh(scheduled: list[Any]) -> None:
    t, icon, controller, _loop = _make_tray()
    assert controller.is_muted() is False
    t._toggle_mute()
    assert len(scheduled) == 1  # a coroutine was marshalled to the loop
    # update_menu() is NOT called eagerly — is_muted() still returns the old value
    # at this point, so an eager call would show the wrong label.
    assert icon.update_menu_calls == 0


def test_toggle_mute_when_muted_schedules_unmute(scheduled: list[Any]) -> None:
    t, _icon, controller, _loop = _make_tray()
    controller._muted = True
    t._toggle_mute()
    assert len(scheduled) == 1


def test_open_dashboard_opens_url(monkeypatch: pytest.MonkeyPatch) -> None:
    opened: list[str] = []
    monkeypatch.setattr("webbrowser.open", lambda url: opened.append(url))
    t, _icon, _controller, _loop = _make_tray()
    t._open_dashboard()
    assert opened == ["http://127.0.0.1:8760"]


def test_quit_stops_icon_and_loop() -> None:
    t, icon, _controller, loop = _make_tray()
    t._quit()
    assert icon.stopped is True
    assert len(loop.soon_calls) == 1  # loop.stop scheduled threadsafe


def test_stop_is_safe_before_start() -> None:
    t = Tray(_FakeLoop(), FakeController(), "http://x")  # type: ignore[arg-type]
    t.stop()  # no icon yet → no-op, must not raise

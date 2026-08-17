"""Windows OS action provider: off-by-default, app allowlist enforcement, the
classify() table, unknown media keys, and cross-OS import. ctypes is monkeypatched
so nothing real fires and the module imports on any OS (safe-computer-use skill)."""

from __future__ import annotations

import ctypes
from typing import Any

import pytest

from dana_agent.config import ActionsConfig
from dana_agent.providers.actions.os_win import MEDIA_KEYS, WindowsOSActions
from dana_agent.types import Action, Reversibility


def make_provider(*, enabled: bool = True, apps: list[str] | None = None) -> WindowsOSActions:
    cfg = ActionsConfig(provider="os", os_actions_enabled=enabled, allowed_apps=apps or [])
    return WindowsOSActions(cfg)


# --- module imports on any OS --------------------------------------------


def test_module_imports_and_registers_on_any_os() -> None:
    # Reaching this line proves the module imported with no top-level windll.
    from dana_agent.providers.registry import get_class

    assert get_class("actions", "os") is WindowsOSActions
    assert "play_pause" in MEDIA_KEYS


# --- off by default ------------------------------------------------------


def test_disabled_by_default_classifies_out_of_scope() -> None:
    provider = make_provider(enabled=False, apps=["notepad"])
    assert provider.classify(Action("os_clipboard_get", {})) is Reversibility.OUT_OF_SCOPE
    assert provider.classify(Action("os_media_key", {"key": "play_pause"})) is (
        Reversibility.OUT_OF_SCOPE
    )


async def test_disabled_execute_refuses() -> None:
    provider = make_provider(enabled=False, apps=["notepad"])
    result = await provider.execute(Action("os_launch_app", {"name": "notepad"}))
    assert result.ok is False
    assert result.error is not None and "disabled" in result.error


# --- launch app allowlist ------------------------------------------------


def test_launch_not_allowlisted_is_out_of_scope() -> None:
    provider = make_provider(enabled=True, apps=[])
    assert provider.classify(Action("os_launch_app", {"name": "notepad"})) is (
        Reversibility.OUT_OF_SCOPE
    )


def test_launch_allowlisted_is_reversible() -> None:
    provider = make_provider(enabled=True, apps=["notepad"])
    assert provider.classify(Action("os_launch_app", {"name": "notepad"})) is (
        Reversibility.REVERSIBLE
    )


async def test_launch_refused_when_not_allowlisted() -> None:
    provider = make_provider(enabled=True, apps=["calculator"])
    result = await provider.execute(Action("os_launch_app", {"name": "notepad"}))
    assert result.ok is False
    assert result.error is not None and "not allowlisted" in result.error


# --- classify() table ----------------------------------------------------


def test_classify_table() -> None:
    provider = make_provider(enabled=True, apps=["notepad"])
    assert provider.classify(Action("os_clipboard_get", {})) is Reversibility.READ_ONLY
    assert provider.classify(Action("os_focus_window", {"title": "x"})) is Reversibility.READ_ONLY
    assert provider.classify(Action("os_media_key", {"key": "next"})) is Reversibility.READ_ONLY
    assert provider.classify(Action("os_clipboard_set", {"text": "x"})) is Reversibility.REVERSIBLE
    assert provider.classify(Action("os_whatever", {})) is Reversibility.OUT_OF_SCOPE


# --- unknown tool / unknown media key (no ctypes needed) -----------------


async def test_unknown_tool_is_structured_error() -> None:
    provider = make_provider(enabled=True, apps=["notepad"])
    result = await provider.execute(Action("os_reboot", {}))
    assert result.ok is False
    assert result.error is not None and "unknown os tool" in result.error


async def test_unknown_media_key_refused_before_touching_ctypes() -> None:
    # An unknown key returns an error BEFORE any ctypes.windll access, so this
    # runs on Linux with no monkeypatch at all.
    provider = make_provider(enabled=True, apps=[])
    result = await provider.execute(Action("os_media_key", {"key": "levitate"}))
    assert result.ok is False
    assert result.error is not None and "unknown media key" in result.error


# --- ctypes monkeypatched so a real media key fires nothing --------------


class _FakeUser32:
    def __init__(self) -> None:
        self.keys: list[tuple[int, int]] = []

    def keybd_event(self, vk: int, scan: int, flags: int, extra: int) -> None:
        self.keys.append((vk, flags))


class _FakeWindll:
    def __init__(self) -> None:
        self.user32 = _FakeUser32()


async def test_media_key_uses_monkeypatched_ctypes(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeWindll()
    monkeypatch.setattr(ctypes, "windll", fake, raising=False)
    provider = make_provider(enabled=True, apps=[])
    result = await provider.execute(Action("os_media_key", {"key": "play_pause"}))
    assert result.ok is True and "play_pause" in result.output
    # a key-down (flags 0) then key-up (flags 0x0002), both on VK 0xB3
    assert [flags for _vk, flags in fake.user32.keys] == [0, 0x0002]
    assert {vk for vk, _flags in fake.user32.keys} == {0xB3}


def test_capabilities_are_declared() -> None:
    provider = make_provider(enabled=True)
    caps: Any = provider.capabilities
    assert caps.self_hosted is True and caps.streaming is False

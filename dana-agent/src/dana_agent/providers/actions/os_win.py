"""Bounded Windows OS actions — a tightly-scoped, off-by-default ActionProvider.

OS automation is a more dangerous layer than the browser, so it is: OFF by
default (``cfg.os_actions_enabled``), **API-first** (no GUI scraping / coordinate
clicking), and bounded to a small tool set — launch an *allowlisted* app, focus a
window, send a media key, get/set the clipboard.

Trust boundary (safe-computer-use skill): this provider only ``classify()``s
honestly and ``execute()``s an already-authorised action. Gating/confirmation is
``ActionGate``'s job. ``OUT_OF_SCOPE`` is how it refuses (not allowlisted, or OS
actions disabled).

Cross-platform import: everything Windows-specific (``ctypes.windll``,
``os.startfile``) is resolved INSIDE methods, so this module imports cleanly on
Linux CI. ``ctypes`` itself is stdlib and safe to import at module top level;
``ctypes.windll`` (Windows-only) is never touched at import time.
"""

from __future__ import annotations

import os
import subprocess
from typing import Any

from dana_agent.config import ActionsConfig
from dana_agent.providers.base import ActionProvider, Capabilities
from dana_agent.providers.registry import register
from dana_agent.types import Action, ActionResult, Reversibility

# App NAME -> resolved launch target. The model supplies a name (never a shell
# string); only names that ALSO appear in cfg.allowed_apps are launchable.
_APP_TABLE: dict[str, str] = {
    "notepad": "notepad.exe",
    "calculator": "calc.exe",
    "explorer": "explorer.exe",
    "browser": "msedge.exe",
    "spotify": os.path.expandvars(r"%APPDATA%\Spotify\Spotify.exe"),
}

# Media virtual-key codes (see research §B.3). Sent via keybd_event.
_MEDIA_VK: dict[str, int] = {
    "play_pause": 0xB3,
    "next": 0xB0,
    "prev": 0xB1,
    "stop": 0xB2,
    "vol_up": 0xAF,
    "vol_down": 0xAE,
    "mute": 0xAD,
}
_KEYEVENTF_KEYUP = 0x0002
_CF_UNICODETEXT = 13
_GMEM_MOVEABLE = 0x0002
_SW_RESTORE = 9

_KNOWN_TOOLS = frozenset(
    {
        "os_launch_app",
        "os_focus_window",
        "os_media_key",
        "os_clipboard_get",
        "os_clipboard_set",
        "os_volume",
        "os_window",
        "os_media_state",
    }
)

_WINDOW_ACTIONS = frozenset({"snap_left", "snap_right", "minimize", "maximize", "restore", "close"})
# Win32 ShowWindow commands + the close message.
_SW_MINIMIZE, _SW_MAXIMIZE, _SW_RESTORE = 6, 3, 9
_WM_CLOSE = 0x0010
_VK_LWIN, _VK_LEFT, _VK_RIGHT = 0x5B, 0x25, 0x27


@register("actions", "os")
class WindowsOSActions(ActionProvider):
    """Bounded, API-first Windows control. Off unless ``os_actions_enabled``."""

    def __init__(self, cfg: ActionsConfig) -> None:
        self._cfg = cfg
        self._enabled = cfg.os_actions_enabled
        self._allowed_apps = frozenset(a.lower().strip() for a in cfg.allowed_apps if a.strip())
        # Injectable factories (offline tests substitute fakes; the real paths
        # import optional extras lazily inside the factory).
        self._volume_factory: Any = None
        self._smtc_factory: Any = None

    @property
    def capabilities(self) -> Capabilities:
        return Capabilities(
            streaming=False,
            languages=frozenset({"en", "vi"}),
            self_hosted=True,  # runs entirely on-device (no network)
            latency_class="fast",
        )

    def _app_allowlisted(self, name: str) -> bool:
        return name.lower().strip() in self._allowed_apps

    # --- classification (honest, so the gate can decide) -----------------

    def classify(self, action: Action) -> Reversibility:
        if not self._enabled:
            return Reversibility.OUT_OF_SCOPE  # OS actions disabled → refuse
        tool = action.tool
        if tool == "os_clipboard_get":
            return Reversibility.READ_ONLY
        if tool in ("os_focus_window", "os_media_key", "os_media_state"):
            return Reversibility.READ_ONLY  # transient control; no lasting state
        if tool == "os_volume":
            act = str(action.arguments.get("action", ""))
            if act == "get":
                return Reversibility.READ_ONLY
            if act in ("set", "mute", "unmute"):
                return Reversibility.REVERSIBLE  # mixer state; trivially set back
            return Reversibility.OUT_OF_SCOPE
        if tool == "os_window":
            act = str(action.arguments.get("action", ""))
            if act not in _WINDOW_ACTIONS:
                return Reversibility.OUT_OF_SCOPE
            if act == "close":
                # Closing may destroy unsaved work → the confirmation gate.
                return Reversibility.DESTRUCTIVE
            return Reversibility.REVERSIBLE  # snap/min/max/restore: transient layout
        if tool == "os_clipboard_set":
            return Reversibility.REVERSIBLE  # overwrites clipboard; recoverable
        if tool == "os_launch_app":
            name = str(action.arguments.get("name", ""))
            return (
                Reversibility.REVERSIBLE
                if self._app_allowlisted(name)
                else Reversibility.OUT_OF_SCOPE
            )
        return Reversibility.OUT_OF_SCOPE

    # --- execution -------------------------------------------------------

    async def execute(self, action: Action) -> ActionResult:
        if not self._enabled:
            return ActionResult(ok=False, error="refused: OS actions are disabled")
        if action.tool not in _KNOWN_TOOLS:
            return ActionResult(ok=False, error=f"unknown os tool {action.tool!r}")
        args = action.arguments
        try:
            if action.tool == "os_launch_app":
                return self._launch(str(args.get("name", "")))
            if action.tool == "os_focus_window":
                return self._focus_window(str(args.get("title", "")))
            if action.tool == "os_media_key":
                return self._media_key(str(args.get("key", "")))
            if action.tool == "os_clipboard_get":
                return self._clipboard_get()
            if action.tool == "os_clipboard_set":
                return self._clipboard_set(str(args.get("text", "")))
            if action.tool == "os_volume":
                return await self._volume(str(args.get("action", "")), args.get("percent"))
            if action.tool == "os_window":
                return self._window(str(args.get("action", "")), args.get("title"))
            if action.tool == "os_media_state":
                return await self._media_state()
            return ActionResult(ok=False, error=f"unknown os tool {action.tool!r}")
        except Exception as exc:  # noqa: BLE001 — structured error, never a raw traceback
            return ActionResult(ok=False, error=f"{type(exc).__name__}: {exc}")

    # --- individual bounded operations -----------------------------------

    def _launch(self, name: str) -> ActionResult:
        if not self._app_allowlisted(name):
            return ActionResult(ok=False, error=f"refused: app not allowlisted: {name}")
        exe = _APP_TABLE.get(name.lower().strip())
        if not exe:
            return ActionResult(ok=False, error=f"refused: no launch target mapped for {name}")
        startfile = getattr(os, "startfile", None)  # Windows-only attribute
        if startfile is not None:
            startfile(exe)  # shell association, no user shell STRING involved
            return ActionResult(ok=True, output=f"launched {name}")
        # Fallback (non-Windows / no startfile): list argv, never shell=True.
        subprocess.Popen([exe], shell=False)  # noqa: S603 — fixed exe from allowlist, no shell
        return ActionResult(ok=True, output=f"launched {name}")

    def _focus_window(self, title_substr: str) -> ActionResult:
        if not title_substr:
            return ActionResult(ok=False, error="refused: empty window title")
        hwnd = self._resolve_hwnd(title_substr)
        if hwnd is None:
            return ActionResult(ok=False, error=f"no window matching {title_substr!r}")
        user32 = self._user32()
        user32.ShowWindow(hwnd, _SW_RESTORE)  # un-minimize first
        user32.SetForegroundWindow(hwnd)
        return ActionResult(ok=True, output=f"focused hwnd {hwnd}")

    @staticmethod
    def _user32() -> Any:
        """user32 with pointer-sized HWND prototypes (64-bit safe)."""
        import ctypes  # noqa: PLC0415
        from ctypes import wintypes  # noqa: PLC0415

        user32 = ctypes.windll.user32
        user32.IsWindowVisible.argtypes = [wintypes.HWND]
        user32.IsWindowVisible.restype = wintypes.BOOL
        user32.GetWindowTextLengthW.argtypes = [wintypes.HWND]
        user32.GetWindowTextLengthW.restype = ctypes.c_int
        user32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
        user32.GetWindowTextW.restype = ctypes.c_int
        user32.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
        user32.ShowWindow.restype = wintypes.BOOL
        user32.SetForegroundWindow.argtypes = [wintypes.HWND]
        user32.SetForegroundWindow.restype = wintypes.BOOL
        user32.GetForegroundWindow.restype = wintypes.HWND
        user32.PostMessageW.argtypes = [
            wintypes.HWND,
            ctypes.c_uint,
            wintypes.WPARAM,
            wintypes.LPARAM,
        ]
        user32.PostMessageW.restype = wintypes.BOOL
        return user32

    def _resolve_hwnd(self, title_substr: str | None) -> int | None:
        """The first visible window whose title contains ``title_substr`` (or the
        foreground window when no title is given). Windows-only; caller gates."""
        import ctypes  # noqa: PLC0415
        from ctypes import wintypes  # noqa: PLC0415

        user32 = self._user32()
        if title_substr is None:
            hwnd = user32.GetForegroundWindow()
            return int(hwnd) if hwnd else None
        match: list[int] = []
        needle = title_substr.lower()
        enum_proc = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

        def _cb(hwnd: int, _lparam: int) -> bool:
            if not user32.IsWindowVisible(hwnd):
                return True
            length = user32.GetWindowTextLengthW(hwnd)
            if length == 0:
                return True
            buf = ctypes.create_unicode_buffer(length + 1)
            user32.GetWindowTextW(hwnd, buf, length + 1)
            if needle in buf.value.lower():
                match.append(hwnd)
                return False  # stop enumeration
            return True

        # Keep the callback referenced for the whole synchronous call (else GC).
        callback = enum_proc(_cb)
        user32.EnumWindows(callback, 0)
        return match[0] if match else None

    def _media_key(self, key: str) -> ActionResult:
        vk = _MEDIA_VK.get(key)
        if vk is None:
            return ActionResult(ok=False, error=f"unknown media key {key!r}")
        import ctypes  # noqa: PLC0415

        user32 = ctypes.windll.user32
        user32.keybd_event(vk, 0, 0, 0)  # key down
        user32.keybd_event(vk, 0, _KEYEVENTF_KEYUP, 0)  # key up
        return ActionResult(ok=True, output=f"media {key}")

    def _clipboard_get(self) -> ActionResult:
        import ctypes  # noqa: PLC0415

        user32, kernel32 = ctypes.windll.user32, ctypes.windll.kernel32
        # Declare pointer-sized restypes so handles aren't truncated on 64-bit.
        kernel32.GlobalLock.restype = ctypes.c_void_p
        kernel32.GlobalLock.argtypes = [ctypes.c_void_p]
        kernel32.GlobalUnlock.argtypes = [ctypes.c_void_p]
        kernel32.GlobalUnlock.restype = ctypes.c_int
        user32.GetClipboardData.restype = ctypes.c_void_p
        user32.GetClipboardData.argtypes = [ctypes.c_uint]
        if not user32.OpenClipboard(0):
            return ActionResult(ok=False, error="cannot open clipboard")
        try:
            handle = user32.GetClipboardData(_CF_UNICODETEXT)
            if not handle:
                return ActionResult(ok=True, output="")
            ptr = kernel32.GlobalLock(handle)
            text = ctypes.c_wchar_p(ptr).value or ""
            kernel32.GlobalUnlock(handle)
            return ActionResult(ok=True, output=text)
        finally:
            user32.CloseClipboard()

    def _clipboard_set(self, text: str) -> ActionResult:
        import ctypes  # noqa: PLC0415

        user32, kernel32 = ctypes.windll.user32, ctypes.windll.kernel32
        kernel32.GlobalAlloc.restype = ctypes.c_void_p
        kernel32.GlobalAlloc.argtypes = [ctypes.c_uint, ctypes.c_size_t]
        kernel32.GlobalLock.restype = ctypes.c_void_p
        kernel32.GlobalLock.argtypes = [ctypes.c_void_p]
        kernel32.GlobalUnlock.argtypes = [ctypes.c_void_p]
        kernel32.GlobalUnlock.restype = ctypes.c_int
        # HANDLE args must be pointer-sized: passing the moveable-memory handle to
        # SetClipboardData as a default C int truncates/overflows on 64-bit Windows.
        user32.SetClipboardData.argtypes = [ctypes.c_uint, ctypes.c_void_p]
        user32.SetClipboardData.restype = ctypes.c_void_p
        data = text.encode("utf-16-le") + b"\x00\x00"
        if not user32.OpenClipboard(0):
            return ActionResult(ok=False, error="cannot open clipboard")
        try:
            user32.EmptyClipboard()
            handle = kernel32.GlobalAlloc(_GMEM_MOVEABLE, len(data))
            buf = kernel32.GlobalLock(handle)
            ctypes.memmove(buf, data, len(data))
            kernel32.GlobalUnlock(handle)
            result = user32.SetClipboardData(_CF_UNICODETEXT, handle)
            if not result:
                # SetClipboardData failed: clipboard does NOT own the handle, so
                # we must free it ourselves to avoid a global heap leak.
                kernel32.GlobalFree.argtypes = [ctypes.c_void_p]
                kernel32.GlobalFree(handle)
                return ActionResult(ok=False, error="SetClipboardData failed")
            return ActionResult(ok=True, output="clipboard set")
        finally:
            user32.CloseClipboard()

    # --- master volume (IAudioEndpointVolume via pycaw) ---------------------

    def _volume_endpoint(self) -> Any:
        """The speaker endpoint's IAudioEndpointVolume. Injectable for tests."""
        if self._volume_factory is not None:
            return self._volume_factory()
        try:
            from ctypes import POINTER, cast  # noqa: PLC0415

            from comtypes import CLSCTX_ALL  # noqa: PLC0415
            from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "pycaw is required for os_volume: install dana-agent[actions]"
            ) from exc
        devices = AudioUtilities.GetSpeakers()
        interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        return cast(interface, POINTER(IAudioEndpointVolume))

    async def _volume(self, action: str, percent: object) -> ActionResult:
        try:
            vol = self._volume_endpoint()
        except ImportError as exc:
            return ActionResult(ok=False, error=str(exc))
        if action == "get":
            level = round(float(vol.GetMasterVolumeLevelScalar()) * 100)
            return ActionResult(ok=True, output=f"volume {level}%")
        if action == "set":
            if (
                not isinstance(percent, (int, float))
                or isinstance(percent, bool)
                or not (0 <= float(percent) <= 100)
            ):
                return ActionResult(
                    ok=False, error=f"refused: percent must be a number 0-100: {percent!r}"
                )
            vol.SetMasterVolumeLevelScalar(float(percent) / 100.0, None)
            return ActionResult(ok=True, output=f"volume set to {round(float(percent))}%")
        if action in ("mute", "unmute"):
            vol.SetMute(1 if action == "mute" else 0, None)
            return ActionResult(ok=True, output=f"volume {action}d")
        return ActionResult(ok=False, error=f"refused: unknown volume action {action!r}")

    # --- window management ---------------------------------------------------

    def _window(self, action: str, title: object) -> ActionResult:
        if action not in _WINDOW_ACTIONS:
            return ActionResult(ok=False, error=f"refused: unknown window action {action!r}")
        title_substr = title if isinstance(title, str) and title else None
        hwnd = self._resolve_hwnd(title_substr)
        if hwnd is None:
            return ActionResult(ok=False, error=f"no window matching {title_substr!r}")
        user32 = self._user32()
        if action == "close":
            user32.PostMessageW(hwnd, _WM_CLOSE, 0, 0)
            return ActionResult(ok=True, output=f"close requested for hwnd {hwnd}")
        if action == "snap_left" or action == "snap_right":
            # Snap = Win+Left/Right on the foreground window: bring the target
            # forward first, then send the chord through the same bounded
            # keybd_event path the media keys use.
            user32.SetForegroundWindow(hwnd)
            arrow = _VK_LEFT if action == "snap_left" else _VK_RIGHT
            for vk, flags in (
                (_VK_LWIN, 0),
                (arrow, 0),
                (arrow, _KEYEVENTF_KEYUP),
                (_VK_LWIN, _KEYEVENTF_KEYUP),
            ):
                user32.keybd_event(vk, 0, flags, 0)
            return ActionResult(ok=True, output=f"snapped hwnd {hwnd} {action}")
        sw = {
            "minimize": _SW_MINIMIZE,
            "maximize": _SW_MAXIMIZE,
            "restore": _SW_RESTORE,
        }[action]
        user32.ShowWindow(hwnd, sw)
        return ActionResult(ok=True, output=f"window {action} (hwnd {hwnd})")

    # --- system media session (SMTC via winsdk, opt-in) -----------------------

    async def _media_state(self) -> ActionResult:
        """What is playing system-wide (any app), via the Windows SMTC.

        ``winsdk`` is an opt-in extra — missing package is an actionable
        error, not a crash. Tests inject ``_smtc_factory`` (an async callable
        returning a session-like object or None). The winsdk path below is
        marked for live verification (TASK-011 M6)."""
        session: Any
        if self._smtc_factory is not None:
            session = await self._smtc_factory()
        else:
            try:
                from winsdk.windows.media.control import (  # noqa: PLC0415
                    GlobalSystemMediaTransportControlsSessionManager as _Gsm,
                )
            except ImportError:
                return ActionResult(
                    ok=False,
                    error="winsdk is required for os_media_state: pip install winsdk (opt-in)",
                )
            manager = await _Gsm.request_async()
            raw = manager.get_current_session()
            if raw is None:
                session = None
            else:
                props = await raw.try_get_media_properties_async()
                session = {
                    "title": props.title,
                    "artist": props.artist,
                    "status": str(raw.playback_info.playback_status),
                }
        if session is None:
            return ActionResult(ok=True, output="no active media session")
        fields: dict[str, Any] = (
            session
            if isinstance(session, dict)
            else {k: getattr(session, k, "") for k in ("title", "artist", "status")}
        )
        title = str(fields.get("title", "") or "")
        artist = str(fields.get("artist", "") or "")
        status = str(fields.get("status", "?"))
        return ActionResult(ok=True, output=f"{status}: '{title}' by {artist}".strip())


# Re-exported for tests / callers that want the canonical media-key names.
MEDIA_KEYS: frozenset[str] = frozenset(_MEDIA_VK)

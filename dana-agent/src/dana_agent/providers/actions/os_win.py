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
    {"os_launch_app", "os_focus_window", "os_media_key", "os_clipboard_get", "os_clipboard_set"}
)


@register("actions", "os")
class WindowsOSActions(ActionProvider):
    """Bounded, API-first Windows control. Off unless ``os_actions_enabled``."""

    def __init__(self, cfg: ActionsConfig) -> None:
        self._cfg = cfg
        self._enabled = cfg.os_actions_enabled
        self._allowed_apps = frozenset(a.lower().strip() for a in cfg.allowed_apps if a.strip())

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
        if tool in ("os_focus_window", "os_media_key"):
            return Reversibility.READ_ONLY  # transient control; no lasting state
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
            # os_clipboard_set
            return self._clipboard_set(str(args.get("text", "")))
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
        import ctypes  # noqa: PLC0415
        from ctypes import wintypes  # noqa: PLC0415

        user32 = ctypes.windll.user32  # Windows-only; resolved inside the method
        # Declare HWND params as pointer-sized so 64-bit window handles are not
        # truncated to 32 bits (would target the wrong window / fail).
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
        if not match:
            return ActionResult(ok=False, error=f"no window matching {title_substr!r}")
        hwnd = match[0]
        user32.ShowWindow(hwnd, _SW_RESTORE)  # un-minimize first
        user32.SetForegroundWindow(hwnd)
        return ActionResult(ok=True, output=f"focused hwnd {hwnd}")

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


# Re-exported for tests / callers that want the canonical media-key names.
MEDIA_KEYS: frozenset[str] = frozenset(_MEDIA_VK)

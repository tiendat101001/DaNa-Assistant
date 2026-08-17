"""System tray icon (pystray) running alongside the asyncio agent.

``pystray.Icon.run()`` is blocking and wants a thread with a message pump, while
the agent owns the asyncio loop on the main thread. Resolution (research §C.2):
run the tray on its own **daemon thread** and marshal every menu action back onto
the agent loop with :func:`asyncio.run_coroutine_threadsafe` /
``loop.call_soon_threadsafe`` — the tray thread never touches asyncio state
directly.

Menu:
- **Mute / Unmute mic** — a dynamic label; drives the controller's REAL hard mute
  (the ``MuteProcessor`` drops capture at the source), not a soft flag.
- **Open dashboard** — ``webbrowser.open`` on the loopback control URL.
- **Quit** — stops the icon and asks the loop to stop.

``pystray`` + ``pillow`` are the ``control`` extra and are imported lazily inside
:meth:`start`, so importing this module offline never fails and the tray logic is
unit-testable with a fake icon/controller (no real tray in CI).
"""

from __future__ import annotations

import asyncio
import threading
import webbrowser
from typing import TYPE_CHECKING, Any

from dana_agent.observability import get_logger

if TYPE_CHECKING:
    from dana_agent.control.controller import Controller

log = get_logger(__name__)


class Tray:
    """A tray icon bound to a running agent loop + controller.

    The controller must expose the async ``mute()`` / ``unmute()`` /
    ``is_muted()`` surface of :class:`~dana_agent.control.controller.Controller`.
    """

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        controller: Controller,
        dashboard_url: str,
    ) -> None:
        self._loop = loop
        self._controller = controller
        self._url = dashboard_url
        self._icon: Any = None
        self._thread: threading.Thread | None = None

    # --- lifecycle ------------------------------------------------------

    def _build_icon(self) -> Any:
        """Construct the pystray Icon (lazy imports). Split out so tests can drive
        the menu callbacks against a fake icon without a real tray backend."""
        try:
            import pystray  # noqa: PLC0415
            from PIL import Image  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "pystray and pillow are required for the tray: "
                "install dana-agent[control]"
            ) from exc

        image = Image.new("RGB", (64, 64), (30, 30, 30))  # placeholder glyph
        menu = pystray.Menu(
            pystray.MenuItem(self._mute_label, self._toggle_mute),
            pystray.MenuItem("Open dashboard", self._open_dashboard),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Quit", self._quit),
        )
        return pystray.Icon("dana", image, "DaNa", menu)

    def start(self) -> None:
        """Build the icon and run it on a daemon thread (returns immediately)."""
        self._icon = self._build_icon()
        self._thread = threading.Thread(
            target=self._icon.run, name="dana-tray", daemon=True
        )
        self._thread.start()
        log.info("tray_started", url=self._url)

    def stop(self) -> None:
        """Stop the tray icon and wait for the tray thread to exit."""
        if self._icon is not None:
            self._icon.stop()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    # --- menu model / callbacks (run on the TRAY thread) ----------------

    def _mute_label(self, _item: Any = None) -> str:
        """Dynamic menu label reflecting live mute state."""
        return "Unmute mic" if self._controller.is_muted() else "Mute mic"

    def _toggle_mute(self, _icon: Any = None, _item: Any = None) -> None:
        # Callback fires on the tray thread → marshal the coroutine to the loop.
        # update_menu() is NOT called here because is_muted() still returns the
        # pre-toggle value at this point. The MIC_STATE WS event (emitted by the
        # mute coroutine) drives any dashboard update; the tray label refreshes
        # automatically the next time the menu is opened.
        coro = (
            self._controller.unmute()
            if self._controller.is_muted()
            else self._controller.mute()
        )
        asyncio.run_coroutine_threadsafe(coro, self._loop)

    def _open_dashboard(self, _icon: Any = None, _item: Any = None) -> None:
        webbrowser.open(self._url)

    def _quit(self, _icon: Any = None, _item: Any = None) -> None:
        if self._icon is not None:
            self._icon.stop()
        self._loop.call_soon_threadsafe(self._loop.stop)

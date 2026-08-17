"""Windows platform helpers (autostart, and future OS-integration bits).

Everything here shells out to Windows-only tooling (``schtasks``) and is reached
only from the explicit ``python -m dana_agent autostart …`` subcommand — never
imported on the agent's hot path and never executed by the test suite (tests
monkeypatch ``subprocess.run``). Importing the module is side-effect-free on any
platform.
"""

from dana_agent.platform_win.autostart import (
    TASK_NAME,
    install_autostart,
    is_autostart_installed,
    uninstall_autostart,
)

__all__ = [
    "TASK_NAME",
    "install_autostart",
    "is_autostart_installed",
    "uninstall_autostart",
]

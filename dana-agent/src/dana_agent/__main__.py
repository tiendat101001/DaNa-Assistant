"""CLI entrypoint: ``python -m dana_agent``.

Loads the shared settings, configures logging, validates + assembles the
provider stack, then starts the Pipecat conversation session.

Flags (default "run" command)
-----------------------------
--bypass-wake     Skip wake-word detection (forward every audio frame to STT).
                  Useful during development without a microphone / wake engine.
--validate-only   Build and validate the stack, then exit (no pipeline).
--control         Start the loopback control API + dashboard (and tray) alongside
                  the session (also enabled via DANA_CONTROL__ENABLED=true). Off by
                  default: headless runs never import fastapi/uvicorn.

Subcommands
-----------
secrets set|get|delete NAME   Manage API keys in the OS credential vault (keyring).
                              ``set`` reads the value from a prompt / piped stdin —
                              never from argv — so it is not echoed to shell history.
autostart install|uninstall|status
                              Manage the Windows Task Scheduler logon task. Performs
                              REAL registration; run explicitly on the target box.
"""

from __future__ import annotations

import argparse
import asyncio
import signal
import sys
from typing import TYPE_CHECKING

from structlog.stdlib import BoundLogger

from dana_agent.app import build_stack
from dana_agent.config import get_settings
from dana_agent.conversation import ConversationSession
from dana_agent.errors import ConfigError
from dana_agent.observability import configure_logging, get_logger

if TYPE_CHECKING:
    from dana_agent.app import AssembledStack
    from dana_agent.config import AppConfig
    from dana_agent.control.controller import Controller
    from dana_agent.control.tray import Tray


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="dana_agent", description="DaNa voice assistant")
    p.add_argument(
        "--bypass-wake",
        action="store_true",
        help="skip wake-word gate (send all mic audio straight to STT)",
    )
    p.add_argument(
        "--validate-only",
        action="store_true",
        help="validate config + providers then exit without running",
    )
    p.add_argument(
        "--control",
        action="store_true",
        help="start the loopback control API + dashboard alongside the session",
    )
    sub = p.add_subparsers(dest="command")

    secrets = sub.add_parser("secrets", help="manage API keys in the OS credential vault")
    secrets.add_argument("action", choices=["set", "get", "delete"])
    secrets.add_argument("name", help="secret name, e.g. ASSEMBLYAI_API_KEY")

    autostart = sub.add_parser("autostart", help="manage the Windows logon autostart task")
    autostart.add_argument("action", choices=["install", "uninstall", "status"])

    return p.parse_args(argv)


async def _run(args: argparse.Namespace) -> int:
    cfg = get_settings()
    configure_logging(cfg.log_level)
    log = get_logger("dana_agent")

    try:
        stack = await build_stack(cfg)
    except ConfigError as exc:
        log.error("config_invalid", error=str(exc))
        return 2

    if args.validate_only:
        log.info("ready", note="provider stack validated — exiting (--validate-only)")
        await stack.aclose()
        return 0

    if args.control or cfg.control.enabled:
        return await _run_with_control(cfg, stack, bypass_wake=args.bypass_wake, log=log)

    return await _run_session(cfg, stack, bypass_wake=args.bypass_wake, log=log)


def _install_stop_handlers(stop_event: asyncio.Event) -> None:
    """POSIX signal handlers set the stop event; Windows relies on KeyboardInterrupt."""
    if sys.platform == "win32":
        return
    loop = asyncio.get_running_loop()

    def _handle_signal() -> None:
        stop_event.set()

    loop.add_signal_handler(signal.SIGINT, _handle_signal)
    loop.add_signal_handler(signal.SIGTERM, _handle_signal)


async def _run_session(
    cfg: AppConfig, stack: AssembledStack, *, bypass_wake: bool, log: BoundLogger
) -> int:
    """Headless path: run the pipeline session with no control plane."""
    session = ConversationSession(cfg, stack, bypass_wake_gate=bypass_wake)
    stop_event = asyncio.Event()
    _install_stop_handlers(stop_event)

    run_task = asyncio.create_task(session.run())
    stop_task = asyncio.create_task(stop_event.wait())
    done, pending = await asyncio.wait([run_task, stop_task], return_when=asyncio.FIRST_COMPLETED)
    for task in pending:
        task.cancel()

    await session.stop()
    await stack.aclose()

    for task in done:
        exc = task.exception() if not task.cancelled() else None
        if exc is not None:
            log.error("session_error", exc_info=exc)
            return 1
    return 0


async def _run_with_control(
    cfg: AppConfig, stack: AssembledStack, *, bypass_wake: bool, log: BoundLogger
) -> int:
    """Control path: start the loopback API + dashboard, then run a session driven
    by the controller (so mic mute is a hard gate in the pipeline). All fastapi /
    uvicorn imports are lazy and confined to this branch."""
    from dana_agent.control import AgentController, EventBus  # noqa: PLC0415
    from dana_agent.control.app import create_control_app  # noqa: PLC0415
    from dana_agent.control.auth import load_or_create_token  # noqa: PLC0415
    from dana_agent.control.server import ControlServer  # noqa: PLC0415

    bus = EventBus()
    controller = AgentController(cfg, stack, bus, bypass_wake_gate=bypass_wake)
    token = load_or_create_token(cfg.control.token_path)
    app = create_control_app(cfg, controller, bus, control_token=token)
    server = ControlServer(app, cfg.control)

    stop_event = asyncio.Event()
    _install_stop_handlers(stop_event)

    await server.start()
    await controller.start_session()
    url = f"http://{cfg.control.host}:{cfg.control.port}"
    log.info("control_ready", url=url, token_path=cfg.control.token_path)

    tray = _maybe_start_tray(controller, url, log=log)

    await stop_event.wait()

    if tray is not None:
        tray.stop()
    await controller.stop_session()
    await server.stop()
    await stack.aclose()
    return 0


def _maybe_start_tray(controller: Controller, url: str, *, log: BoundLogger) -> Tray | None:
    """Start the system tray when its deps are present; degrade gracefully if not.

    The tray is part of the control experience, but ``pystray``/``pillow`` are the
    ``control`` extra and may be absent (e.g. a server-style run). A missing tray
    must never take down the control API — we log and continue headless."""
    from dana_agent.control.tray import Tray  # noqa: PLC0415

    tray = Tray(asyncio.get_running_loop(), controller, url)
    try:
        tray.start()
    except ImportError as exc:
        log.warning("tray_unavailable", error=str(exc))
        return None
    return tray


# --------------------------------------------------------------------------
# secrets subcommand
# --------------------------------------------------------------------------


def _read_secret_value(name: str) -> str:
    """Read a secret value without echoing it into shell history.

    Piped stdin (``echo $KEY | python -m dana_agent secrets set NAME``) is read
    directly; an interactive TTY uses a no-echo password prompt.
    """
    if not sys.stdin.isatty():
        return sys.stdin.readline().rstrip("\n")
    import getpass  # noqa: PLC0415

    return getpass.getpass(f"Value for {name}: ")


def _handle_secrets(args: argparse.Namespace) -> int:
    from dana_agent.secrets import (  # noqa: PLC0415
        KeyringUnavailableError,
        delete_secret,
        get_secret,
        set_secret,
    )

    try:
        if args.action == "set":
            value = _read_secret_value(args.name)
            if not value:
                print("no value provided; aborted", file=sys.stderr)
                return 2
            set_secret(args.name, value)
            print(f"stored {args.name} in the OS credential vault")
            return 0
        if args.action == "get":
            stored = get_secret(args.name)
            if stored is None:
                print(f"{args.name} is not set in the vault", file=sys.stderr)
                return 1
            print(stored)
            return 0
        # delete
        removed = delete_secret(args.name)
        print(f"deleted {args.name}" if removed else f"{args.name} was not set")
        return 0
    except KeyringUnavailableError as exc:
        print(str(exc), file=sys.stderr)
        return 2


# --------------------------------------------------------------------------
# autostart subcommand
# --------------------------------------------------------------------------


def _handle_autostart(args: argparse.Namespace) -> int:
    from dana_agent.platform_win.autostart import (  # noqa: PLC0415
        install_autostart,
        is_autostart_installed,
        uninstall_autostart,
    )

    if args.action == "status":
        installed = is_autostart_installed()
        print("installed" if installed else "not installed")
        return 0 if installed else 1

    result = install_autostart() if args.action == "install" else uninstall_autostart()
    if result.ok:
        print(f"autostart {args.action}: ok")
        return 0
    print(f"autostart {args.action} failed (exit {result.returncode})", file=sys.stderr)
    if result.stderr.strip():
        print(result.stderr.strip(), file=sys.stderr)
    return 1


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.command == "secrets":
        return _handle_secrets(args)
    if args.command == "autostart":
        return _handle_autostart(args)
    try:
        return asyncio.run(_run(args))
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())

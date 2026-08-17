"""CLI entrypoint: ``python -m dana_agent``.

Loads the shared settings, configures logging, validates + assembles the
provider stack, then starts the Pipecat conversation session.

Flags
-----
--bypass-wake     Skip wake-word detection (forward every audio frame to STT).
                  Useful during development without a microphone / wake engine.
--validate-only   Build and validate the stack, then exit (no pipeline).
"""

from __future__ import annotations

import argparse
import asyncio
import signal
import sys

from dana_agent.app import build_stack
from dana_agent.config import get_settings
from dana_agent.conversation import ConversationSession
from dana_agent.errors import ConfigError
from dana_agent.observability import configure_logging, get_logger


def _parse_args() -> argparse.Namespace:
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
    return p.parse_args()


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

    session = ConversationSession(cfg, stack, bypass_wake_gate=args.bypass_wake)

    stop_event = asyncio.Event()

    # loop.add_signal_handler is POSIX-only; Windows uses KeyboardInterrupt instead.
    if sys.platform != "win32":
        loop = asyncio.get_running_loop()

        def _handle_signal() -> None:
            log.info("shutting_down", cause="signal")
            stop_event.set()

        loop.add_signal_handler(signal.SIGINT, _handle_signal)
        loop.add_signal_handler(signal.SIGTERM, _handle_signal)

    run_task = asyncio.create_task(session.run())
    stop_task = asyncio.create_task(stop_event.wait())

    done, pending = await asyncio.wait(
        [run_task, stop_task], return_when=asyncio.FIRST_COMPLETED
    )

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


def main() -> int:
    args = _parse_args()
    try:
        return asyncio.run(_run(args))
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())

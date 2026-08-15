"""CLI entrypoint: ``python -m dana_agent``.

Phase 1–4: load the shared settings, configure logging, validate + assemble the
provider stack (incl. the brain and its checkpointer), and report readiness. The
always-on run loop (wake gate -> session -> brain -> actions, plus the control
API) is wired in later phases.
"""

from __future__ import annotations

import asyncio
import sys

from dana_agent.app import build_stack
from dana_agent.config import get_settings
from dana_agent.errors import ConfigError
from dana_agent.observability import configure_logging, get_logger


async def _run() -> int:
    cfg = get_settings()
    configure_logging(cfg.log_level)
    log = get_logger("dana_agent")
    try:
        stack = await build_stack(cfg)
    except ConfigError as exc:
        log.error("config_invalid", error=str(exc))
        return 2
    await stack.aclose()
    log.info("ready", note="provider stack + wake gate validated and assembled")
    return 0


def main() -> int:
    return asyncio.run(_run())


if __name__ == "__main__":
    sys.exit(main())

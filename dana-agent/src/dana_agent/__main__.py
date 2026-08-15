"""CLI entrypoint: ``python -m dana_agent``.

Phase 1: load config, configure logging, validate + assemble the provider
stack, and report readiness. The always-on run loop (wake gate -> session ->
brain -> actions, plus the control API) is wired in later phases.
"""

from __future__ import annotations

import sys

from dana_agent.app import build_stack
from dana_agent.config import AppConfig
from dana_agent.errors import ConfigError
from dana_agent.observability import configure_logging, get_logger


def main() -> int:
    cfg = AppConfig()
    configure_logging(cfg.log_level)
    log = get_logger("dana_agent")
    try:
        build_stack(cfg)
    except ConfigError as exc:
        log.error("config_invalid", error=str(exc))
        return 2
    log.info("ready", note="provider stack + wake gate validated and assembled")
    return 0


if __name__ == "__main__":
    sys.exit(main())

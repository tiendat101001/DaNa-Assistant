"""Structured logging setup (structlog).

Log key/value events, not f-string prose. JSON in production (searchable),
pretty console in dev. Never log secrets or raw audio/PII — redact.
"""

from __future__ import annotations

import logging
import sys
from typing import Any, cast

import structlog


def configure_logging(level: str = "INFO", *, json_logs: bool | None = None) -> None:
    """Configure structlog once at startup.

    ``json_logs`` defaults to True when stdout is not a TTY (i.e. in production
    / when piped), and pretty console rendering when attached to a terminal.
    """
    if json_logs is None:
        json_logs = not sys.stdout.isatty()

    shared: list[structlog.typing.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]
    renderer: structlog.typing.Processor = (
        structlog.processors.JSONRenderer() if json_logs else structlog.dev.ConsoleRenderer()
    )

    structlog.configure(
        processors=[*shared, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelNamesMapping().get(level.upper(), logging.INFO)
        ),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(*args: Any, **kwargs: Any) -> structlog.stdlib.BoundLogger:
    """Return a bound structlog logger."""
    return cast(structlog.stdlib.BoundLogger, structlog.get_logger(*args, **kwargs))

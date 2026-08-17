"""Structured logging setup (structlog + loguru bridge).

Log key/value events, not f-string prose. JSON in production (searchable),
pretty console in dev. Never log secrets or raw audio/PII — redact.

Output:
  console  — pretty (TTY) or JSON (piped/redirected)
  file     — logs/YYYY-MM-DD.log, always JSON, one line per event
             includes both structlog (DaNa) and loguru (Pipecat) lines

File lives at logs/ relative to the working directory when the agent starts.
When launched from dana-agent/ (the normal case):
  dana-agent/logs/YYYY-MM-DD.log

Tail live:  Get-Content dana-agent\\logs\\2026-08-17.log -Wait
"""

from __future__ import annotations

import datetime
import logging
import sys
from pathlib import Path
from typing import Any, cast

import structlog

# logs/ relative to CWD — always correct when invoked from dana-agent/
_LOG_DIR = Path.cwd() / "logs"


def _log_file() -> Path:
    """Return today's log file, creating the directory if needed."""
    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    return _LOG_DIR / f"{datetime.date.today().isoformat()}.log"


def configure_logging(level: str = "INFO", *, json_logs: bool | None = None) -> None:
    """Configure structlog + loguru to write to console and a daily log file.

    ``json_logs`` controls the *console* renderer; the file is always JSON.
    Defaults to pretty console when stdout is a TTY, JSON when piped.
    """
    if json_logs is None:
        json_logs = not sys.stdout.isatty()

    int_level = logging.getLevelNamesMapping().get(level.upper(), logging.INFO)
    log_path = _log_file()

    # ------------------------------------------------------------------
    # Shared pre-processors (run before any renderer)
    # ------------------------------------------------------------------
    shared: list[structlog.typing.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
    ]

    # ------------------------------------------------------------------
    # Console handler
    # ------------------------------------------------------------------
    console_renderer: structlog.typing.Processor = (
        structlog.processors.JSONRenderer()
        if json_logs
        else structlog.dev.ConsoleRenderer()
    )
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(int_level)
    console_handler.setFormatter(
        structlog.stdlib.ProcessorFormatter(processor=console_renderer)
    )

    # ------------------------------------------------------------------
    # File handler — always JSON
    # ------------------------------------------------------------------
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(int_level)
    file_handler.setFormatter(
        structlog.stdlib.ProcessorFormatter(processor=structlog.processors.JSONRenderer())
    )

    # ------------------------------------------------------------------
    # Root stdlib logger — both handlers attached once
    # ------------------------------------------------------------------
    root = logging.getLogger()
    root.setLevel(int_level)
    if not any(isinstance(h, logging.FileHandler) for h in root.handlers):
        root.addHandler(console_handler)
        root.addHandler(file_handler)

    # ------------------------------------------------------------------
    # structlog — route through stdlib so both handlers fire
    # ------------------------------------------------------------------
    structlog.configure(
        processors=shared,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # ------------------------------------------------------------------
    # Loguru bridge — capture Pipecat's loguru output to the same file
    # ------------------------------------------------------------------
    _setup_loguru_file_sink(log_path, int_level)

    get_logger("dana_agent").info("log_file_opened", path=str(log_path))


def _setup_loguru_file_sink(log_path: Path, int_level: int) -> None:
    """Add a loguru sink that writes Pipecat log lines to the daily log file."""
    try:
        from loguru import logger  # noqa: PLC0415

        level_name = logging.getLevelName(int_level)

        logger.add(
            str(log_path),
            level=level_name,
            format="{time:YYYY-MM-DDTHH:mm:ss.SSSSSSZ} [{level}] {name}: {message}",
            encoding="utf-8",
            mode="a",
            enqueue=True,  # async-safe
        )
    except ImportError:
        pass  # loguru not installed — Pipecat lines won't reach the file


def get_logger(*args: Any, **kwargs: Any) -> structlog.stdlib.BoundLogger:
    """Return a bound structlog logger."""
    return cast(structlog.stdlib.BoundLogger, structlog.get_logger(*args, **kwargs))

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
from collections.abc import Callable, MutableMapping
from pathlib import Path
from typing import Any, cast

import structlog

# logs/ relative to CWD — always correct when invoked from dana-agent/
_LOG_DIR = Path.cwd() / "logs"

# Optional structlog->EventBus tap. The control plane installs a sink here so
# structlog-emitted `latency` and `action_audit` events reach the WS stream
# WITHOUT the pipeline ever importing the control package. None by default →
# zero overhead and no behaviour change when control is off.
_EVENT_SINK: Callable[[MutableMapping[str, Any]], None] | None = None


def set_event_sink(sink: Callable[[MutableMapping[str, Any]], None] | None) -> None:
    """Install (or clear) the structlog event tap used by the control plane."""
    global _EVENT_SINK
    _EVENT_SINK = sink


def _forward_events(
    _logger: object, _method: str, event_dict: MutableMapping[str, Any]
) -> MutableMapping[str, Any]:
    """structlog processor: mirror events to the control EventBus, never mutating
    or dropping the log record. Best-effort — a sink error must not break logging."""
    sink = _EVENT_SINK
    if sink is not None:
        try:
            sink(event_dict)
        except Exception:  # noqa: BLE001, S110 — logging must never crash on the tap
            pass
    return event_dict


# Events whose ``text`` field carries user/assistant transcript PII. The live
# control bus receives the full text (``_forward_events`` runs in the shared
# chain BEFORE the per-sink renderers below, feeding the loopback dashboard), but
# the durable log outputs must NOT persist it: this module's contract and
# docs/PLAN.md §7/§8 say never write PII/transcripts to the logs. So the rendered
# console/file sinks redact ``text`` for these events.
_PII_TEXT_EVENTS = frozenset({"brain_turn_start", "transcript_final", "assistant_turn"})


def _redact_pii(event_dict: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
    """Return the record with transcript/assistant ``text`` redacted. Never
    mutates the shared dict — the bus already saw the original upstream."""
    if event_dict.get("event") in _PII_TEXT_EVENTS and event_dict.get("text"):
        redacted = dict(event_dict)
        redacted["text"] = "[redacted]"
        return redacted
    return event_dict


def _redacting_renderer(inner: structlog.typing.Processor) -> structlog.typing.Processor:
    """Wrap a terminal renderer so it redacts transcript PII first. Applied to the
    console and file sinks; the EventBus tap upstream keeps the full text."""

    def _render(logger: object, method: str, event_dict: MutableMapping[str, Any]) -> Any:
        return inner(logger, method, _redact_pii(event_dict))

    return _render


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
        _forward_events,  # tap events to the control EventBus (no-op if unset)
        structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
    ]

    # ------------------------------------------------------------------
    # Console handler
    # ------------------------------------------------------------------
    console_renderer: structlog.typing.Processor = (
        structlog.processors.JSONRenderer() if json_logs else structlog.dev.ConsoleRenderer()
    )
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(int_level)
    console_handler.setFormatter(
        structlog.stdlib.ProcessorFormatter(processor=_redacting_renderer(console_renderer))
    )

    # ------------------------------------------------------------------
    # File handler — always JSON
    # ------------------------------------------------------------------
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(int_level)
    file_handler.setFormatter(
        structlog.stdlib.ProcessorFormatter(
            processor=_redacting_renderer(structlog.processors.JSONRenderer())
        )
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

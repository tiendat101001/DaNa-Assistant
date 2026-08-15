"""Open the brain's conversation-memory checkpointer per config.

- ``memory`` -> LangGraph ``InMemorySaver`` (in-process, non-persistent).
- ``sqlite`` -> ``AsyncSqliteSaver`` over an aiosqlite connection. A file path
  gives memory that survives restarts; the literal ``":memory:"`` gives a SQLite
  in-memory DB (ephemeral, but real SQLite).

Returns the checkpointer plus an async ``close`` callback the caller must invoke
on shutdown (it closes the SQLite connection; a no-op for the memory backend).
Opening SQLite is genuinely async (connect + create tables), which is why this
is a coroutine and why the composition root has an async phase.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from langgraph.checkpoint.memory import InMemorySaver

from dana_agent.config import MemoryConfig
from dana_agent.observability import get_logger

log = get_logger(__name__)

Checkpointer = Any  # a LangGraph BaseCheckpointSaver (Any at our type boundary)
CloseFn = Callable[[], Awaitable[None]]

_SQLITE_MEMORY = ":memory:"


async def _noop() -> None:
    return None


async def open_checkpointer(cfg: MemoryConfig) -> tuple[Checkpointer, CloseFn]:
    if cfg.backend == "memory":
        return InMemorySaver(), _noop

    import aiosqlite
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    if cfg.sqlite_path != _SQLITE_MEMORY:
        # One-time, fast, local mkdir at startup — not worth an async-fs dep.
        Path(cfg.sqlite_path).expanduser().parent.mkdir(parents=True, exist_ok=True)  # noqa: ASYNC240

    conn = await aiosqlite.connect(cfg.sqlite_path)
    saver = AsyncSqliteSaver(conn)
    await saver.setup()

    async def close() -> None:
        await conn.close()

    log.info("checkpointer_opened", backend="sqlite", path=cfg.sqlite_path)
    return saver, close

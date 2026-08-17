"""Run the control app inside the agent's own event loop.

The agent already owns an asyncio loop (the pipeline runs on it), so uvicorn must
NOT own the loop. We construct ``uvicorn.Server`` and run its ``serve()``
coroutine as a task, polling ``server.started`` to know when the socket is up.
Binding is loopback-only (``cfg.control.host`` defaults to ``127.0.0.1``).

uvicorn is imported lazily so importing this module offline (without the
``control`` extra) never fails — only ``start()`` requires it.
"""

from __future__ import annotations

import asyncio
from typing import Any

from dana_agent.config import ControlConfig
from dana_agent.observability import get_logger

log = get_logger(__name__)

_STARTUP_POLL_S = 0.02
_STARTUP_TIMEOUT_S = 10.0


class ControlServer:
    """Owns the uvicorn Server task for a control :class:`FastAPI` app."""

    def __init__(self, app: Any, cfg: ControlConfig) -> None:
        self._app = app
        self._cfg = cfg
        self._server: Any = None
        self._task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        """Start serving in the background; return once the socket is accepting."""
        try:
            import uvicorn  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - requires the control extra
            raise ImportError(
                "uvicorn is required to run the control server: install dana-agent[control]"
            ) from exc

        config = uvicorn.Config(
            self._app,
            host=self._cfg.host,  # 127.0.0.1 — loopback only
            port=self._cfg.port,
            log_config=None,  # reuse the agent's structlog/loguru config
            lifespan="on",
        )
        self._server = uvicorn.Server(config)
        self._task = asyncio.create_task(self._server.serve())

        waited = 0.0
        while not self._server.started:
            if self._task.done():  # serve() failed before signalling started
                await self._task
                return
            await asyncio.sleep(_STARTUP_POLL_S)
            waited += _STARTUP_POLL_S
            if waited >= _STARTUP_TIMEOUT_S:
                raise TimeoutError("control server did not start within timeout")
        log.info("control_server_listening", host=self._cfg.host, port=self._cfg.port)

    async def stop(self) -> None:
        """Signal uvicorn to shut down and await the serve task."""
        if self._server is not None:
            self._server.should_exit = True
        if self._task is not None:
            try:
                await self._task
            except asyncio.CancelledError:  # pragma: no cover
                pass
        self._server = None
        self._task = None
        log.info("control_server_stopped")

"""FastAPI application factory for the control plane.

``create_control_app`` wires the router, the controller, the event bus, and the
bearer token together and (in production) mounts the built ``dana-ui`` bundle at
``/``. It uses a ``lifespan`` async context manager (not the deprecated
``on_event`` hooks) so the log→bus bridge is installed while the app is live and
torn down cleanly on shutdown.

This module imports FastAPI at top level, so it must only be imported when the
control plane is actually enabled — the agent's headless path never touches it.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from dana_agent.config import AppConfig
from dana_agent.control.controller import Controller
from dana_agent.control.events import EventBus
from dana_agent.control.log_bridge import install_log_bridge, uninstall_log_bridge
from dana_agent.control.routes import api
from dana_agent.observability import get_logger

log = get_logger(__name__)


def create_control_app(
    cfg: AppConfig,
    controller: Controller,
    bus: EventBus,
    *,
    control_token: str,
    serve_ui: bool = True,
) -> FastAPI:
    """Build the loopback control app. State is stashed on ``app.state`` so the
    stateless router can reach the controller/bus/token."""

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        # Tap structlog latency/audit events into the bus for the WS stream.
        install_log_bridge(bus)
        log.info("control_plane_up", host=cfg.control.host, port=cfg.control.port)
        try:
            yield
        finally:
            uninstall_log_bridge()
            log.info("control_plane_down")

    app = FastAPI(title="DaNa Control API", version="1", lifespan=lifespan)
    app.state.controller = controller
    app.state.bus = bus
    app.state.control_token = control_token

    app.include_router(api)

    if serve_ui:
        dist = Path(cfg.control.ui_dist_dir)
        if dist.is_dir():
            # html=True serves index.html for SPA routes; mounted last so it never
            # shadows /api/* or /health.
            app.mount("/", StaticFiles(directory=str(dist), html=True), name="ui")
            log.info("control_ui_mounted", dist=str(dist))
        else:
            log.info("control_ui_absent", dist=str(dist), hint="run `npm run build` in dana-ui/")

    return app

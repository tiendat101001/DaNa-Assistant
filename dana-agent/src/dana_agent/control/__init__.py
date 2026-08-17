"""Local control plane (TASK-007 / docs/PLAN.md §5, ADR-0002).

A FastAPI REST + WebSocket app mounted in the agent's own asyncio process and
bound to loopback (127.0.0.1). It exposes config CRUD, provider catalog, session
and mic control, pending-confirmation resolution, the audit trail, and a live
event stream the dashboard renders.

The core pipeline stays decoupled from FastAPI: the API talks to a ``Controller``
abstraction and pushes/receives events through an in-process ``EventBus``.
``conversation/*`` never imports this package.

Everything here is import-guarded so the agent runs headless with the control
plane OFF and no fastapi/uvicorn import unless it is explicitly enabled.
"""

from dana_agent.control.controller import (
    AgentController,
    ConfirmationInfo,
    Controller,
    StatusInfo,
)
from dana_agent.control.events import Envelope, EventBus, EventType

__all__ = [
    "AgentController",
    "ConfirmationInfo",
    "Controller",
    "Envelope",
    "EventBus",
    "EventType",
    "StatusInfo",
]

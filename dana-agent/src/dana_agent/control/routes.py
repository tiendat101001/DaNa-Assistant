"""REST + WebSocket routes for the control plane (docs/PLAN.md §5).

Everything lives under ``/api/v1``. Every route except ``/health`` requires the
bearer token (via the ``require_auth`` dependency). Secrets are never returned:
config responses run through ``model_dump(mode="json")`` where ``SecretStr``
renders masked.

The router is contract-only — it holds no state. The controller, event bus, and
token are read from ``request.app.state`` (populated by ``create_control_app``),
so the same router serves the real agent and the test fakes unchanged.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from fastapi import APIRouter, Body, Depends, HTTPException, Query, Request, WebSocket
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from starlette.websockets import WebSocketDisconnect

from dana_agent.config import AppConfig
from dana_agent.control.auth import token_matches
from dana_agent.control.catalog import build_provider_catalog
from dana_agent.control.controller import Controller
from dana_agent.control.events import EventBus
from dana_agent.observability import get_logger

log = get_logger(__name__)

api = APIRouter(prefix="/api/v1")


# --- dependencies --------------------------------------------------------


def get_controller(request: Request) -> Controller:
    controller: Controller = request.app.state.controller
    return controller


def get_bus(request: Request) -> EventBus:
    bus: EventBus = request.app.state.bus
    return bus


def require_auth(request: Request) -> None:
    """Reject any request without the correct ``Authorization: Bearer`` token."""
    expected: str = request.app.state.control_token
    if not token_matches(expected, request.headers.get("authorization")):
        raise HTTPException(status_code=401, detail="missing or invalid bearer token")


_auth = Depends(require_auth)


# --- health (no auth) ----------------------------------------------------


@api.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


# --- config --------------------------------------------------------------


@api.get("/config", dependencies=[_auth])
def get_config(controller: Controller = Depends(get_controller)) -> dict[str, Any]:
    # mode="json" masks SecretStr -> "**********"; no secret ever leaves here.
    return controller.config().model_dump(mode="json")


@api.get("/config/schema", dependencies=[_auth])
def get_config_schema() -> dict[str, Any]:
    return AppConfig.model_json_schema()


@api.put("/config", dependencies=[_auth], response_model=None)
def put_config(
    patch: Mapping[str, Any] = Body(...),
    controller: Controller = Depends(get_controller),
) -> JSONResponse | dict[str, Any]:
    """Validate a (partial or full) config patch, hot-apply the safe fields, and
    label every change applied|restart_required. Invalid → 422, no mutation."""
    current = controller.config()
    merged = _deep_merge(current.model_dump(), dict(patch))
    try:
        new_config = AppConfig.model_validate(merged)
    except ValidationError as exc:
        return JSONResponse(status_code=422, content={"detail": exc.errors(include_url=False)})
    changes = controller.apply_config(new_config)
    return {"changes": [c.to_json() for c in changes]}


# --- providers -----------------------------------------------------------


@api.get("/providers", dependencies=[_auth])
def get_providers() -> dict[str, Any]:
    return build_provider_catalog()


# --- session -------------------------------------------------------------


@api.post("/session/start", dependencies=[_auth])
async def session_start(controller: Controller = Depends(get_controller)) -> dict[str, Any]:
    await controller.start_session()
    return {"status": "started", **controller.status().to_json()}


@api.post("/session/stop", dependencies=[_auth])
async def session_stop(controller: Controller = Depends(get_controller)) -> dict[str, Any]:
    await controller.stop_session()
    return {"status": "stopped", **controller.status().to_json()}


@api.post("/session/activate", dependencies=[_auth])
async def session_activate(controller: Controller = Depends(get_controller)) -> dict[str, Any]:
    """Trip the wake gate to ACTIVE — the Talk button. Like saying the wake
    phrase: the next spoken sentence goes straight to STT."""
    try:
        await controller.activate()
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"status": "active", **controller.status().to_json()}


@api.post("/session/turn", dependencies=[_auth])
async def session_turn(
    body: Mapping[str, Any] = Body(...),
    controller: Controller = Depends(get_controller),
) -> dict[str, Any]:
    """Run a typed user turn through the live pipeline: the brain answers and
    TTS speaks the reply, exactly like a spoken turn."""
    text = str(body.get("text", "")).strip()
    if not text:
        raise HTTPException(status_code=422, detail="text is required")
    try:
        await controller.submit_text(text)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"status": "submitted", "text": text}


@api.get("/status", dependencies=[_auth])
def status(controller: Controller = Depends(get_controller)) -> dict[str, Any]:
    return controller.status().to_json()


# --- mic -----------------------------------------------------------------


@api.post("/mic/mute", dependencies=[_auth])
async def mic_mute(controller: Controller = Depends(get_controller)) -> dict[str, bool]:
    await controller.mute()
    return {"muted": controller.is_muted()}


@api.post("/mic/unmute", dependencies=[_auth])
async def mic_unmute(controller: Controller = Depends(get_controller)) -> dict[str, bool]:
    await controller.unmute()
    return {"muted": controller.is_muted()}


# --- confirmations -------------------------------------------------------


@api.get("/confirmations", dependencies=[_auth])
def list_confirmations(controller: Controller = Depends(get_controller)) -> dict[str, Any]:
    return {"pending": [c.to_json() for c in controller.pending_confirmations()]}


@api.post("/confirmations/{confirmation_id}", dependencies=[_auth])
async def resolve_confirmation(
    confirmation_id: str,
    body: Mapping[str, Any] = Body(default={}),
    controller: Controller = Depends(get_controller),
) -> dict[str, Any]:
    approved = _decision(body)
    ok = await controller.resolve_confirmation(confirmation_id, approved=approved)
    if not ok:
        raise HTTPException(status_code=404, detail=f"unknown confirmation {confirmation_id!r}")
    return {"id": confirmation_id, "approved": approved}


# --- bot check -----------------------------------------------------------


@api.post("/session/bot-check/resolve", dependencies=[_auth])
async def resolve_bot_check(
    body: Mapping[str, Any] = Body(default={}),
    controller: Controller = Depends(get_controller),
) -> dict[str, Any]:
    """Signal that a bot check / CAPTCHA challenge has been resolved or bypassed."""
    bypassed = bool(body.get("bypassed", True))
    await controller.resolve_bot_check(bypassed=bypassed)
    return {"status": "resolved", "bypassed": bypassed}


# --- audit ---------------------------------------------------------------


@api.get("/audit", dependencies=[_auth])
def get_audit(
    since: float | None = Query(default=None),
    controller: Controller = Depends(get_controller),
) -> dict[str, Any]:
    return {"entries": controller.audit_entries(since=since)}


# --- websocket -----------------------------------------------------------


@api.websocket("/events")
async def events(websocket: WebSocket) -> None:
    """Stream server-push envelopes. Auth is by ``?token=`` (browsers can't set an
    Authorization header on a WebSocket) or the Authorization header if present."""
    expected: str = websocket.app.state.control_token
    token = websocket.query_params.get("token")
    header = websocket.headers.get("authorization")
    ok = token_matches(expected, header) or (
        token is not None and token_matches(expected, f"Bearer {token}")
    )
    if not ok:
        await websocket.close(code=4401)  # 4401: application-level "unauthorized"
        return

    bus: EventBus = websocket.app.state.bus
    await websocket.accept()
    async with bus.subscribe() as sub:
        try:
            async for envelope in sub:
                await websocket.send_json(envelope.to_json())
        except WebSocketDisconnect:
            return


# --- helpers -------------------------------------------------------------


def _deep_merge(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    """Recursively overlay ``patch`` onto ``base`` (patch wins at each leaf)."""
    out = dict(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _decision(body: Mapping[str, Any]) -> bool:
    """Interpret a confirmation body as approve/deny.

    Accepts ``{"approved": bool}`` or ``{"decision": "approve"|"deny"}``.
    Defaults to deny (fail-safe: never execute a destructive action on an
    ambiguous request)."""
    if "approved" in body:
        return bool(body["approved"])
    decision = str(body.get("decision", "")).lower()
    return decision in ("approve", "approved", "yes", "allow")

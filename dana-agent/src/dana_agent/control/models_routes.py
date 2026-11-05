"""Model registry REST routes (``/api/v1/models``).

The :class:`~dana_agent.models.ModelManager` lives at ``app.state.models``.
"""

from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, ValidationError

from dana_agent.config import AppConfig
from dana_agent.control.auth import token_matches
from dana_agent.control.controller import Controller
from dana_agent.control.routes import _deep_merge, require_auth
from dana_agent.models import (
    HubAuthError,
    HubUnavailableError,
    ModelBusyError,
    ModelInputError,
    ModelManager,
    ModelNotFoundError,
)

models_api = APIRouter(prefix="/api/v1/models")
_auth = Depends(require_auth)


def get_manager(request: Request) -> ModelManager:
    manager: ModelManager = request.app.state.models
    return manager


class PullBody(BaseModel):
    repo_id: str
    revision: str = "main"
    files: list[str] | None = None
    include: list[str] | None = None
    load: bool = False
    backend: Literal["cpu", "webgpu"] = "cpu"
    num_threads: int = 4


class LoadBody(BaseModel):
    repo_id: str
    backend: Literal["cpu", "webgpu"] = "cpu"
    num_threads: int = 4
    file: str | None = None


class RepoBody(BaseModel):
    repo_id: str


class AssignBody(BaseModel):
    role: Literal["stt", "tts"]
    repo_id: str
    backend: Literal["cpu", "webgpu"] | None = None
    num_threads: int | None = None


def _http(exc: Exception) -> HTTPException:
    if isinstance(exc, HubAuthError):
        return HTTPException(403, str(exc))
    if isinstance(exc, HubUnavailableError):
        return HTTPException(502, str(exc))
    if isinstance(exc, ModelNotFoundError):
        return HTTPException(404, str(exc))
    if isinstance(exc, ModelBusyError):
        return HTTPException(409, str(exc))
    return HTTPException(422, str(exc))


MODEL_ERRORS = (
    HubAuthError,
    HubUnavailableError,
    ModelBusyError,
    ModelInputError,
    ModelNotFoundError,
)


@models_api.get("/hub", dependencies=[_auth])
async def hub_list(
    search: str | None = None,
    task: str | None = None,
    refresh: bool = False,
    limit: int = Query(default=200, ge=1, le=1000),
    mgr: ModelManager = Depends(get_manager),
) -> dict[str, Any]:
    try:
        return await mgr.list_hub(search, task, refresh, limit)
    except MODEL_ERRORS as exc:
        raise _http(exc) from exc


@models_api.get("/hub/{repo_id:path}", dependencies=[_auth])
async def hub_detail(repo_id: str, mgr: ModelManager = Depends(get_manager)) -> dict[str, Any]:
    try:
        return await mgr.hub_detail(repo_id)
    except MODEL_ERRORS as exc:
        raise _http(exc) from exc


@models_api.get("/local", dependencies=[_auth])
def local_list(mgr: ModelManager = Depends(get_manager)) -> dict[str, Any]:
    return {"models_dir": mgr.cfg.models_dir, "models": mgr.local_models()}


@models_api.post("/pull", dependencies=[_auth], status_code=202)
async def pull(body: PullBody, mgr: ModelManager = Depends(get_manager)) -> dict[str, Any]:
    try:
        job = await mgr.start_pull(
            body.repo_id,
            body.revision,
            body.files,
            body.include,
            body.load,
            body.backend,
            body.num_threads,
        )
    except MODEL_ERRORS as exc:
        raise _http(exc) from exc
    return {"job": job.to_json()}


@models_api.get("/jobs", dependencies=[_auth])
def jobs_list(mgr: ModelManager = Depends(get_manager)) -> dict[str, Any]:
    return {"jobs": [j.to_json() for j in mgr.jobs()]}


@models_api.get("/jobs/{job_id}", dependencies=[_auth])
def job_get(job_id: str, mgr: ModelManager = Depends(get_manager)) -> dict[str, Any]:
    try:
        return mgr.get_job(job_id).to_json()
    except MODEL_ERRORS as exc:
        raise _http(exc) from exc


@models_api.delete("/jobs/{job_id}", dependencies=[_auth])
async def job_cancel(job_id: str, mgr: ModelManager = Depends(get_manager)) -> dict[str, Any]:
    try:
        return (await mgr.cancel_job(job_id)).to_json()
    except MODEL_ERRORS as exc:
        raise _http(exc) from exc


@models_api.post("/load", dependencies=[_auth])
async def load(body: LoadBody, mgr: ModelManager = Depends(get_manager)) -> dict[str, Any]:
    try:
        return await mgr.load(body.repo_id, body.backend, body.num_threads, body.file)
    except MODEL_ERRORS as exc:
        raise _http(exc) from exc


@models_api.post("/unload", dependencies=[_auth])
async def unload(body: RepoBody, mgr: ModelManager = Depends(get_manager)) -> dict[str, Any]:
    try:
        return await mgr.unload(body.repo_id)
    except MODEL_ERRORS as exc:
        raise _http(exc) from exc


@models_api.delete("/local/{repo_id:path}", dependencies=[_auth])
async def local_delete(repo_id: str, mgr: ModelManager = Depends(get_manager)) -> dict[str, Any]:
    try:
        await mgr.delete(repo_id)
    except MODEL_ERRORS as exc:
        raise _http(exc) from exc
    return {"deleted": True, "repo_id": repo_id}


@models_api.post("/assign", dependencies=[_auth], response_model=None)
async def assign(
    body: AssignBody, request: Request, mgr: ModelManager = Depends(get_manager)
) -> JSONResponse | dict[str, Any]:
    try:
        mgr.get_local(body.repo_id)
    except MODEL_ERRORS as exc:
        raise _http(exc) from exc
    controller: Controller = request.app.state.controller
    section: dict[str, Any] = {"provider": "litert", "model": body.repo_id}
    if body.backend is not None:
        section["backend"] = body.backend
    if body.num_threads is not None:
        section["num_threads"] = body.num_threads
    patch = {"conversation": {body.role: section}}
    merged = _deep_merge(controller.config().model_dump(), patch)
    try:
        new_config = AppConfig.model_validate(merged)
    except ValidationError as exc:
        return JSONResponse(status_code=422, content={"detail": exc.errors(include_url=False)})
    changes = controller.apply_config(new_config)
    return {"changes": [c.to_json() for c in changes]}


@models_api.get("/file")
def file_get(
    request: Request,
    repo_id: str,
    path: str,
    token: str | None = None,
    mgr: ModelManager = Depends(get_manager),
) -> FileResponse:
    expected: str = request.app.state.control_token
    ok = token_matches(expected, request.headers.get("authorization")) or (
        token is not None and token_matches(expected, f"Bearer {token}")
    )
    if not ok:
        raise HTTPException(401, "missing or invalid bearer token")
    try:
        target = mgr.file_path(repo_id, path)
    except MODEL_ERRORS as exc:
        raise _http(exc) from exc
    return FileResponse(target, media_type="application/octet-stream")

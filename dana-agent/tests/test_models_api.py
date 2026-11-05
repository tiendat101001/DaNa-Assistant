"""Model registry REST routes via TestClient (offline)."""

from __future__ import annotations

import time
import warnings
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from dana_agent.config import ModelsConfig
from dana_agent.control.events import EventBus
from dana_agent.control.models_routes import models_api
from dana_agent.models import ModelManager
from tests.fakes.control import FakeController
from tests.test_models_pull import BLOB, FakeCache

warnings.filterwarnings("ignore", category=DeprecationWarning)

TOKEN = "tok"
AUTH = {"Authorization": f"Bearer {TOKEN}"}
HUB = "https://hub.test"


def handler(request: httpx.Request) -> httpx.Response:
    p = request.url.path
    if p == "/api/models":
        return httpx.Response(
            200,
            json=[
                {
                    "id": "litert-community/whisper-tiny",
                    "downloads": 5,
                    "pipeline_tag": "automatic-speech-recognition",
                    "tags": [],
                },
                {"id": "litert-community/gated", "downloads": 1, "tags": [], "gated": "manual"},
            ],
        )
    if "gated" in p:
        return httpx.Response(403)
    if p.endswith("/zzz-nope"):
        return httpx.Response(401)  # anonymous HF: nonexistent looks like 401
    if p.endswith("/missing"):
        return httpx.Response(404)
    if p == "/api/models/litert-community/whisper-tiny":
        return httpx.Response(200, json={"id": "litert-community/whisper-tiny"})
    if p.endswith("/tree/main"):
        return httpx.Response(
            200,
            json=[{"type": "file", "path": "w_i8.tflite", "size": 1, "lfs": {"size": len(BLOB)}}],
        )
    if request.url.host == "hub.test" and "/resolve/" in p:
        return httpx.Response(302, headers={"Location": "https://cdn.test/b"})
    return httpx.Response(200, content=BLOB)


@pytest.fixture
def ctx(tmp_path: Path) -> Iterator[tuple[TestClient, FakeController, ModelManager]]:
    cfg = ModelsConfig(hub_endpoint=HUB, models_dir=str(tmp_path))
    mgr = ModelManager(cfg, transport=httpx.MockTransport(handler), cache=FakeCache())
    app = FastAPI()
    app.include_router(models_api)
    controller = FakeController()
    app.state.models = mgr
    app.state.controller = controller
    app.state.bus = EventBus()
    app.state.control_token = TOKEN
    with TestClient(app) as c:
        yield c, controller, mgr


def _wait_done(c: TestClient, job_id: str) -> dict[str, Any]:
    for _ in range(100):
        job: dict[str, Any] = c.get(f"/api/v1/models/jobs/{job_id}", headers=AUTH).json()
        if job["status"] not in ("queued", "running"):
            return job
        time.sleep(0.02)
    raise AssertionError("job did not finish")


def test_auth_required(ctx: Any) -> None:
    c = ctx[0]
    assert c.get("/api/v1/models/hub").status_code == 401
    assert c.get("/api/v1/models/local").status_code == 401
    assert c.get("/api/v1/models/file?repo_id=a/b&path=x").status_code == 401


def test_hub_list_and_errors(ctx: Any) -> None:
    c = ctx[0]
    body = c.get("/api/v1/models/hub", headers=AUTH).json()
    assert body["author"] == "litert-community" and body["models"][0]["task"] == "asr"
    assert c.get("/api/v1/models/hub?task=tts", headers=AUTH).json()["models"] == []
    assert c.get("/api/v1/models/hub/litert-community/gated", headers=AUTH).status_code == 403
    assert c.get("/api/v1/models/hub/litert-community/missing", headers=AUTH).status_code == 404
    assert c.get("/api/v1/models/hub/bad", headers=AUTH).status_code in (404, 422)
    d = c.get("/api/v1/models/hub/litert-community/whisper-tiny", headers=AUTH).json()
    assert d["recommended_files"] == ["w_i8.tflite"] and d["total_size"] == len(BLOB)


def test_pull_load_assign_file_delete(ctx: Any) -> None:
    c, controller, _ = ctx
    rid = "litert-community/whisper-tiny"
    r = c.post("/api/v1/models/pull", headers=AUTH, json={"repo_id": rid})
    assert r.status_code == 202
    job = _wait_done(c, r.json()["job"]["id"])
    assert job["status"] == "done", job
    assert c.get("/api/v1/models/jobs", headers=AUTH).json()["jobs"][0]["id"] == job["id"]

    local = c.get("/api/v1/models/local", headers=AUTH).json()
    assert local["models"][0]["id"] == rid and local["models"][0]["status"] == "ready"
    hub = c.get("/api/v1/models/hub", headers=AUTH).json()
    assert hub["models"][0]["pulled"] is True

    loaded = c.post("/api/v1/models/load", headers=AUTH, json={"repo_id": rid}).json()
    assert loaded["status"] == "loaded" and loaded["loaded"]["load_ms"] == 12.5
    unloaded = c.post("/api/v1/models/unload", headers=AUTH, json={"repo_id": rid}).json()
    assert unloaded["loaded"] is None

    a = c.post(
        "/api/v1/models/assign",
        headers=AUTH,
        json={"role": "stt", "repo_id": rid, "backend": "cpu", "num_threads": 2},
    )
    assert a.status_code == 200, a.text
    stt = controller.config().conversation.stt
    assert (stt.provider, stt.model) == ("litert", rid) or a.json()["changes"]
    assert (
        c.post(
            "/api/v1/models/assign", headers=AUTH, json={"role": "stt", "repo_id": "a/unknown"}
        ).status_code
        == 404
    )

    f = c.get(f"/api/v1/models/file?repo_id={rid}&path=w_i8.tflite&token={TOKEN}")
    assert f.status_code == 200 and f.content == BLOB
    assert (
        c.get(f"/api/v1/models/file?repo_id={rid}&path=w_i8.tflite", headers=AUTH).status_code
        == 200
    )
    assert (
        c.get(f"/api/v1/models/file?repo_id={rid}&path=../registry.json", headers=AUTH).status_code
        == 422
    )
    assert c.get(f"/api/v1/models/file?repo_id={rid}&path=nope&token=bad").status_code == 401

    d = c.delete(f"/api/v1/models/local/{rid}", headers=AUTH)
    assert d.json() == {"deleted": True, "repo_id": rid}
    assert c.delete(f"/api/v1/models/local/{rid}", headers=AUTH).status_code == 404


def test_pull_errors_and_job_404(ctx: Any) -> None:
    c = ctx[0]
    assert (
        c.post(
            "/api/v1/models/pull", headers=AUTH, json={"repo_id": "litert-community/gated"}
        ).status_code
        == 403
    )
    assert c.post("/api/v1/models/pull", headers=AUTH, json={"repo_id": "../x"}).status_code == 422
    r = c.post(
        "/api/v1/models/pull",
        headers=AUTH,
        json={"repo_id": "litert-community/whisper-tiny", "files": ["../../etc/passwd"]},
    )
    assert r.status_code == 422
    assert c.get("/api/v1/models/jobs/nope", headers=AUTH).status_code == 404
    assert c.delete("/api/v1/models/jobs/nope", headers=AUTH).status_code == 404
    assert c.post("/api/v1/models/load", headers=AUTH, json={"repo_id": "a/b"}).status_code == 404


def test_nonexistent_repo_is_404_not_gated(ctx: Any) -> None:
    c = ctx[0]
    r = c.get("/api/v1/models/hub/litert-community/zzz-nope", headers=AUTH)
    assert r.status_code == 404
    r = c.post("/api/v1/models/pull", headers=AUTH, json={"repo_id": "litert-community/zzz-nope"})
    assert r.status_code == 404

"""QA regression tests for the model registry control API edge cases (offline)."""

from __future__ import annotations

import threading
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
M = "/api/v1/models"
RID = "litert-community/whisper-tiny"
GATE = threading.Event()


def handler(request: httpx.Request) -> httpx.Response:
    p = request.url.path
    if p == "/api/models":
        return httpx.Response(200, json=[{"id": RID, "downloads": 5, "tags": []}])
    if p.endswith("/does-not-exist"):
        return httpx.Response(401)  # anonymous HF answers 401 for nonexistent repos
    if p == f"/api/models/{RID}":
        return httpx.Response(200, json={"id": RID})
    if p.endswith("/tree/main"):
        return httpx.Response(
            200, json=[{"type": "file", "path": "w.tflite", "size": 1, "lfs": {"size": len(BLOB)}}]
        )
    if "/resolve/" in p:
        GATE.wait(5)  # hold downloads open until the test releases them
    return httpx.Response(200, content=BLOB)


@pytest.fixture
def client(tmp_path: Path) -> Iterator[TestClient]:
    GATE.clear()
    cfg = ModelsConfig(hub_endpoint=HUB, models_dir=str(tmp_path))
    mgr = ModelManager(cfg, transport=httpx.MockTransport(handler), cache=FakeCache())
    app = FastAPI()
    app.include_router(models_api)
    app.state.models = mgr
    app.state.controller = FakeController()
    app.state.bus = EventBus()
    app.state.control_token = TOKEN
    with TestClient(app) as c:
        yield c
    GATE.set()


def _wait(c: TestClient, jid: str) -> dict[str, Any]:
    for _ in range(200):
        j: dict[str, Any] = c.get(f"{M}/jobs/{jid}", headers=AUTH).json()
        if j["status"] not in ("queued", "running"):
            return j
        time.sleep(0.02)
    raise AssertionError("job stuck")


def test_duplicate_pull_and_delete_while_pulling_409(client: TestClient) -> None:
    r = client.post(f"{M}/pull", headers=AUTH, json={"repo_id": RID})
    assert r.status_code == 202
    assert client.post(f"{M}/pull", headers=AUTH, json={"repo_id": RID}).status_code == 409
    assert client.delete(f"{M}/local/{RID}", headers=AUTH).status_code == 409
    assert client.post(f"{M}/load", headers=AUTH, json={"repo_id": RID}).status_code == 409
    GATE.set()
    assert _wait(client, r.json()["job"]["id"])["status"] == "done"


def test_cancel_then_repull_succeeds(client: TestClient) -> None:
    jid = client.post(f"{M}/pull", headers=AUTH, json={"repo_id": RID}).json()["job"]["id"]
    assert client.delete(f"{M}/jobs/{jid}", headers=AUTH).status_code == 200
    GATE.set()
    assert _wait(client, jid)["status"] == "cancelled"
    r2 = client.post(f"{M}/pull", headers=AUTH, json={"repo_id": RID})
    assert r2.status_code == 202
    assert _wait(client, r2.json()["job"]["id"])["status"] == "done"


@pytest.mark.parametrize("bad", ["../x", "a", "a/b/c/../../x", "foo bar/x", ""])
def test_bad_repo_ids_422(client: TestClient, bad: str) -> None:
    assert client.post(f"{M}/pull", headers=AUTH, json={"repo_id": bad}).status_code == 422


def test_unknown_repo_is_404_not_gated_403(client: TestClient) -> None:
    """BUG-QA1 (live: anonymous HF 401 on missing repo was reported as 403 gated)."""
    r = client.post(f"{M}/pull", headers=AUTH, json={"repo_id": "litert-community/does-not-exist"})
    assert r.status_code == 404, r.text
    d = client.get(f"{M}/hub/litert-community/does-not-exist", headers=AUTH)
    assert d.status_code == 404, d.text


def test_progress_never_exceeds_total(client: TestClient) -> None:
    """BUG-QA2 (live whisper-tiny: bytes_done 44976622 > bytes_total 41129939)."""
    GATE.set()
    jid = client.post(f"{M}/pull", headers=AUTH, json={"repo_id": RID}).json()["job"]["id"]
    j = _wait(client, jid)
    assert j["status"] == "done"
    assert j["bytes_done"] <= j["bytes_total"], j


@pytest.mark.parametrize(
    "path", ["../../registry.json", "..%2F..%2Fregistry.json", "/etc/passwd", "%2Fetc%2Fpasswd", ""]
)
def test_file_traversal_rejected(client: TestClient, path: str) -> None:
    GATE.set()
    jid = client.post(f"{M}/pull", headers=AUTH, json={"repo_id": RID}).json()["job"]["id"]
    _wait(client, jid)
    r = client.get(f"{M}/file?repo_id={RID}&path={path}&token={TOKEN}")
    assert r.status_code in (404, 422), (path, r.status_code)
    assert client.get(f"{M}/file?repo_id={RID}&path=w.tflite").status_code == 401
    assert client.get(f"{M}/file?repo_id={RID}&path=w.tflite", headers=AUTH).status_code == 200

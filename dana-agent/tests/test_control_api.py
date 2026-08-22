"""Control REST + WS surface, driven offline via FastAPI's TestClient.

No socket is bound and no network is used — TestClient runs the ASGI app
in-process. A FakeController stands in for the pipeline so these assertions are
about the HTTP/WS contract (docs/PLAN.md §5), not the audio stack.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from dana_agent.control.app import create_control_app
from dana_agent.control.controller import ConfirmationInfo
from dana_agent.control.events import EventBus
from tests.fakes.control import FakeController
from tests.support import clean_config

# starlette's TestClient warns about the installed httpx version; irrelevant here.
warnings.filterwarnings("ignore", category=DeprecationWarning)

TOKEN = "test-token"
AUTH = {"Authorization": f"Bearer {TOKEN}"}


@pytest.fixture
def bus() -> EventBus:
    return EventBus()


@pytest.fixture
def controller() -> FakeController:
    return FakeController()


@pytest.fixture
def client(controller: FakeController, bus: EventBus) -> Iterator[TestClient]:
    app = create_control_app(
        controller.config(), controller, bus, control_token=TOKEN, serve_ui=False
    )
    with TestClient(app) as c:
        yield c


# --- auth ----------------------------------------------------------------


def test_health_is_exempt_from_auth(client: TestClient) -> None:
    r = client.get("/api/v1/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_missing_token_is_401(client: TestClient) -> None:
    assert client.get("/api/v1/config").status_code == 401
    assert client.get("/api/v1/status").status_code == 401


def test_wrong_token_is_401(client: TestClient) -> None:
    r = client.get("/api/v1/config", headers={"Authorization": "Bearer nope"})
    assert r.status_code == 401


def test_correct_token_is_200(client: TestClient) -> None:
    assert client.get("/api/v1/config", headers=AUTH).status_code == 200


# --- config --------------------------------------------------------------


def test_get_config_redacts_secrets(bus: EventBus) -> None:
    cfg = clean_config(conversation={"stt": {"api_key": "supersecret"}})
    ctrl = FakeController(cfg)
    app = create_control_app(cfg, ctrl, bus, control_token=TOKEN, serve_ui=False)
    with TestClient(app) as c:
        body = c.get("/api/v1/config", headers=AUTH).json()
    assert body["conversation"]["stt"]["api_key"] == "**********"
    assert "supersecret" not in str(body)


def test_config_schema_is_json_schema(client: TestClient) -> None:
    schema = client.get("/api/v1/config/schema", headers=AUTH).json()
    assert schema["type"] == "object"
    assert "properties" in schema
    assert "conversation" in schema["properties"]


def test_put_config_labels_applied_and_restart_required(client: TestClient) -> None:
    r = client.put(
        "/api/v1/config",
        headers=AUTH,
        json={"log_level": "DEBUG", "conversation": {"stt": {"provider": "deepgram"}}},
    )
    assert r.status_code == 200
    by_path = {c["path"]: c for c in r.json()["changes"]}
    assert by_path["log_level"]["status"] == "applied"
    assert by_path["conversation.stt.provider"]["status"] == "restart_required"
    # An applied change round-trips through a subsequent GET.
    assert client.get("/api/v1/config", headers=AUTH).json()["log_level"] == "DEBUG"


def test_put_invalid_config_is_422_and_no_mutation(client: TestClient) -> None:
    before = client.get("/api/v1/config", headers=AUTH).json()
    r = client.put("/api/v1/config", headers=AUTH, json={"brain": {"max_steps": -5}})
    assert r.status_code == 422
    assert "detail" in r.json()
    after = client.get("/api/v1/config", headers=AUTH).json()
    assert before == after  # rejected write mutated nothing


# --- providers -----------------------------------------------------------


def test_providers_lists_capabilities(client: TestClient) -> None:
    cat = client.get("/api/v1/providers", headers=AUTH).json()
    assert "stt" in cat and "brain_llm" in cat
    stt = {e["name"]: e for e in cat["stt"]}
    assert "mock" in stt
    caps = stt["mock"]["capabilities"]
    assert caps["streaming"] is True
    assert "en" in caps["languages"]
    assert "latency_class" in caps


# --- session + mic drive the controller ----------------------------------


def test_session_start_stop_drive_controller(
    client: TestClient, controller: FakeController
) -> None:
    assert client.post("/api/v1/session/start", headers=AUTH).json()["status"] == "started"
    assert client.post("/api/v1/session/stop", headers=AUTH).json()["status"] == "stopped"
    assert controller.calls == ["start", "stop"]


def test_mic_mute_unmute_drive_controller(client: TestClient, controller: FakeController) -> None:
    assert client.post("/api/v1/mic/mute", headers=AUTH).json() == {"muted": True}
    assert client.post("/api/v1/mic/unmute", headers=AUTH).json() == {"muted": False}
    assert controller.calls == ["mute", "unmute"]


def test_status_reports_controller_state(client: TestClient) -> None:
    s = client.get("/api/v1/status", headers=AUTH).json()
    assert s["state"] == "idle"
    assert set(s) >= {"state", "session_running", "muted", "uptime_s", "providers"}


# --- confirmations -------------------------------------------------------


def test_resolve_known_confirmation(client: TestClient, controller: FakeController) -> None:
    controller.add_pending(
        ConfirmationInfo(
            id="c1",
            tool="delete_file",
            args_resolved={"path": "x"},
            reversibility="destructive",
            preview="delete_file(path='x')",
        )
    )
    r = client.post("/api/v1/confirmations/c1", headers=AUTH, json={"approved": True})
    assert r.status_code == 200
    assert r.json() == {"id": "c1", "approved": True}
    assert "resolve:c1:True" in controller.calls


def test_resolve_unknown_confirmation_is_404(client: TestClient) -> None:
    r = client.post("/api/v1/confirmations/nope", headers=AUTH, json={"approved": False})
    assert r.status_code == 404


def test_confirmation_defaults_to_deny(client: TestClient, controller: FakeController) -> None:
    controller.add_pending(
        ConfirmationInfo(
            id="c2", tool="t", args_resolved={}, reversibility="destructive", preview="t()"
        )
    )
    # Empty body → fail-safe deny.
    r = client.post("/api/v1/confirmations/c2", headers=AUTH, json={})
    assert r.json()["approved"] is False


# --- audit ---------------------------------------------------------------


def test_audit_since_filter(client: TestClient, controller: FakeController) -> None:
    controller.add_audit({"at": 100.0, "tool": "a"})
    controller.add_audit({"at": 200.0, "tool": "b"})
    all_entries = client.get("/api/v1/audit", headers=AUTH).json()["entries"]
    assert len(all_entries) == 2
    recent = client.get("/api/v1/audit", headers=AUTH, params={"since": 150}).json()["entries"]
    assert [e["tool"] for e in recent] == ["b"]


# --- websocket -----------------------------------------------------------


def test_ws_requires_token(client: TestClient) -> None:
    with pytest.raises(Exception):  # noqa: B017,PT011 - starlette closes with 4401
        with client.websocket_connect("/api/v1/events"):
            pass


def test_ws_streams_envelope(client: TestClient, bus: EventBus) -> None:
    with client.websocket_connect(f"/api/v1/events?token={TOKEN}") as ws:
        bus.publish(
            "state_transition", {"from": "idle", "to": "active", "cause": "wake"}, session_id="s1"
        )
        msg = ws.receive_json()
    assert set(msg) == {"type", "ts", "session_id", "seq", "data"}
    assert msg["type"] == "state_transition"
    assert msg["session_id"] == "s1"
    assert msg["data"]["cause"] == "wake"


# --- Talk UI: activate + typed turns ------------------------------------------


async def test_session_activate_trips_controller(
    client: TestClient, controller: FakeController
) -> None:
    client.post("/api/v1/session/start", headers=AUTH)
    res = client.post("/api/v1/session/activate", headers=AUTH)
    assert res.status_code == 200
    assert res.json()["status"] == "active"
    assert "activate" in controller.calls


async def test_session_activate_without_session_is_409(
    client: TestClient, controller: FakeController
) -> None:
    res = client.post("/api/v1/session/activate", headers=AUTH)
    assert res.status_code == 409
    assert "start the session" in res.json()["detail"]


async def test_session_turn_submits_text(client: TestClient, controller: FakeController) -> None:
    client.post("/api/v1/session/start", headers=AUTH)
    res = client.post("/api/v1/session/turn", json={"text": "play lofi on youtube"}, headers=AUTH)
    assert res.status_code == 200
    assert res.json()["status"] == "submitted"
    assert "turn:play lofi on youtube" in controller.calls


async def test_session_turn_rejects_empty_text(
    client: TestClient, controller: FakeController
) -> None:
    client.post("/api/v1/session/start", headers=AUTH)
    res = client.post("/api/v1/session/turn", json={"text": "   "}, headers=AUTH)
    assert res.status_code == 422


async def test_session_turn_without_session_is_409(
    client: TestClient, controller: FakeController
) -> None:
    res = client.post("/api/v1/session/turn", json={"text": "hello"}, headers=AUTH)
    assert res.status_code == 409

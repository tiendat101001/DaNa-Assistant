from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from dana_agent.config import AppConfig
from dana_agent.control import llm_routes
from dana_agent.control.llm_routes import llm_api
from tests.support import clean_config

TOKEN = "t0ken"
H = {"Authorization": f"Bearer {TOKEN}"}


class _Ctl:
    def __init__(self, cfg: AppConfig) -> None:
        self._cfg = cfg

    def config(self) -> AppConfig:
        return self._cfg


@pytest.fixture
def client() -> Iterator[TestClient]:
    app = FastAPI()
    app.state.controller = _Ctl(clean_config())
    app.state.control_token = TOKEN
    app.include_router(llm_api)
    with TestClient(app) as c:
        yield c


def test_auth_required(client: TestClient) -> None:
    assert client.get("/api/v1/llm/presets").status_code == 401


def test_presets_shape(client: TestClient) -> None:
    presets = client.get("/api/v1/llm/presets", headers=H).json()["presets"]
    assert len(presets) >= 8
    ids = {p["id"] for p in presets}
    assert "gemini-2.5-flash-lite" in ids
    for p in presets:
        for k in ("id", "label", "provider", "model", "base_url", "notes", "expected_ttft_ms"):
            assert k in p
        llm = p["patch"]["conversation"]["llm"]
        assert llm["provider"] == p["provider"]
        assert llm["model"] == p["model"]
        assert llm["max_tokens"] == 160
        # every patch must be a valid config overlay
        clean_config(**p["patch"])
    gem = next(p for p in presets if p["id"] == "gemini-2.5-flash-lite")
    assert gem["patch"]["conversation"]["llm"]["thinking_budget"] == 0


def test_probe_mock(client: TestClient) -> None:
    r = client.post("/api/v1/llm/probe", headers=H, json={}).json()
    assert r["ok"] is True
    assert r["ttft_ms"] is not None


def test_probe_bad_provider(client: TestClient) -> None:
    r = client.post("/api/v1/llm/probe", headers=H, json={"provider": "zzz"}).json()
    assert r["ok"] is False


def _patch_httpx(monkeypatch: pytest.MonkeyPatch, handler: Any) -> None:
    real = httpx.AsyncClient

    def factory(*a: Any, **k: Any) -> httpx.AsyncClient:
        k["transport"] = httpx.MockTransport(handler)
        return real(*a, **k)

    monkeypatch.setattr(httpx, "AsyncClient", factory)


def test_health_ok(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[str] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(str(req.url))
        return httpx.Response(200, json={"data": [{"id": "Qwen/Qwen3-1.7B"}]})

    _patch_httpx(monkeypatch, handler)
    r = client.get(
        "/api/v1/llm/health", params={"base_url": "http://127.0.0.1:30000/v1/"}, headers=H
    )
    assert r.json() == {"ok": True, "models": ["Qwen/Qwen3-1.7B"], "error": None}
    assert seen == ["http://127.0.0.1:30000/v1/models"]


def test_health_down(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    _patch_httpx(monkeypatch, handler)
    r = client.get(
        "/api/v1/llm/health", params={"base_url": "http://127.0.0.1:1/v1"}, headers=H
    ).json()
    assert r["ok"] is False
    assert r["models"] == []
    assert r["error"]


def test_probe_resolves_env_key_fallback(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TG_API_KEY", "tg-secret")
    monkeypatch.setenv("TG_BASE_URL", "https://example.invalid/v1")
    seen: dict[str, Any] = {}

    async def fake_probe(cfg: Any) -> dict[str, Any]:
        seen["cfg"] = cfg
        return {"ok": True, "ttft_ms": 1.0, "total_ms": 2.0, "text": "x", "error": None}

    monkeypatch.setattr(llm_routes, "probe_llm", fake_probe)
    r = client.post(
        "/api/v1/llm/probe", headers=H, json={"provider": "together", "model": "org/m"}
    ).json()
    assert r["ok"] is True
    cfg = seen["cfg"]
    assert cfg.api_key is not None and cfg.api_key.get_secret_value() == "tg-secret"
    assert cfg.base_url == "https://example.invalid/v1"
    assert "tg-secret" not in str(r)

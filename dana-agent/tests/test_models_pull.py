"""Downloader + ModelManager pull flow against mocked transports."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import httpx
import pytest
from pydantic import SecretStr

from dana_agent.config import ModelsConfig
from dana_agent.models import Downloader, ModelManager, ModelStore
from dana_agent.models.errors import ModelBusyError, ModelInputError
from dana_agent.models.pull import new_job
from dana_agent.models.store import model_dir, resolve_model_path, safe_relpath

HUB = "https://hub.test"
CDN = "https://cdn.test"
BLOB = bytes(range(256)) * 40  # 10240 bytes


class Server:
    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self.honor_range = True
        self.slow = False

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if request.url.host == "hub.test" and "/resolve/" in request.url.path:
            return httpx.Response(302, headers={"Location": f"{CDN}/blob{request.url.path}"})
        if request.url.host == "cdn.test":
            rng = request.headers.get("range")
            if rng and self.honor_range:
                start = int(rng.split("=")[1].rstrip("-"))
                return httpx.Response(206, content=BLOB[start:])
            return httpx.Response(200, content=BLOB)
        return httpx.Response(404)


def _downloader(server: Server, token: str | None = None) -> Downloader:
    cfg = ModelsConfig(hub_endpoint=HUB, hf_token=SecretStr(token) if token else None)
    return Downloader(cfg, httpx.MockTransport(server.handler))


async def test_download_atomic_progress_and_auth_stripped(tmp_path: Path) -> None:
    server = Server()
    dl = _downloader(server, "hf_secret")
    job = new_job("org/m")
    await dl.run(job, [{"path": "sub/a.bin", "size": len(BLOB)}], tmp_path)
    dest = model_dir(tmp_path, "org/m") / "sub" / "a.bin"
    assert job.status == "done" and dest.read_bytes() == BLOB
    assert not dest.with_name("a.bin.part").exists()
    assert (job.bytes_done, job.bytes_total, job.files_done) == (len(BLOB), len(BLOB), 1)
    hub_req = next(r for r in server.requests if r.url.host == "hub.test")
    cdn_req = next(r for r in server.requests if r.url.host == "cdn.test")
    assert hub_req.headers["authorization"] == "Bearer hf_secret"
    assert "authorization" not in cdn_req.headers


async def test_resume_uses_range(tmp_path: Path) -> None:
    server = Server()
    dest_dir = model_dir(tmp_path, "org/m")
    dest_dir.mkdir(parents=True)
    (dest_dir / "a.bin.part").write_bytes(BLOB[:4000])
    job = new_job("org/m")
    await _downloader(server).run(job, [{"path": "a.bin", "size": len(BLOB)}], tmp_path)
    assert job.status == "done"
    assert (dest_dir / "a.bin").read_bytes() == BLOB
    assert any(r.headers.get("range") == "bytes=4000-" for r in server.requests)
    assert job.bytes_done == len(BLOB)


async def test_resume_when_server_ignores_range(tmp_path: Path) -> None:
    server = Server()
    server.honor_range = False
    dest_dir = model_dir(tmp_path, "org/m")
    dest_dir.mkdir(parents=True)
    (dest_dir / "a.bin.part").write_bytes(b"garbage")
    job = new_job("org/m")
    await _downloader(server).run(job, [{"path": "a.bin", "size": len(BLOB)}], tmp_path)
    assert (dest_dir / "a.bin").read_bytes() == BLOB


async def test_size_mismatch_fails_and_keeps_part(tmp_path: Path) -> None:
    job = new_job("org/m")
    await _downloader(Server()).run(job, [{"path": "a.bin", "size": len(BLOB) + 5}], tmp_path)
    assert job.status == "failed" and "incomplete" in (job.error or "")
    assert not (model_dir(tmp_path, "org/m") / "a.bin").exists()


@pytest.mark.parametrize(
    "bad", ["../evil", "a/../../evil", "/etc/passwd", "C:/x", "a\\b", "", "a//b", "./a"]
)
def test_traversal_rejected(bad: str) -> None:
    with pytest.raises(ModelInputError):
        safe_relpath(bad)


async def test_run_rejects_traversal_before_network(tmp_path: Path) -> None:
    server = Server()
    job = new_job("org/m")
    with pytest.raises(ModelInputError):
        await _downloader(server).run(job, [{"path": "../x", "size": 1}], tmp_path)
    assert server.requests == []


async def test_cancel_via_task_cancel(tmp_path: Path) -> None:
    started = asyncio.Event()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=_Hang(started))

    dl = Downloader(ModelsConfig(hub_endpoint=HUB), httpx.MockTransport(handler))
    job = new_job("org/m")
    task = asyncio.create_task(dl.run(job, [{"path": "a.bin", "size": 0}], tmp_path))
    await asyncio.wait_for(started.wait(), 2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert job.status == "cancelled"


class _Hang(httpx.AsyncByteStream):
    def __init__(self, started: asyncio.Event) -> None:
        self._started = started

    async def __aiter__(self) -> Any:
        yield b"x" * 10
        self._started.set()
        await asyncio.sleep(60)
        yield b""


# --- manager end-to-end ---------------------------------------------------


class FakeCache:
    def __init__(self) -> None:
        self.loaded: list[Path] = []
        self.unloaded: list[Path] = []

    def get_or_load(self, path: Path, *, backend: str = "cpu", num_threads: int = 4) -> Any:
        self.loaded.append(path)

        class M:
            load_ms = 12.5

            def info(self) -> dict[str, Any]:
                return {"path": str(path)}

        return M()

    def unload(self, path: Path) -> bool:
        self.unloaded.append(path)
        return True


def _hub_handler(server: Server) -> Any:
    tree = [
        {"type": "file", "path": "README.md", "size": 10},
        {"type": "file", "path": "m_i8.tflite", "size": len(BLOB), "lfs": {"size": len(BLOB)}},
        {"type": "file", "path": "m_f32_Qualcomm_SM8650.tflite", "size": 1, "lfs": {"size": 5}},
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        p = request.url.path
        if p == "/api/models/litert-community/whisper-tiny":
            return httpx.Response(200, json={"id": "litert-community/whisper-tiny"})
        if p.endswith("/tree/main"):
            return httpx.Response(200, json=tree)
        if "/resolve/" in p and p.endswith("README.md"):
            return httpx.Response(200, content=b"x" * 10)
        if "/resolve/" in p and request.url.host == "hub.test":
            return httpx.Response(302, headers={"Location": f"{CDN}/b"})
        if request.url.host == "cdn.test":
            return httpx.Response(200, content=BLOB)
        return httpx.Response(404)

    return handler


async def test_manager_pull_load_unload_delete(tmp_path: Path) -> None:
    cfg = ModelsConfig(hub_endpoint=HUB, models_dir=str(tmp_path))
    cache = FakeCache()
    mgr = ModelManager(cfg, transport=httpx.MockTransport(_hub_handler(Server())), cache=cache)
    job = await mgr.start_pull("litert-community/whisper-tiny", load=True)
    with pytest.raises(ModelBusyError):
        await mgr.start_pull("litert-community/whisper-tiny")
    with pytest.raises(ModelBusyError):
        await mgr.delete("litert-community/whisper-tiny")
    await mgr.wait(job.id)
    assert job.status == "done", job.error
    rid = "litert-community/whisper-tiny"
    rec = mgr.store.get(rid)
    assert rec is not None and rec.status == "ready"
    # NPU variant excluded; tokenizer companions land in the openai dir.
    assert [f["path"] for f in rec.files] == ["README.md", "m_i8.tflite"]
    assert (model_dir(tmp_path, "openai/whisper-tiny") / "vocab.json").is_file()
    assert resolve_model_path(tmp_path, rid) == model_dir(tmp_path, rid) / "m_i8.tflite"
    assert ModelStore(tmp_path).get(rid) is not None
    assert not list(tmp_path.glob("*.tmp"))  # noqa: ASYNC240

    local = mgr.get_local(rid)
    assert local["status"] == "loaded" and local["loaded"]["backend"] == "cpu"
    assert cache.loaded == [model_dir(tmp_path, rid) / "m_i8.tflite"]

    out = await mgr.unload(rid)
    assert out["status"] == "ready" and out["loaded"] is None and cache.unloaded

    web = await mgr.load(rid, backend="webgpu")
    assert web["loaded"]["backend"] == "webgpu"
    assert web["loaded"]["info"]["files"][0]["url"].startswith("/api/v1/models/file?")

    assert mgr.file_path(rid, "README.md").is_file()
    with pytest.raises(ModelInputError):
        mgr.file_path(rid, "../registry.json")

    await mgr.delete(rid)
    assert mgr.local_models() == [] and not model_dir(tmp_path, rid).exists()


async def test_manager_cancel_job(tmp_path: Path) -> None:
    started = asyncio.Event()

    def handler(request: httpx.Request) -> httpx.Response:
        p = request.url.path
        if p == "/api/models/org/m":
            return httpx.Response(200, json={"id": "org/m"})
        if p.endswith("/tree/main"):
            return httpx.Response(200, json=[{"type": "file", "path": "a.json", "size": 100}])
        return httpx.Response(200, content=_Hang(started))

    cfg = ModelsConfig(hub_endpoint=HUB, models_dir=str(tmp_path))
    mgr = ModelManager(cfg, transport=httpx.MockTransport(handler), cache=FakeCache())
    job = await mgr.start_pull("org/m")
    await asyncio.wait_for(started.wait(), 2)
    out = await mgr.cancel_job(job.id)
    assert out.status == "cancelled"
    rec = mgr.store.get("org/m")
    assert rec is not None and rec.status == "failed"
    # Partial file kept for resume.
    assert (model_dir(tmp_path, "org/m") / "a.json.part").exists()


async def test_unknown_size_files_grow_total(tmp_path: Path) -> None:
    job = new_job("org/m")
    files = [{"path": "a.bin", "size": len(BLOB)}, {"path": "tok.json", "size": 0}]
    await _downloader(Server()).run(job, files, tmp_path)
    assert job.status == "done" and job.bytes_done == job.bytes_total == 2 * len(BLOB)


@pytest.mark.parametrize(
    "bad", ["a:b", "con", "NUL.txt", "dir/aux.bin", "x.", "x ", "a<b", 'a"b', "a|b", "a?b", "a*b"]
)
def test_windows_unsafe_names_rejected(bad: str) -> None:
    with pytest.raises(ModelInputError):
        safe_relpath(bad)


def test_store_does_not_wipe_on_read_error(tmp_path: Path) -> None:
    from dana_agent.models.store import ModelRecord

    store = ModelStore(tmp_path)
    store.put(ModelRecord(id="a/b"))
    store.manifest_path.write_text("{corrupt", encoding="utf-8")
    assert [r.id for r in store.all()] == ["a/b"]  # last-good, not empty
    fresh = ModelStore(tmp_path)
    with pytest.raises(ValueError):
        fresh.all()


async def test_delete_busy_keeps_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import shutil

    mgr = ModelManager(ModelsConfig(models_dir=str(tmp_path)), cache=FakeCache())
    from dana_agent.models.store import ModelRecord

    mgr.store.put(ModelRecord(id="a/b", status="ready"))
    (model_dir(tmp_path, "a/b")).mkdir(parents=True)
    monkeypatch.setattr(shutil, "rmtree", lambda *a, **k: None)
    monkeypatch.setattr(asyncio, "sleep", lambda *_: _noop())
    with pytest.raises(ModelBusyError):
        await mgr.delete("a/b")
    assert mgr.store.get("a/b") is not None


async def _noop() -> None:
    return None


def test_repo_id_and_revision_validation() -> None:
    from dana_agent.models.store import validate_repo_id, validate_revision

    with pytest.raises(ModelInputError):
        validate_repo_id("org/name\n")
    for bad in ("main..x", "a b", "", "x" * 101, "/abs", "a?b"):
        with pytest.raises(ModelInputError):
            validate_revision(bad)
    assert validate_revision("refs/pr/1") == "refs/pr/1"


async def test_https_to_http_redirect_refused(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "hub.test":
            return httpx.Response(302, headers={"Location": "http://cdn.test/b"})
        return httpx.Response(200, content=BLOB)

    dl = Downloader(ModelsConfig(hub_endpoint=HUB), httpx.MockTransport(handler))
    job = new_job("org/m")
    await dl.run(job, [{"path": "a.bin", "size": 0}], tmp_path)
    assert job.status == "failed" and "redirect" in (job.error or "")

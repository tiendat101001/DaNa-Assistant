"""Orchestrates browsing, pulling, loading and serving of registry models."""

from __future__ import annotations

import asyncio
import fnmatch
import gc
import shutil
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import quote

import httpx

from dana_agent.config import ModelsConfig
from dana_agent.models.errors import (
    ModelBusyError,
    ModelInputError,
    ModelNotFoundError,
    ModelsError,
)
from dana_agent.models.hub import HubClient, HubListing
from dana_agent.models.pull import Downloader, PullJob, new_job
from dana_agent.models.store import (
    ModelRecord,
    ModelStore,
    model_dir,
    resolve_in_model_dir,
    safe_relpath,
    validate_repo_id,
)
from dana_agent.observability import get_logger

log = get_logger(__name__)

_QUANT = ("q8", "int8", "q4", "int4")


class ModelCacheLike(Protocol):
    def get_or_load(self, path: Path, *, backend: str = "cpu", num_threads: int = 4) -> Any: ...
    def unload(self, path: Path) -> bool: ...


class ModelManager:
    def __init__(
        self,
        cfg: ModelsConfig,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        cache: ModelCacheLike | None = None,
    ) -> None:
        self.cfg = cfg
        self.hub = HubClient(cfg, transport)
        self.store = ModelStore(cfg.models_dir)
        self._downloader = Downloader(cfg, transport)
        self._cache = cache
        self._jobs: dict[str, PullJob] = {}
        self._loaded: dict[str, dict[str, Any]] = {}
        self._loaded_paths: dict[str, list[Path]] = {}

    # --- hub ----------------------------------------------------------
    def _pulled_ids(self) -> set[str]:
        return {r.id for r in self.store.all() if r.status == "ready"}

    async def list_hub(
        self,
        search: str | None = None,
        task: str | None = None,
        refresh: bool = False,
        limit: int = 200,
    ) -> dict[str, Any]:
        listing: HubListing = await self.hub.list_models(search, task, refresh, limit)
        pulled = self._pulled_ids()
        for m in listing.models:
            m["pulled"] = m["id"] in pulled
        return {
            "author": self.cfg.hub_author,
            "fetched_at": listing.fetched_at,
            "cached": listing.cached,
            "models": listing.models,
        }

    async def hub_detail(self, repo_id: str, revision: str = "main") -> dict[str, Any]:
        detail = await self.hub.model_detail(repo_id, revision)
        detail["pulled"] = repo_id in self._pulled_ids()
        return detail

    # --- local --------------------------------------------------------
    def local_models(self) -> list[dict[str, Any]]:
        return [self._local_json(r) for r in self.store.all()]

    def get_local(self, repo_id: str) -> dict[str, Any]:
        return self._local_json(self._require(repo_id))

    def _require(self, repo_id: str) -> ModelRecord:
        validate_repo_id(repo_id)
        rec = self.store.get(repo_id)
        if rec is None:
            raise ModelNotFoundError(f"model {repo_id!r} is not pulled")
        return rec

    def _local_json(self, rec: ModelRecord) -> dict[str, Any]:
        out = rec.to_json()
        loaded = self._loaded.get(rec.id)
        if loaded is not None and rec.status == "ready":
            out["status"] = "loaded"
        out["loaded"] = loaded
        return out

    # --- pull ---------------------------------------------------------
    async def start_pull(
        self,
        repo_id: str,
        revision: str = "main",
        files: list[str] | None = None,
        include: list[str] | None = None,
        load: bool = False,
        backend: str = "cpu",
        num_threads: int = 4,
    ) -> PullJob:
        validate_repo_id(repo_id)
        for job in self._jobs.values():
            if job.repo_id == repo_id and job.active:
                raise ModelBusyError(f"{repo_id} is already being pulled (job {job.id})")
        detail = await self.hub.model_detail(repo_id, revision)
        by_path = {f["path"]: f for f in detail["files"]}
        if files:
            chosen = []
            for p in files:
                safe_relpath(p)
                if p not in by_path:
                    raise ModelInputError(f"file {p!r} not in {repo_id}@{revision}")
                chosen.append(by_path[p])
        elif include:
            chosen = _match(detail["files"], include)
        else:
            key = "recommended_files_webgpu" if backend == "webgpu" else "recommended_files"
            chosen = [by_path[p] for p in detail[key] if p in by_path]
        if not chosen:
            raise ModelInputError("no files selected to pull")
        extra = [] if files else list(detail.get("companions", []))

        job = new_job(repo_id, revision, load_after=load)
        job.backend, job.num_threads = backend, num_threads
        self._jobs[job.id] = job
        rec = ModelRecord(
            id=repo_id,
            revision=revision,
            task=detail["task"],
            status="pulling",
            files=[{"path": f["path"], "size": f["size"]} for f in chosen],
            size_bytes=sum(f["size"] for f in chosen),
        )
        self.store.put(rec)
        job.task = asyncio.create_task(self._run_pull(job, [*chosen, *extra], rec))
        return job

    async def _run_pull(self, job: PullJob, files: list[dict[str, Any]], rec: ModelRecord) -> None:
        try:
            await self._downloader.run(job, files, Path(self.cfg.models_dir))
        except asyncio.CancelledError:
            rec.status, rec.error = "failed", "cancelled"
            self.store.put(rec)
            raise
        if job.status == "done":
            self.store.mark_ready(rec)
            log.info("model_pulled", repo_id=job.repo_id, bytes=job.bytes_done)
            if job.load_after:
                try:
                    await self.load(job.repo_id, job.backend, job.num_threads)
                except Exception as exc:  # noqa: BLE001 - pull succeeded; report load failure
                    job.error = f"pulled, but load failed: {exc}"
                    log.warning("model_load_after_pull_failed", repo_id=job.repo_id, error=str(exc))
        else:
            rec.status, rec.error = "failed", job.error
            self.store.put(rec)

    def jobs(self) -> list[PullJob]:
        return sorted(self._jobs.values(), key=lambda j: j.started_at or 0.0, reverse=True)

    def get_job(self, job_id: str) -> PullJob:
        job = self._jobs.get(job_id)
        if job is None:
            raise ModelNotFoundError(f"unknown job {job_id!r}")
        return job

    async def cancel_job(self, job_id: str) -> PullJob:
        job = self.get_job(job_id)
        if job.active and job.task is not None:
            job.task.cancel()
            await asyncio.gather(job.task, return_exceptions=True)
        return job

    async def wait(self, job_id: str) -> PullJob:
        job = self.get_job(job_id)
        if job.task is not None:
            await asyncio.gather(job.task, return_exceptions=True)
        return job

    # --- load / unload ------------------------------------------------
    def _get_cache(self) -> ModelCacheLike:
        if self._cache is None:
            try:
                from dana_agent.runtime.litert import MODEL_CACHE
            except ImportError as exc:
                raise ModelInputError(f"LiteRT runtime unavailable: {exc}") from exc
            self._cache = MODEL_CACHE
        return self._cache

    def _pick_tflite(self, rec: ModelRecord, file: str | None) -> Path | None:
        if file:
            safe_relpath(file)
            if file not in {f["path"] for f in rec.files}:
                raise ModelInputError(f"file {file!r} is not part of {rec.id}")
            return resolve_in_model_dir(self.cfg.models_dir, rec.id, file)
        tfl = [f for f in rec.files if f["path"].lower().endswith(".tflite")]
        if not tfl:
            return None
        quant = [f for f in tfl if any(q in f["path"].lower() for q in _QUANT)]
        best = min(quant or tfl, key=lambda f: (f["size"], f["path"]))
        return resolve_in_model_dir(self.cfg.models_dir, rec.id, best["path"])

    def _stage_paths(self, rec: ModelRecord, file: str | None) -> list[Path]:
        """Every LiteRT graph to load. Multi-stage models (e.g. KittenTTS:
        predictor -> prosody -> vocoder, plus a G2P graph) are only usable when
        *all* stages load, so a cpu load covers every pulled ``.tflite``. An
        explicit ``file`` loads just that one graph."""
        if file:
            one = self._pick_tflite(rec, file)
            return [one] if one is not None else []
        tfl = sorted(f["path"] for f in rec.files if f["path"].lower().endswith(".tflite"))
        return [resolve_in_model_dir(self.cfg.models_dir, rec.id, p) for p in tfl]

    async def load(
        self, repo_id: str, backend: str = "cpu", num_threads: int = 4, file: str | None = None
    ) -> dict[str, Any]:
        rec = self._require(repo_id)
        if rec.status != "ready":
            raise ModelBusyError(f"{repo_id} is not ready (status={rec.status})")
        if backend not in ("cpu", "webgpu"):
            raise ModelInputError(f"unknown backend {backend!r}")
        paths = self._stage_paths(rec, file) if backend == "cpu" else []
        if not paths:
            # Browser-served runtime: nothing to load server-side.
            self._loaded[repo_id] = {
                "backend": "webgpu",
                "num_threads": 0,
                "load_ms": 0.0,
                "info": {
                    "served_to_browser": True,
                    "files": [
                        {
                            "path": f["path"],
                            "size": f["size"],
                            "url": self.file_url(repo_id, f["path"]),
                        }
                        for f in rec.files
                    ],
                },
            }
            return self.get_local(repo_id)
        for path in paths:
            if not path.is_file():
                raise ModelNotFoundError(f"model file missing on disk: {path.name}")
        cache = self._get_cache()
        stages: list[dict[str, Any]] = []
        loaded: list[Path] = []
        try:
            for path in paths:
                model = await asyncio.to_thread(
                    cache.get_or_load, path, backend=backend, num_threads=num_threads
                )
                loaded.append(path)
                stages.append(
                    {
                        "file": path.name,
                        "load_ms": float(getattr(model, "load_ms", 0.0)),
                        "info": model.info() if hasattr(model, "info") else {},
                    }
                )
        except Exception as exc:
            # All-or-nothing: a half-loaded multi-stage model is useless.
            for p in loaded:
                await asyncio.to_thread(cache.unload, p)
            if isinstance(exc, ModelsError):
                raise
            if isinstance(exc, ValueError):
                raise ModelInputError(str(exc)) from exc
            raise ModelInputError(f"load failed: {type(exc).__name__}: {exc}") from exc
        self._loaded[repo_id] = {
            "backend": backend,
            "num_threads": num_threads,
            "load_ms": sum(st["load_ms"] for st in stages),
            "info": stages[0]["info"] if len(stages) == 1 else {"stages": stages},
        }
        self._loaded_paths[repo_id] = loaded
        log.info("model_loaded", repo_id=repo_id, backend=backend, stages=len(stages))
        return self.get_local(repo_id)

    async def unload(self, repo_id: str) -> dict[str, Any]:
        self._require(repo_id)
        paths = self._loaded_paths.pop(repo_id, [])
        self._loaded.pop(repo_id, None)
        for path in paths:
            await asyncio.to_thread(self._get_cache().unload, path)
        return self.get_local(repo_id)

    async def delete(self, repo_id: str) -> None:
        validate_repo_id(repo_id)
        if any(j.repo_id == repo_id and j.active for j in self._jobs.values()):
            raise ModelBusyError(f"{repo_id} is being pulled; cancel the job first")
        if self.store.get(repo_id) is None and not model_dir(self.cfg.models_dir, repo_id).exists():
            raise ModelNotFoundError(f"model {repo_id!r} is not pulled")
        if repo_id in self._loaded:
            await self.unload(repo_id)
        target = model_dir(self.cfg.models_dir, repo_id)
        for attempt in range(5):
            # Windows keeps mmapped .tflite files locked until released.
            gc.collect()
            await asyncio.to_thread(shutil.rmtree, target, True)
            if not target.exists():
                break
            await asyncio.sleep(0.1 * (attempt + 1))
        else:
            raise ModelBusyError(f"could not remove {repo_id}: files are in use")
        self.store.remove(repo_id)

    # --- files --------------------------------------------------------
    def file_url(self, repo_id: str, path: str) -> str:
        return (
            f"/api/v1/models/file?repo_id={quote(repo_id, safe='/')}&path={quote(path, safe='/')}"
        )

    def file_path(self, repo_id: str, rel_path: str) -> Path:
        validate_repo_id(repo_id)
        target = resolve_in_model_dir(self.cfg.models_dir, repo_id, rel_path)
        if not target.is_file():
            raise ModelNotFoundError(f"{rel_path!r} not found in {repo_id}")
        return target


def _match(files: list[dict[str, Any]], patterns: list[str]) -> list[dict[str, Any]]:
    return [
        f
        for f in files
        if any(
            fnmatch.fnmatch(f["path"], p) or fnmatch.fnmatch(f["path"].rsplit("/", 1)[-1], p)
            for p in patterns
        )
    ]

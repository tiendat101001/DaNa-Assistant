"""Resumable, atomic model downloads from the Hub."""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import quote, urljoin, urlsplit

import httpx

from dana_agent.config import ModelsConfig
from dana_agent.models.errors import (
    GATED_MESSAGE,
    HubAuthError,
    HubUnavailableError,
    ModelNotFoundError,
    ModelsError,
)
from dana_agent.models.store import (
    replace_retry,
    resolve_in_model_dir,
    safe_relpath,
    validate_repo_id,
)
from dana_agent.observability import get_logger

log = get_logger(__name__)

_CHUNK = 1024 * 1024
_MAX_REDIRECTS = 5


@dataclass
class PullJob:
    id: str
    repo_id: str
    revision: str = "main"
    status: str = "queued"  # queued | running | done | failed | cancelled
    bytes_done: int = 0
    bytes_total: int = 0
    files_done: int = 0
    files_total: int = 0
    current_file: str | None = None
    error: str | None = None
    started_at: float | None = None
    finished_at: float | None = None
    load_after: bool = False
    # Runtime-only fields (not serialised).
    backend: str = "cpu"
    num_threads: int = 4
    task: asyncio.Task[None] | None = field(default=None, repr=False, compare=False)

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "repo_id": self.repo_id,
            "revision": self.revision,
            "status": self.status,
            "bytes_done": self.bytes_done,
            "bytes_total": self.bytes_total,
            "files_done": self.files_done,
            "files_total": self.files_total,
            "current_file": self.current_file,
            "error": self.error,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "load_after": self.load_after,
        }

    @property
    def active(self) -> bool:
        return self.status in ("queued", "running")


def new_job(repo_id: str, revision: str = "main", *, load_after: bool = False) -> PullJob:
    return PullJob(
        id=uuid.uuid4().hex[:12], repo_id=repo_id, revision=revision, load_after=load_after
    )


class Downloader:
    """Downloads files of one repo into a model directory.

    The Authorization header is attached only to requests whose host equals the
    hub host; redirects (e.g. to a CDN) are followed manually so it is never
    forwarded cross-host.
    """

    def __init__(
        self,
        cfg: ModelsConfig,
        transport: httpx.AsyncBaseTransport | None = None,
        semaphore: asyncio.Semaphore | None = None,
    ) -> None:
        self._cfg = cfg
        self._transport = transport
        self._sem = semaphore or asyncio.Semaphore(cfg.max_concurrent_pulls)
        self._hub_host = urlsplit(cfg.hub_endpoint).netloc.lower()

    def _headers_for(self, url: str) -> dict[str, str]:
        headers = {"Accept-Encoding": "identity"}
        if urlsplit(url).netloc.lower() == self._hub_host and self._cfg.hf_token is not None:
            token = self._cfg.hf_token.get_secret_value()
            if token:
                headers["Authorization"] = f"Bearer {token}"
        return headers

    def file_url(self, repo_id: str, revision: str, path: str) -> str:
        return (
            f"{self._cfg.hub_endpoint.rstrip('/')}/{repo_id}/resolve/"
            f"{quote(revision, safe='')}/{quote(path, safe='/')}"
        )

    async def run(
        self,
        job: PullJob,
        files: list[dict[str, Any]],
        dest_root: Path,
        on_update: Callable[[], None] | None = None,
    ) -> None:
        """Download ``files`` (``{path, size}``) for ``job``. Updates job fields;
        re-raises cancellation after marking the job cancelled."""
        validate_repo_id(job.repo_id)
        for f in files:
            validate_repo_id(f.get("repo_id", job.repo_id))
            safe_relpath(f["path"])  # reject traversal before touching the network
        job.files_total = len(files)
        job.bytes_total = sum(int(f.get("size") or 0) for f in files)
        job.status = "queued"
        try:
            async with self._sem:
                job.status = "running"
                job.started_at = time.time()
                async with httpx.AsyncClient(
                    transport=self._transport,
                    timeout=self._cfg.request_timeout_s,
                ) as client:
                    for f in files:
                        job.current_file = f["path"]
                        dest = resolve_in_model_dir(
                            dest_root, f.get("repo_id", job.repo_id), f["path"]
                        )
                        dest.parent.mkdir(parents=True, exist_ok=True)

                        unknown_size = not int(f.get("size") or 0)

                        def bump(n: int, grow: bool = unknown_size) -> None:
                            job.bytes_done += n
                            if grow:  # size unknown up front: grow the total
                                job.bytes_total += n
                            if on_update:
                                on_update()

                        url = self.file_url(
                            f.get("repo_id", job.repo_id),
                            job.revision if "repo_id" not in f else "main",
                            f["path"],
                        )
                        await self._download(client, url, dest, int(f.get("size") or 0), bump)
                        job.files_done += 1
            job.status = "done"
            job.current_file = None
        except asyncio.CancelledError:
            job.status = "cancelled"
            job.error = "cancelled"
            raise
        except ModelsError as exc:
            job.status = "failed"
            job.error = str(exc)
        except Exception as exc:  # noqa: BLE001 - a job must never be left "running"
            job.status = "failed"
            job.error = f"{type(exc).__name__}: {exc}"
        finally:
            job.finished_at = time.time()
            if on_update:
                on_update()

    async def _open(self, client: httpx.AsyncClient, url: str, offset: int) -> httpx.Response:
        for _ in range(_MAX_REDIRECTS + 1):
            headers = self._headers_for(url)
            if offset:
                headers["Range"] = f"bytes={offset}-"
            try:
                resp = await client.send(
                    client.build_request("GET", url, headers=headers),
                    stream=True,
                    follow_redirects=False,
                )
            except httpx.HTTPError as exc:
                raise HubUnavailableError(f"download failed: {type(exc).__name__}: {exc}") from exc
            if resp.is_redirect and "location" in resp.headers:
                url = urljoin(str(resp.url), resp.headers["location"])
                await resp.aclose()
                continue
            return resp
        raise HubUnavailableError("too many redirects")

    async def _download(
        self,
        client: httpx.AsyncClient,
        url: str,
        dest: Path,
        expected: int,
        bump: Callable[[int], None],
    ) -> None:
        part = dest.with_name(dest.name + ".part")
        offset = part.stat().st_size if part.exists() else 0
        if expected and offset > expected:
            part.unlink()
            offset = 0
        if expected and offset == expected:
            bump(offset)
            await asyncio.to_thread(replace_retry, part, dest)
            return

        for attempt in range(2):
            resp = await self._open(client, url, offset)
            try:
                if resp.status_code in (401, 403):
                    raise HubAuthError(GATED_MESSAGE)
                if resp.status_code == 404:
                    raise ModelNotFoundError(f"file not found on hub: {dest.name}")
                if resp.status_code == 416 and attempt == 0:
                    part.unlink(missing_ok=True)
                    offset = 0
                    continue
                if resp.status_code not in (200, 206):
                    raise HubUnavailableError(f"hub returned HTTP {resp.status_code}")
                resumed = resp.status_code == 206 and offset > 0
                if offset and not resumed:
                    offset = 0  # server ignored Range: restart from scratch
                written = offset
                if resumed:
                    bump(offset)
                with part.open("ab" if resumed else "wb") as fh:
                    async for chunk in resp.aiter_bytes(_CHUNK):
                        fh.write(chunk)
                        written += len(chunk)
                        bump(len(chunk))
                if expected and written != expected:
                    raise HubUnavailableError(
                        f"incomplete download of {dest.name}: {written}/{expected} bytes"
                    )
                await asyncio.to_thread(replace_retry, part, dest)
                return
            finally:
                await resp.aclose()
        raise HubUnavailableError(f"could not download {dest.name}")

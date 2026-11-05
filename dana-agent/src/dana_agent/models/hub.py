"""Async HuggingFace Hub client for browsing an organisation's models."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any

import httpx

from dana_agent.config import ModelsConfig
from dana_agent.models.errors import (
    GATED_MESSAGE,
    HubAuthError,
    HubUnavailableError,
    ModelNotFoundError,
)
from dana_agent.models.store import validate_repo_id
from dana_agent.observability import get_logger

log = get_logger(__name__)

# HF caps one listing page at 1000; the org is ~450 repos today. _MAX_CATALOG
# bounds pagination so a misbehaving Link header can never loop forever.
_PAGE_SIZE = 1000
_MAX_CATALOG = 5000

_PIPELINE_TASKS = {
    "automatic-speech-recognition": "asr",
    "text-to-speech": "tts",
    "text-to-audio": "tts",
    "text-generation": "llm",
    "text2text-generation": "llm",
    "image-text-to-text": "llm",
    "feature-extraction": "embedding",
    "sentence-similarity": "embedding",
    "object-detection": "vision",
}
_KEYWORDS = (
    ("asr", ("whisper", "moonshine", "asr", "speech")),
    ("tts", ("tts", "piper", "kokoro", "vits", "melo")),
    (
        "llm",
        ("gemma", "qwen", "llama", "phi", "smol", "deepseek", "mistral", "tinyllama"),
    ),
)
_FORMATS = {".tflite": "tflite", ".litertlm": "litertlm", ".task": "task"}


def classify_task(pipeline_tag: str | None, tags: list[str], name: str) -> str:
    if pipeline_tag:
        if pipeline_tag.startswith("audio-") or pipeline_tag == "voice-activity-detection":
            return "audio"
        if pipeline_tag in _PIPELINE_TASKS:
            return _PIPELINE_TASKS[pipeline_tag]
        if pipeline_tag.startswith("image-"):
            return "vision"
    haystack = " ".join([name, *tags]).lower()
    for task, words in _KEYWORDS:
        if any(re.search(rf"(?<![a-z]){re.escape(w)}", haystack) for w in words):
            return task
    return "other"


def file_format(path: str) -> str:
    lower = path.lower()
    for ext, fmt in _FORMATS.items():
        if lower.endswith(ext):
            return fmt
    return "json" if lower.endswith(".json") else "other"


def _gated(value: Any) -> bool:
    if isinstance(value, str):
        return value.lower() not in ("", "false")
    return bool(value)


def hub_model(item: dict[str, Any]) -> dict[str, Any]:
    """Normalise one ``/api/models`` item into the ``HubModel`` shape."""
    repo_id = str(item.get("id") or item.get("modelId") or "")
    author, _, name = repo_id.partition("/")
    tags = [str(t) for t in item.get("tags") or []]
    pipeline_tag = item.get("pipeline_tag")
    formats: list[str] = []
    for sib in item.get("siblings") or []:
        fmt = _FORMATS.get(_ext(str(sib.get("rfilename", ""))))
        if fmt and fmt not in formats:
            formats.append(fmt)
    return {
        "id": repo_id,
        "author": str(item.get("author") or author),
        "name": name or repo_id,
        "task": classify_task(pipeline_tag, tags, name or repo_id),
        "pipeline_tag": pipeline_tag,
        "tags": tags,
        "downloads": int(item.get("downloads") or 0),
        "likes": int(item.get("likes") or 0),
        "last_modified": item.get("lastModified") or item.get("last_modified"),
        "gated": _gated(item.get("gated")),
        "formats": formats,
        "pulled": False,
    }


def _ext(path: str) -> str:
    dot = path.lower().rfind(".")
    return path.lower()[dot:] if dot >= 0 else ""


_NPU_RE = re.compile(r"(mediatek_|qualcomm_|_gpu(_|\.))", re.IGNORECASE)
_FP16 = {"fp16", "f16"}
_QUANT_TOKENS = {"i8", "int8", "q8", "q4", "int4"}
_F32 = {"f32", "fp32"}
_PRECISION = _FP16 | _QUANT_TOKENS | _F32
_SIDECAR_EXT = (".json", ".txt", ".model", ".md", ".npz", ".bin", ".gz")
_MAX_SIDECAR_BYTES = 64 * 1024 * 1024

# Whisper .tflite files do not embed the tokenizer: pull the upstream one too.
_WHISPER_TOKENIZER = (
    "vocab.json",
    "added_tokens.json",
    "tokenizer.json",
    "merges.txt",
    "special_tokens_map.json",
)
COMPANIONS: dict[str, tuple[str, tuple[str, ...]]] = {
    f"litert-community/whisper-{size}": (f"openai/whisper-{size}", _WHISPER_TOKENIZER)
    for size in ("tiny", "base", "small", "medium", "large-v3", "large-v3-turbo")
}
# Moonshine .tflite exports don't embed the SentencePiece tokenizer either.
COMPANIONS.update(
    {
        f"litert-community/moonshine-{size}": (
            f"UsefulSensors/moonshine-{size}",
            ("tokenizer.json",),
        )
        for size in ("tiny", "base")
    }
)


def companion_files(repo_id: str) -> list[dict[str, Any]]:
    """Files from an upstream repo a model needs (``{repo_id, path, size}``)."""
    entry = COMPANIONS.get(repo_id)
    if entry is None:
        return []
    upstream, names = entry
    return [{"repo_id": upstream, "path": n, "size": 0} for n in names]


def _tokens(path: str) -> set[str]:
    stem = path.rsplit("/", 1)[-1].rsplit(".", 1)[0].lower()
    return set(re.split(r"[_\-.]", stem))


def _group_key(path: str) -> tuple[str, str]:
    head, _, name = path.rpartition("/")
    parts = [t for t in re.split(r"([_\-.])", name.rsplit(".", 1)[0].lower())]
    kept = [t for t in parts if t.lower() not in _PRECISION]
    return head, re.sub(r"[_\-.]{2,}", "_", "".join(kept)).strip("_-.")


def _pick_tflite(files: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cands = [f for f in files if _ext(f["path"]) == ".tflite" and not _NPU_RE.search(f["path"])]
    tiers = (
        [f for f in cands if _tokens(f["path"]) & _FP16],
        [f for f in cands if _tokens(f["path"]) & _QUANT_TOKENS],
        [f for f in cands if not _tokens(f["path"]) & _F32],
        cands,
    )
    pool = next((t for t in tiers if t), [])
    groups: dict[tuple[str, str], dict[str, Any]] = {}
    for f in pool:
        key = _group_key(f["path"])
        if key not in groups or (f["size"], f["path"]) < (groups[key]["size"], groups[key]["path"]):
            groups[key] = f
    return sorted(groups.values(), key=lambda f: f["path"])


def recommended_files(files: list[dict[str, Any]], backend: str = "cpu") -> list[str]:
    """Sidecars plus the runtime file(s) for ``backend`` ("cpu" or "webgpu").

    Skips hardware-specific NPU/GPU variants and picks one precision tier, so a
    default pull is never every variant of a repo.
    """
    sidecars = [
        f["path"]
        for f in files
        if _ext(f["path"]) in _SIDECAR_EXT
        and not f["path"].rsplit("/", 1)[-1].startswith((".", "bench"))
        and not f["path"].startswith("samples/")
        and f["size"] <= _MAX_SIDECAR_BYTES
    ]
    litertlm = sorted(
        (f for f in files if _ext(f["path"]) == ".litertlm"), key=lambda f: (f["size"], f["path"])
    )
    task = sorted(
        (f for f in files if _ext(f["path"]) == ".task"), key=lambda f: (f["size"], f["path"])
    )
    tflite = _pick_tflite(files)
    if backend == "webgpu":
        runtime = litertlm[:1] or task[:1] or tflite
    else:
        generic = [f for f in litertlm if not _NPU_RE.search(f["path"])]
        runtime = tflite or generic[:1] or litertlm[:1] or task[:1]
    return [*sidecars, *(f["path"] for f in runtime)]


@dataclass
class HubListing:
    models: list[dict[str, Any]]
    fetched_at: float
    cached: bool


class HubClient:
    def __init__(
        self,
        cfg: ModelsConfig,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._cfg = cfg
        self._transport = transport
        self._cache: dict[str | None, tuple[float, float, list[dict[str, Any]]]] = {}

    @property
    def endpoint(self) -> str:
        return self._cfg.hub_endpoint.rstrip("/")

    def auth_headers(self) -> dict[str, str]:
        if self._cfg.hf_token is not None:
            token = self._cfg.hf_token.get_secret_value()
            if token:
                return {"Authorization": f"Bearer {token}"}
        return {}

    def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            transport=self._transport,
            timeout=self._cfg.request_timeout_s,
            headers=self.auth_headers(),
        )

    async def _get(
        self, client: httpx.AsyncClient, url: str, params: dict[str, Any] | None = None
    ) -> httpx.Response:
        try:
            resp = await client.get(url, params=params)
        except httpx.HTTPError as exc:
            raise HubUnavailableError(f"hub unreachable: {type(exc).__name__}: {exc}") from exc
        if resp.status_code in (401, 403):
            raise HubAuthError(GATED_MESSAGE)
        if resp.status_code == 404:
            raise ModelNotFoundError(f"not found on hub: {url}")
        if resp.status_code >= 400:
            raise HubUnavailableError(f"hub returned HTTP {resp.status_code}")
        return resp

    async def _get_json_pages(
        self, client: httpx.AsyncClient, url: str, params: dict[str, Any], max_items: int
    ) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        next_url: str | None = url
        next_params: dict[str, Any] | None = params
        while next_url and len(items) < max_items:
            resp = await self._get(client, next_url, next_params)
            data = resp.json()
            if not isinstance(data, list):
                raise HubUnavailableError("unexpected hub response shape")
            items.extend(d for d in data if isinstance(d, dict))
            nxt = resp.links.get("next")
            next_url = nxt["url"] if nxt else None
            next_params = None
        return items

    async def list_models(
        self,
        search: str | None = None,
        task: str | None = None,
        refresh: bool = False,
        limit: int = 200,
    ) -> HubListing:
        """Whole-org catalog (cached per search term), then task filter, then
        sort + ``limit``. The limit applies to the *result*, never to the
        fetch: HF's listing order is not by downloads, so truncating before
        filtering silently dropped popular repos (e.g. Kokoro / KittenTTS)."""
        key = search or None
        now = time.monotonic()
        hit = self._cache.get(key)
        if hit and not refresh and now - hit[0] < self._cfg.catalog_ttl_s:
            models, cached, fetched = hit[2], True, hit[1]
        else:
            params: dict[str, Any] = {
                "author": self._cfg.hub_author,
                "full": "true",
                "limit": _PAGE_SIZE,
            }
            if search:
                params["search"] = search
            async with self._client() as client:
                raw = await self._get_json_pages(
                    client, f"{self.endpoint}/api/models", params, _MAX_CATALOG
                )
            models = sorted((hub_model(i) for i in raw), key=lambda m: (-m["downloads"], m["id"]))
            fetched = time.time()
            self._cache[key] = (now, fetched, models)
            cached = False
            log.info("hub_catalog_fetched", count=len(models), search=search)
        if task:
            models = [m for m in models if m["task"] == task]
        return HubListing([dict(m) for m in models[:limit]], fetched, cached)

    async def model_detail(self, repo_id: str, revision: str = "main") -> dict[str, Any]:
        validate_repo_id(repo_id)
        async with self._client() as client:
            meta_resp = await self._get(client, f"{self.endpoint}/api/models/{repo_id}")
            meta = meta_resp.json()
            tree = await self._get_json_pages(
                client,
                f"{self.endpoint}/api/models/{repo_id}/tree/{revision}",
                {"recursive": "true"},
                100_000,
            )
        files: list[dict[str, Any]] = []
        for entry in tree:
            if entry.get("type") != "file":
                continue
            lfs = entry.get("lfs")
            size = (
                int(lfs["size"])
                if isinstance(lfs, dict) and "size" in lfs
                else int(entry.get("size") or 0)
            )
            path = str(entry.get("path", ""))
            files.append({"path": path, "size": size, "format": file_format(path)})
        if not isinstance(meta, dict):
            raise HubUnavailableError("unexpected hub response shape")
        meta.setdefault("id", repo_id)
        model = hub_model(meta)
        for f in files:
            fmt = _FORMATS.get(_ext(f["path"]))
            if fmt and fmt not in model["formats"]:
                model["formats"].append(fmt)
        return {
            **model,
            "files": files,
            "total_size": sum(f["size"] for f in files),
            "recommended_files": recommended_files(files, "cpu"),
            "recommended_files_webgpu": recommended_files(files, "webgpu"),
            "companions": companion_files(repo_id),
        }

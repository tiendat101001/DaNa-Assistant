"""On-disk layout and JSON manifest for pulled models."""

from __future__ import annotations

import json
import os
import re
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

from dana_agent.models.errors import ModelInputError

_REPO_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*$")


_WIN_BAD_CHARS = re.compile(r'[<>:"|?*\x00-\x1f]')
_WIN_RESERVED = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}


def replace_retry(src: str | Path, dst: str | Path, attempts: int = 5) -> None:
    """``os.replace`` with backoff: on Windows AV/indexers briefly lock files."""
    for i in range(attempts):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if i == attempts - 1:
                raise
            time.sleep(0.05 * 2**i)


def validate_repo_id(repo_id: str) -> str:
    """Return ``repo_id`` if it is a well-formed ``org/name`` id, else raise."""
    if not isinstance(repo_id, str) or not _REPO_RE.match(repo_id) or ".." in repo_id:
        raise ModelInputError(f"invalid repo id {repo_id!r} (expected 'org/name')")
    return repo_id


def safe_relpath(path: str) -> PurePosixPath:
    """Validate a repo-relative file name: no absolute paths, drive letters,
    backslashes, NULs, empty or ``..`` segments."""
    if not isinstance(path, str) or not path or "\x00" in path or "\\" in path:
        raise ModelInputError(f"unsafe file path {path!r}")
    if path.startswith("/") or re.match(r"^[A-Za-z]:", path):
        raise ModelInputError(f"unsafe file path {path!r}")
    parts = path.split("/")
    if any(p in ("", ".", "..") for p in parts):
        raise ModelInputError(f"unsafe file path {path!r}")
    # Windows: reserved chars (':' also creates NTFS alternate streams), control
    # chars, trailing dot/space, and reserved device names.
    for part in parts:
        if (
            _WIN_BAD_CHARS.search(part)
            or part[-1] in ". "
            or part.split(".", 1)[0].upper() in _WIN_RESERVED
        ):
            raise ModelInputError(f"unsafe file path {path!r}")
    return PurePosixPath(*parts)


def model_dir(models_dir: str | Path, repo_id: str) -> Path:
    """``<models_dir>/<org>__<name>``."""
    validate_repo_id(repo_id)
    return Path(models_dir) / repo_id.replace("/", "__")


def resolve_in_model_dir(models_dir: str | Path, repo_id: str, rel_path: str) -> Path:
    """Resolve ``rel_path`` inside the model dir; raise if it escapes it."""
    base = model_dir(models_dir, repo_id).resolve()
    target = (base / safe_relpath(rel_path)).resolve()
    if base != target and base not in target.parents:
        raise ModelInputError(f"unsafe file path {rel_path!r}")
    return target


def resolve_model_path(
    models_dir: str | Path, repo_id: str, *, suffixes: tuple[str, ...] = (".tflite",)
) -> Path | None:
    """First file under the model dir with one of ``suffixes`` (largest first)."""
    base = model_dir(models_dir, repo_id)
    if not base.is_dir():
        return None
    found = [
        p
        for p in base.rglob("*")
        if p.is_file() and p.suffix.lower() in suffixes and not p.name.endswith(".part")
    ]
    if not found:
        return None
    found.sort(key=lambda p: (-p.stat().st_size, str(p)))
    return found[0]


@dataclass
class ModelRecord:
    id: str
    revision: str = "main"
    task: str = "other"
    status: str = "pulling"  # pulling | ready | failed
    files: list[dict[str, Any]] = field(default_factory=list)  # {path, size}
    size_bytes: int = 0
    pulled_at: float | None = None
    error: str | None = None

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


class ModelStore:
    """JSON manifest ``<models_dir>/registry.json`` with atomic writes."""

    def __init__(self, models_dir: str | Path) -> None:
        self.models_dir = Path(models_dir)
        self._lock = threading.Lock()
        self._last_good: dict[str, dict[str, Any]] | None = None

    @property
    def manifest_path(self) -> Path:
        return self.models_dir / "registry.json"

    def _read(self) -> dict[str, dict[str, Any]]:
        # Missing file means an empty registry. Any other failure must not look
        # empty (the next write would wipe the registry): fall back to the last
        # good copy, else raise.
        try:
            raw = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {}
        except (OSError, json.JSONDecodeError):
            if self._last_good is not None:
                return dict(self._last_good)
            raise
        models = raw.get("models") if isinstance(raw, dict) else None
        out: dict[str, dict[str, Any]] = models if isinstance(models, dict) else {}
        self._last_good = dict(out)
        return out

    def _write(self, models: dict[str, dict[str, Any]]) -> None:
        self.models_dir.mkdir(parents=True, exist_ok=True)
        tmp = self.manifest_path.with_name(self.manifest_path.name + ".tmp")
        tmp.write_text(json.dumps({"version": 1, "models": models}, indent=2), encoding="utf-8")
        replace_retry(tmp, self.manifest_path)
        self._last_good = dict(models)

    def all(self) -> list[ModelRecord]:
        with self._lock:
            return [_record(v) for v in self._read().values()]

    def get(self, repo_id: str) -> ModelRecord | None:
        with self._lock:
            raw = self._read().get(repo_id)
        return _record(raw) if raw else None

    def put(self, record: ModelRecord) -> None:
        with self._lock:
            models = self._read()
            models[record.id] = record.to_json()
            self._write(models)

    def remove(self, repo_id: str) -> bool:
        with self._lock:
            models = self._read()
            if repo_id not in models:
                return False
            del models[repo_id]
            self._write(models)
            return True

    def mark_ready(self, record: ModelRecord) -> ModelRecord:
        record.status = "ready"
        record.pulled_at = time.time()
        record.error = None
        self.put(record)
        return record


def _record(raw: dict[str, Any]) -> ModelRecord:
    known = ModelRecord.__dataclass_fields__
    return ModelRecord(**{k: v for k, v in raw.items() if k in known})

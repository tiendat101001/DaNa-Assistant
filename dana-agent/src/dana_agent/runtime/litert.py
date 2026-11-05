"""LiteRT (ex-TFLite) CPU runtime: model wrapper + process-wide cache.

``ai_edge_litert`` is imported lazily (falling back to ``tflite_runtime`` or
``tensorflow.lite``) so the offline suite never needs it.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from dana_agent.observability import get_logger

log = get_logger(__name__)

_INTERPRETER_MODULES = (
    "ai_edge_litert.interpreter",
    "tflite_runtime.interpreter",
    "tensorflow.lite",
)


def _import_interpreter_cls() -> Any:
    import importlib

    for name in _INTERPRETER_MODULES:
        try:
            return importlib.import_module(name).Interpreter
        except (ImportError, AttributeError):
            continue
    raise ImportError(
        "No LiteRT interpreter found: install `ai-edge-litert` (uv sync --extra local)"
    )


def litert_available() -> bool:
    try:
        _import_interpreter_cls()
    except ImportError:
        return False
    return True


def _dtype_name(d: Any) -> str:
    return str(getattr(d, "__name__", d))


class LiteRTModel:
    """One loaded ``.tflite`` model. ``interpreter_factory`` is injectable for tests."""

    def __init__(
        self,
        path: str | Path,
        *,
        backend: str = "cpu",
        num_threads: int = 4,
        interpreter_factory: Any | None = None,
    ) -> None:
        if backend == "webgpu":
            raise ValueError("webgpu runs in the browser runtime")
        self.path = Path(path)
        self.backend = backend
        self.num_threads = num_threads
        t0 = time.perf_counter()
        factory = interpreter_factory or _import_interpreter_cls()
        self._interp: Any = factory(model_path=str(self.path), num_threads=num_threads)
        self._lock = threading.Lock()
        self._allocated = False
        self.load_ms = (time.perf_counter() - t0) * 1000.0
        log.info("litert_model_loaded", path=str(self.path), load_ms=round(self.load_ms, 1))

    # -- introspection ---------------------------------------------------
    def signatures(self) -> dict[str, dict[str, list[str]]]:
        try:
            raw = self._interp.get_signature_list() or {}
        except Exception:  # noqa: BLE001 - old interpreters lack signatures
            return {}
        return {
            str(k): {
                "inputs": [str(x) for x in (v.get("inputs") or [])],
                "outputs": [str(x) for x in (v.get("outputs") or [])],
            }
            for k, v in raw.items()
        }

    def runner(self, signature: str | None = None) -> Any:
        sigs = self.signatures()
        if signature is None:
            if len(sigs) != 1:
                raise ValueError(f"signature required; model has {sorted(sigs)}")
            signature = next(iter(sigs))
        elif signature not in sigs:
            raise KeyError(f"unknown signature {signature!r}; have {sorted(sigs)}")
        return self._interp.get_signature_runner(signature)

    def input_details(self) -> list[dict[str, Any]]:
        return list(self._interp.get_input_details())

    def output_details(self) -> list[dict[str, Any]]:
        return list(self._interp.get_output_details())

    def invoke_default(self, inputs: list[np.ndarray]) -> list[np.ndarray]:
        """Run a signature-less model: inputs map to input tensors in order."""
        with self._lock:
            details = self.input_details()
            if len(inputs) != len(details):
                raise ValueError(f"model takes {len(details)} inputs, got {len(inputs)}")
            for arr, d in zip(inputs, details, strict=True):
                if tuple(d["shape"]) != tuple(arr.shape):
                    self._interp.resize_tensor_input(d["index"], list(arr.shape))
                    self._allocated = False
            if not self._allocated:
                self._interp.allocate_tensors()
                self._allocated = True
                details = self.input_details()
            for arr, d in zip(inputs, details, strict=True):
                self._interp.set_tensor(d["index"], np.asarray(arr, dtype=d["dtype"]))
            self._interp.invoke()
            return [np.array(self._interp.get_tensor(o["index"])) for o in self.output_details()]

    def info(self) -> dict[str, Any]:
        def tensors(details: list[dict[str, Any]]) -> list[dict[str, Any]]:
            return [
                {
                    "name": str(d.get("name", "")),
                    "shape": [int(s) for s in d.get("shape", [])],
                    "dtype": _dtype_name(d.get("dtype")),
                }
                for d in details
            ]

        try:
            size = self.path.stat().st_size
        except OSError:
            size = 0
        try:
            ins, outs = tensors(self.input_details()), tensors(self.output_details())
        except Exception:  # noqa: BLE001
            ins, outs = [], []
        return {
            "path": str(self.path),
            "size": size,
            "backend": self.backend,
            "num_threads": self.num_threads,
            "load_ms": round(self.load_ms, 2),
            "signatures": self.signatures(),
            "inputs": ins,
            "outputs": outs,
        }

    def close(self) -> None:
        """Drop every interpreter reference so the mmapped .tflite is released
        (Windows keeps the file locked until the interpreter is collected)."""
        import gc

        with self._lock:
            interp, self._interp = self._interp, None
            self._allocated = False
            del interp
        gc.collect()


class ModelCache:
    """Process-wide, thread-safe cache keyed by (resolved path, backend, threads)."""

    def __init__(self, interpreter_factory: Any | None = None) -> None:
        self._lock = threading.Lock()
        self._models: dict[tuple[str, str, int], LiteRTModel] = {}
        self._factory = interpreter_factory

    def get_or_load(
        self, path: str | Path, *, backend: str = "cpu", num_threads: int = 4
    ) -> LiteRTModel:
        if backend == "webgpu":
            raise ValueError("webgpu runs in the browser runtime")
        resolved = Path(path).resolve()
        key = (str(resolved), backend, num_threads)
        with self._lock:
            model = self._models.get(key)
            if model is None:
                if not resolved.is_file():
                    raise FileNotFoundError(str(resolved))
                model = LiteRTModel(
                    resolved,
                    backend=backend,
                    num_threads=num_threads,
                    interpreter_factory=self._factory,
                )
                self._models[key] = model
            return model

    def unload(self, path: str | Path) -> bool:
        resolved = str(Path(path).resolve())
        with self._lock:
            keys = [k for k in self._models if k[0] == resolved]
            models = [self._models.pop(k) for k in keys]
        for m in models:
            m.close()
        return bool(models)

    def unload_under(self, directory: str | Path) -> int:
        """Unload every model whose file lives under ``directory`` (call before
        deleting / re-pulling a repo). Returns how many were closed."""
        root = Path(directory).resolve()
        with self._lock:
            keys = [k for k in self._models if Path(k[0]).is_relative_to(root)]
            models = [self._models.pop(k) for k in keys]
        for m in models:
            m.close()
        return len(models)

    def loaded(self) -> list[dict[str, Any]]:
        with self._lock:
            return [m.info() for m in self._models.values()]


MODEL_CACHE = ModelCache()

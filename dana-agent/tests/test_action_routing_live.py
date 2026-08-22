"""Live pytest wrapper for the routing eval — `pytest -m live -k routing`.

The canonical implementation (cases, predicates, runner) is the standalone
script ``scripts/action_routing_eval.py``; this module runs each of its 20
cases as one live test so the eval integrates with the usual opt-in flow.
Requires the real Gemini key in ``dana-agent/.env`` (one model call per case).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.live


def _load_eval_module() -> Any:
    path = Path(__file__).resolve().parents[1] / "scripts" / "action_routing_eval.py"
    spec = importlib.util.spec_from_file_location("action_routing_eval", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)  # main() is __main__-guarded; import is inert
    return module


_EVAL = _load_eval_module()
_CASES = _EVAL.CASES


@pytest.fixture(scope="module")
def bound_model() -> Any:
    return _EVAL.build_eval_model(None, 0.0)


@pytest.fixture(scope="module")
def bound_tools() -> Any:
    return _EVAL.build_bound_tools()


@pytest.mark.parametrize("case", _CASES, ids=[c.id for c in _CASES])
def test_routing_case(bound_model: Any, bound_tools: Any, case: Any) -> None:
    ok, detail = _EVAL.evaluate_case(bound_model, bound_tools, case)
    assert ok, f"{case.utterance!r} — expected {case.describe()}; {detail}"

"""Startup warmup: pay model-load / connection cold-start before the first turn."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from dana_agent.observability import get_logger

if TYPE_CHECKING:
    from dana_agent.app import AssembledStack
    from dana_agent.config import AppConfig

log = get_logger(__name__)

_Step = Callable[[], Awaitable[object]]


async def _timed(component: str, make: _Step) -> tuple[str, float]:
    """Run ``make()``; never raises. -1.0 on failure."""
    t0 = time.perf_counter()
    try:
        await make()
    except Exception as exc:  # noqa: BLE001 - warmup must never break startup
        log.warning("warmup_failed", component=component, exc=str(exc))
        return component, -1.0
    ms = round((time.perf_counter() - t0) * 1000, 1)
    log.info("warmup", component=component, ms=ms)
    return component, ms


def _brain_model(stack: AssembledStack) -> Any | None:
    brain = stack.brain
    if brain is None:
        return None
    for attr in ("model", "_model", "chat_model", "_chat_model"):
        model = getattr(brain, attr, None)
        if model is not None:
            return model
    return None


async def warmup_stack(stack: AssembledStack, cfg: AppConfig) -> dict[str, float]:
    """Warm every component concurrently; returns ``{component: ms}`` (-1 = failed)."""
    steps: list[tuple[str, _Step]] = []
    for name in ("stt", "tts", "s2s"):
        fn = getattr(getattr(stack, name, None), "warmup", None)
        if callable(fn):
            steps.append((name, fn))

    model = _brain_model(stack)
    if model is not None:
        try:
            from dana_agent.brain.model import warmup_chat_model  # noqa: PLC0415
        except (ImportError, AttributeError):
            log.debug("warmup_chat_model_unavailable")
        else:

            async def _llm() -> object:
                return await warmup_chat_model(model)

            steps.append(("llm", _llm))

    if not steps:
        return {}
    results = await asyncio.gather(*(_timed(n, f) for n, f in steps))
    return dict(results)

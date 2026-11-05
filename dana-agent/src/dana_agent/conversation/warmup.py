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


async def keepalive_ping(model: Any) -> str:
    """One token-free ping on the connection pool the model itself uses.

    OpenAI-compatible LangChain models expose ``root_async_client`` (an
    ``openai.AsyncOpenAI``); ``models.list()`` is a cheap authenticated GET over
    the same httpx pool, which keeps the TLS connection warm without spending
    tokens. Returns ``"ping"``, ``"warmup"`` (fallback: a 1-token completion,
    which DOES cost a few tokens per call) or ``"failed"``. Never raises.
    """
    client = getattr(model, "root_async_client", None)
    lister = getattr(getattr(client, "models", None), "list", None)
    try:
        if callable(lister):
            await asyncio.wait_for(lister(), timeout=8.0)
            return "ping"
        from dana_agent.brain.model import warmup_chat_model  # noqa: PLC0415

        return "warmup" if await warmup_chat_model(model) >= 0 else "failed"
    except Exception as exc:  # noqa: BLE001 - keepalive is best-effort
        log.debug("keepalive_failed", exc=str(exc)[:120])
        return "failed"


async def keepalive_loop(stack: AssembledStack, interval_s: float) -> None:
    """Re-warm the LLM connection every ``interval_s`` until cancelled."""
    model = _brain_model(stack)
    if model is None or interval_s <= 0:
        return
    log.info("keepalive_started", interval_s=interval_s)
    try:
        while True:
            await asyncio.sleep(interval_s)
            how = await keepalive_ping(model)
            log.debug("keepalive", how=how)
    except asyncio.CancelledError:
        log.info("keepalive_stopped")
        raise

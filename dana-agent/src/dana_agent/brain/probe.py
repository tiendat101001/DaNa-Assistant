"""Measure an LLM endpoint's time-to-first-token with a real streamed call."""

from __future__ import annotations

import time
from typing import Any

from dana_agent.brain.model import build_chat_model
from dana_agent.config import LLMConfig
from dana_agent.observability import get_logger

log = get_logger(__name__)


def _chunk_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            p if isinstance(p, str) else str(p.get("text", ""))
            for p in content
            if isinstance(p, str | dict)
        )
    return ""


async def probe_llm(
    cfg: LLMConfig, prompt: str = "Reply with one short friendly sentence."
) -> dict[str, Any]:
    """Stream one reply; return ``{ok, ttft_ms, total_ms, text, error}``. Never raises."""
    t0 = time.perf_counter()
    ttft: float | None = None
    parts: list[str] = []
    try:
        model = build_chat_model(cfg)
        async for chunk in model.astream(prompt):
            text = _chunk_text(chunk.content)
            if not text:
                continue
            if ttft is None:
                ttft = (time.perf_counter() - t0) * 1000
            parts.append(text)
    except Exception as exc:  # noqa: BLE001 - report, never raise
        log.warning("llm_probe_failed", provider=cfg.provider, error=str(exc)[:200])
        return {
            "ok": False,
            "ttft_ms": None,
            "total_ms": round((time.perf_counter() - t0) * 1000, 1),
            "text": "".join(parts),
            "error": f"{type(exc).__name__}: {str(exc)[:300]}",
        }
    total = (time.perf_counter() - t0) * 1000
    ok = ttft is not None
    return {
        "ok": ok,
        "ttft_ms": round(ttft, 1) if ttft is not None else None,
        "total_ms": round(total, 1),
        "text": "".join(parts),
        "error": None if ok else "no text received",
    }

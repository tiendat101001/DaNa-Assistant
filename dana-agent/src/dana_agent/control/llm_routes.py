"""LLM control routes: latency-tuned presets, a TTFT probe, and an endpoint
health check for self-hosted servers. Mounted by the owner under /api/v1."""

from __future__ import annotations

import os as _os
from typing import Any

import httpx
from fastapi import APIRouter, Body, Depends, Query, Request

from dana_agent.brain.probe import probe_llm
from dana_agent.config import AppConfig, LLMConfig
from dana_agent.control.routes import require_auth

llm_api = APIRouter(prefix="/api/v1/llm", dependencies=[Depends(require_auth)])

_MAX_TOKENS = 160


def _preset(
    id_: str,
    label: str,
    provider: str,
    model: str,
    base_url: str | None,
    notes: str,
    ttft: int,
    **knobs: Any,
) -> dict[str, Any]:
    llm: dict[str, Any] = {
        "provider": provider,
        "model": model,
        "max_tokens": _MAX_TOKENS,
        **knobs,
    }
    if base_url:
        llm["base_url"] = base_url
    return {
        "id": id_,
        "label": label,
        "provider": provider,
        "model": model,
        "base_url": base_url,
        "notes": notes,
        "expected_ttft_ms": ttft,
        "ttft_is_estimate": True,
        "patch": {"conversation": {"llm": llm}},
    }


_SG = "http://127.0.0.1:30000/v1"

PRESETS: list[dict[str, Any]] = [
    _preset(
        "together-openai-compatible",
        "Together AI (OpenAI-compatible) - hosted open models",
        "together",
        _os.environ.get("TG_MODEL_ID") or "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        None,
        "Open-weight models on Together's serverless GPUs via the OpenAI-compatible API. "
        "Reads TG_API_KEY (or TOGETHER_API_KEY), TG_BASE_URL (default "
        "https://api.together.xyz/v1) and TG_MODEL_ID. Pick a small *-Turbo / 8B-class "
        "model for the lowest first-token latency. Hidden reasoning is turned off "
        "(reasoning.enabled=false): measured live on DeepSeek-V4.1-Flash, warm TTFT "
        "~375 ms vs ~505 ms with reasoning on; the first (cold) call took ~7.5 s, "
        "which is why DaNa warms the LLM at session start.",
        375,
        extra={"extra_body": {"reasoning": {"enabled": False}}},
    ),
    _preset(
        "gemini-2.5-flash-lite",
        "Gemini 2.5 Flash-Lite (thinking off) - recommended cloud",
        "gemini",
        "gemini-2.5-flash-lite",
        None,
        "Primary cloud choice: fastest Gemini, good Vietnamese, native tool calling. "
        "Needs GEMINI_API_KEY. thinking_budget=0 removes hidden reasoning latency.",
        350,
        thinking_budget=0,
    ),
    _preset(
        "gemini-2.5-flash",
        "Gemini 2.5 Flash (thinking off)",
        "gemini",
        "gemini-2.5-flash",
        None,
        "Smarter than Flash-Lite, a little slower. Strong Vietnamese and tool calling. "
        "Needs GEMINI_API_KEY.",
        450,
        thinking_budget=0,
    ),
    _preset(
        "openai-gpt-4.1-nano",
        "OpenAI gpt-4.1-nano",
        "openai",
        "gpt-4.1-nano",
        None,
        "Very fast non-reasoning model; decent tool calling, acceptable Vietnamese. "
        "Needs OPENAI_API_KEY.",
        350,
    ),
    _preset(
        "openai-gpt-4o-mini",
        "OpenAI gpt-4o-mini",
        "openai",
        "gpt-4o-mini",
        None,
        "Reliable tool calling and good Vietnamese; slightly slower than nano. "
        "Needs OPENAI_API_KEY.",
        450,
    ),
    _preset(
        "openai-gpt-5-nano",
        "OpenAI gpt-5-nano (reasoning minimal)",
        "openai",
        "gpt-5-nano",
        None,
        "Reasoning model; reasoning_effort=minimal keeps first token fast. "
        "Good tool calling. Needs OPENAI_API_KEY.",
        600,
        reasoning_effort="minimal",
    ),
    _preset(
        "sglang-qwen2.5-1.5b",
        "SGLang Qwen2.5-1.5B-Instruct (self-hosted)",
        "sglang",
        "Qwen/Qwen2.5-1.5B-Instruct",
        _SG,
        "Smallest and fastest; weak Vietnamese, basic tool calling "
        "(--tool-call-parser qwen25). ~4 GB VRAM. Serve: scripts/serve_sglang.sh.",
        120,
    ),
    _preset(
        "sglang-qwen3-1.7b",
        "SGLang Qwen3-1.7B (self-hosted)",
        "sglang",
        "Qwen/Qwen3-1.7B",
        _SG,
        "Better multilingual (Vietnamese OK) and tool use than Qwen2.5-1.5B; "
        "thinking mode should be disabled. Serve with PARSER=qwen. ~5 GB VRAM.",
        150,
        extra={"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}},
    ),
    _preset(
        "sglang-llama-3.2-3b",
        "SGLang Llama-3.2-3B-Instruct (self-hosted)",
        "sglang",
        "meta-llama/Llama-3.2-3B-Instruct",
        _SG,
        "Good English and tool calling (PARSER=llama3); Vietnamese is mediocre. "
        "Gated HF model (needs HF_TOKEN). ~8 GB VRAM.",
        180,
    ),
]


@llm_api.get("/presets")
def presets() -> dict[str, Any]:
    return {"presets": PRESETS}


@llm_api.post("/probe")
async def probe(
    request: Request, body: dict[str, Any] | None = Body(default=None)
) -> dict[str, Any]:
    current: LLMConfig = request.app.state.controller.config().conversation.llm
    try:
        overlay = body or {}
        merged = {**current.model_dump(), **overlay}
        # model_dump() masks SecretStr; restore the real key, but only when the
        # provider is unchanged (never hand one vendor's key to another).
        same_provider = overlay.get("provider", current.provider) == current.provider
        if "api_key" not in overlay:
            merged["api_key"] = current.api_key if same_provider else None
        new_url = overlay.get("base_url", current.base_url)
        if new_url != current.base_url:
            merged["api_key"] = overlay.get("api_key")  # never send a stored key elsewhere
        if not same_provider:
            for k in ("base_url", "model"):
                if k not in overlay:
                    merged.pop(k, None)
        # Run through AppConfig so the env fallbacks (GEMINI_/OPENAI_/TG_* ...)
        # apply exactly as they do at startup.
        cfg = AppConfig.model_validate({"conversation": {"llm": merged}}).conversation.llm
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "ttft_ms": None, "total_ms": None, "text": "", "error": str(exc)}
    return await probe_llm(cfg)


@llm_api.get("/health")
async def health(base_url: str = Query(min_length=1)) -> dict[str, Any]:
    url = base_url.rstrip("/") + "/models"
    try:
        async with httpx.AsyncClient(timeout=3.0, follow_redirects=False) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            data = resp.json()
        models = [str(m.get("id")) for m in data.get("data", []) if isinstance(m, dict)]
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "models": [], "error": f"{type(exc).__name__}: {exc}"[:300]}
    return {"ok": True, "models": models, "error": None}

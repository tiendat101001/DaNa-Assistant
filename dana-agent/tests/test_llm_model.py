"""LLM builder param mapping (init_chat_model is monkeypatched; offline)."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import SecretStr

from dana_agent.brain import model as model_mod
from dana_agent.brain.model import MockChatModel, build_chat_model, warmup_chat_model
from dana_agent.config import LLMConfig


@pytest.fixture
def captured(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    box: dict[str, Any] = {}

    def fake_init(**kwargs: Any) -> MockChatModel:
        box.update(kwargs)
        return MockChatModel()

    import langchain.chat_models as cm

    monkeypatch.setattr(cm, "init_chat_model", fake_init)
    return box


def test_sglang_defaults(captured: dict[str, Any]) -> None:
    build_chat_model(LLMConfig(provider="sglang", model="Qwen/Qwen3-1.7B"))
    assert captured["model_provider"] == "openai"
    assert captured["base_url"] == "http://127.0.0.1:30000/v1"
    assert captured["api_key"] == "EMPTY"


@pytest.mark.parametrize(
    ("provider", "url"),
    [
        ("vllm", "http://127.0.0.1:8000/v1"),
        ("ollama", "http://127.0.0.1:11434/v1"),
        ("lmstudio", "http://127.0.0.1:1234/v1"),
    ],
)
def test_other_self_hosted_defaults(captured: dict[str, Any], provider: str, url: str) -> None:
    build_chat_model(LLMConfig(provider=provider, model="m"))
    assert captured["base_url"] == url
    assert captured["api_key"] == "EMPTY"


def test_explicit_base_url_and_key_win(captured: dict[str, Any]) -> None:
    build_chat_model(
        LLMConfig(
            provider="sglang", model="m", base_url="http://10.0.0.5:1/v1", api_key=SecretStr("k")
        )
    )
    assert captured["base_url"] == "http://10.0.0.5:1/v1"
    assert captured["api_key"] == "k"


def test_gemini_params(captured: dict[str, Any]) -> None:
    build_chat_model(
        LLMConfig(
            provider="gemini",
            model="gemini-2.5-flash-lite",
            max_tokens=160,
            thinking_budget=0,
            reasoning_effort="minimal",
            timeout_s=5,
        )
    )
    assert captured["model_provider"] == "google_genai"
    assert captured["thinking_budget"] == 0
    assert captured["max_tokens"] == 160
    assert captured["timeout"] == 5
    assert "reasoning_effort" not in captured
    assert "base_url" not in captured


def test_openai_params(captured: dict[str, Any]) -> None:
    build_chat_model(
        LLMConfig(
            provider="openai",
            model="gpt-5-nano",
            reasoning_effort="minimal",
            thinking_budget=0,
            max_tokens=100,
        )
    )
    assert captured["reasoning_effort"] == "minimal"
    assert "thinking_budget" not in captured
    assert captured["max_tokens"] == 100
    assert "api_key" not in captured  # cloud OpenAI: no placeholder


def test_unset_knobs_not_sent(captured: dict[str, Any]) -> None:
    build_chat_model(LLMConfig(provider="openai", model="gpt-4o-mini"))
    for k in ("max_tokens", "timeout", "reasoning_effort", "thinking_budget"):
        assert k not in captured


def test_extra_merged_last(captured: dict[str, Any]) -> None:
    build_chat_model(
        LLMConfig(provider="sglang", model="m", max_tokens=10, extra={"max_tokens": 5, "x": 1})
    )
    assert captured["max_tokens"] == 5
    assert captured["x"] == 1


@pytest.mark.parametrize("provider", ["openai", "gemini", "sglang"])
def test_real_clients_construct_with_params(provider: str) -> None:
    """The kwargs we pass are accepted by the installed LangChain classes."""
    cfg = LLMConfig(
        provider=provider,
        model="gemini-2.5-flash-lite" if provider == "gemini" else "gpt-5-nano",
        api_key=SecretStr("k") if provider != "sglang" else None,
        max_tokens=50,
        timeout_s=3,
        thinking_budget=0 if provider == "gemini" else None,
        reasoning_effort="minimal" if provider == "openai" else None,
    )
    assert build_chat_model(cfg) is not None


async def test_warmup_mock_returns_ms() -> None:
    assert await warmup_chat_model(MockChatModel()) >= 0.0


async def test_warmup_never_raises() -> None:
    class Boom(MockChatModel):
        async def ainvoke(self, *a: Any, **k: Any) -> Any:
            raise RuntimeError("down")

    assert await warmup_chat_model(Boom()) == -1.0


def test_module_exports() -> None:
    assert model_mod.warmup_chat_model is warmup_chat_model


# --- Together AI (OpenAI-compatible cloud) ----------------------------------


def test_together_maps_tg_env_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    from dana_agent.config import AppConfig

    monkeypatch.setenv("TG_API_KEY", "tg-test-key")
    monkeypatch.setenv("TG_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("TG_MODEL_ID", "org/some-model")
    cfg = AppConfig.model_validate({"conversation": {"llm": {"provider": "together"}}})
    llm = cfg.conversation.llm
    assert llm.api_key is not None and llm.api_key.get_secret_value() == "tg-test-key"
    assert llm.base_url == "https://example.invalid/v1"
    assert llm.model == "org/some-model"


def test_together_default_base_url_and_not_self_hosted(monkeypatch: pytest.MonkeyPatch) -> None:
    from dana_agent.brain import model as model_mod
    from dana_agent.brain.providers import SELF_HOSTED_PROVIDERS
    from dana_agent.config import LLMConfig

    captured: dict[str, object] = {}

    def fake_init_chat_model(**kwargs: object) -> object:
        captured.update(kwargs)
        return object()

    import langchain.chat_models as lc

    monkeypatch.setattr(lc, "init_chat_model", fake_init_chat_model)
    model_mod.build_chat_model(LLMConfig(provider="together", model="m", api_key=SecretStr("k")))
    assert captured["model_provider"] == "openai"
    assert captured["base_url"] == "https://api.together.xyz/v1"
    assert "together" not in SELF_HOSTED_PROVIDERS  # cloud: local_only must reject it


def test_together_env_values_tolerate_crlf(monkeypatch: pytest.MonkeyPatch) -> None:
    """A Windows-edited env file `source`d into the shell leaves trailing \\r."""
    from dana_agent.config import AppConfig

    monkeypatch.setenv("TG_API_KEY", "tg-key\r")
    monkeypatch.setenv("TG_BASE_URL", "https://api.together.ai/v1\r")
    monkeypatch.setenv("TG_MODEL_ID", "org/model\r")
    llm = AppConfig.model_validate(
        {"conversation": {"llm": {"provider": "together"}}}
    ).conversation.llm
    assert llm.api_key is not None and llm.api_key.get_secret_value() == "tg-key"
    assert llm.base_url == "https://api.together.ai/v1"
    assert llm.model == "org/model"

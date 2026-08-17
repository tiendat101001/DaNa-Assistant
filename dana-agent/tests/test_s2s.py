"""Speech-to-speech path — proves the cascade-or-S2S abstraction end to end.

Selection is config-only (``conversation.mode = "s2s"``): the same
``build_stack`` / ``AssembledStack`` carries an S2S provider instead of the
cascade, ``ConversationSession`` routes to the S2S pipeline builder, and a
real-ish reference provider (openai_realtime) validates offline and is refused
under local_only.
"""

from __future__ import annotations

import pytest

from dana_agent.app import build_stack
from dana_agent.capabilities import enforce_local_only
from dana_agent.conversation import session as session_mod
from dana_agent.conversation.pipeline import build_s2s_pipeline
from dana_agent.conversation.processors import S2SProcessor
from dana_agent.conversation.session import ConversationSession
from dana_agent.errors import ConfigError
from dana_agent.providers.registry import build_s2s, known
from tests.support import clean_config


def test_reference_provider_registered() -> None:
    assert "openai_realtime" in known("s2s")


def test_reference_provider_capabilities() -> None:
    cfg = clean_config(conversation={"mode": "s2s", "s2s": {"provider": "openai_realtime"}})
    caps = build_s2s(cfg.conversation.s2s).capabilities
    assert caps.streaming is True
    assert caps.self_hosted is False  # cloud → refused under local_only
    assert caps.latency_class == "realtime"


async def test_build_stack_s2s_mock() -> None:
    cfg = clean_config(conversation={"mode": "s2s"})
    stack = await build_stack(cfg)
    assert stack.s2s is not None
    assert stack.stt is None and stack.tts is None and stack.brain is None
    await stack.aclose()


async def test_build_stack_s2s_openai_realtime_validates() -> None:
    # Construction + startup validation must pass offline (no SDK call is made).
    cfg = clean_config(conversation={"mode": "s2s", "s2s": {"provider": "openai_realtime"}})
    stack = await build_stack(cfg)
    assert stack.s2s is not None
    await stack.aclose()


def test_reference_provider_refused_under_local_only() -> None:
    cfg = clean_config(
        privacy={"local_only": True},
        conversation={"mode": "s2s", "s2s": {"provider": "openai_realtime"}},
    )
    with pytest.raises(ConfigError, match="self-hosted"):
        enforce_local_only(cfg)


def test_s2s_processor_wraps_provider() -> None:
    cfg = clean_config(conversation={"mode": "s2s"})
    provider = build_s2s(cfg.conversation.s2s)
    proc = S2SProcessor(provider)
    assert isinstance(proc, S2SProcessor)


def test_build_s2s_pipeline_requires_s2s_provider() -> None:
    # A cascade stack has s2s=None → the s2s builder refuses before any audio I/O.
    from dana_agent.app import AssembledStack

    stack = AssembledStack.__new__(AssembledStack)  # avoid real providers/audio
    object.__setattr__(stack, "s2s", None)
    with pytest.raises(ValueError, match="S2S provider is not set"):
        build_s2s_pipeline(clean_config(), stack)


async def test_session_routes_to_s2s_builder(monkeypatch: pytest.MonkeyPatch) -> None:
    """s2s mode must call build_s2s_pipeline, never the cascade builder."""
    called: dict[str, bool] = {}

    def fake_s2s(*_a: object, **_k: object) -> None:
        called["s2s"] = True
        raise RuntimeError("stop-here")

    def fake_cascade(*_a: object, **_k: object) -> None:
        called["cascade"] = True
        raise RuntimeError("wrong-branch")

    monkeypatch.setattr(session_mod, "build_s2s_pipeline", fake_s2s)
    monkeypatch.setattr(session_mod, "build_cascade_pipeline", fake_cascade)

    cfg = clean_config(conversation={"mode": "s2s"})
    stack = await build_stack(cfg)
    try:
        sess = ConversationSession(cfg, stack)
        with pytest.raises(RuntimeError, match="stop-here"):
            await sess.run()
        assert called == {"s2s": True}
    finally:
        await stack.aclose()


async def test_session_routes_to_cascade_builder(monkeypatch: pytest.MonkeyPatch) -> None:
    called: dict[str, bool] = {}

    def fake_cascade(*_a: object, **_k: object) -> None:
        called["cascade"] = True
        raise RuntimeError("stop-here")

    monkeypatch.setattr(session_mod, "build_cascade_pipeline", fake_cascade)

    cfg = clean_config()  # default cascade
    stack = await build_stack(cfg)
    try:
        sess = ConversationSession(cfg, stack)
        with pytest.raises(RuntimeError, match="stop-here"):
            await sess.run()
        assert called == {"cascade": True}
    finally:
        await stack.aclose()

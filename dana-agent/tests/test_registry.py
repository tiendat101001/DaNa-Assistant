"""Registry + factory mechanics and error paths."""

from __future__ import annotations

import pytest

from dana_agent.errors import ConfigError
from dana_agent.providers import registry
from dana_agent.providers.base import Provider


def test_every_capability_has_a_mock() -> None:
    for capability in registry.CAPABILITIES:
        assert "mock" in registry.known(capability), capability


def test_get_class_unknown_provider_lists_known() -> None:
    with pytest.raises(ConfigError) as exc:
        registry.get_class("llm", "does-not-exist")
    msg = str(exc.value)
    assert "unknown llm provider" in msg
    assert "mock" in msg  # actionable: names what IS available


def test_get_class_returns_provider_subclass() -> None:
    cls = registry.get_class("stt", "mock")
    assert issubclass(cls, Provider)


def test_duplicate_registration_rejected(registry_snapshot: None) -> None:
    @registry.register("llm", "dup")
    class _A(Provider):  # pragma: no cover - never constructed
        @property
        def capabilities(self):  # type: ignore[no-untyped-def]
            ...

    with pytest.raises(ConfigError, match="duplicate provider registration"):

        @registry.register("llm", "dup")
        class _B(Provider):  # pragma: no cover
            @property
            def capabilities(self):  # type: ignore[no-untyped-def]
                ...


def test_reregistering_same_class_is_idempotent(registry_snapshot: None) -> None:
    @registry.register("tts", "same")
    class _C(Provider):  # pragma: no cover
        @property
        def capabilities(self):  # type: ignore[no-untyped-def]
            ...

    # decorating the same class again must not raise
    registry.register("tts", "same")(_C)

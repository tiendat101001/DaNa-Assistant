"""Shared fixtures. The suite runs with no network, no secrets, no microphone —
mock providers are the default backend (python-service-scaffold skill)."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from typing import TypeVar

import pytest

from dana_agent.config import AppConfig
from dana_agent.providers import registry

T = TypeVar("T")


async def async_iter(items: list[T]) -> AsyncIterator[T]:
    """Wrap a list as an async iterator for streaming provider inputs."""
    for item in items:
        yield item


async def collect(stream: AsyncIterator[T]) -> list[T]:
    """Drain an async iterator into a list."""
    return [item async for item in stream]


@pytest.fixture
def config() -> AppConfig:
    """Default config: all-mock cascade stack, English."""
    return AppConfig()


@pytest.fixture
def registry_snapshot() -> Iterator[None]:
    """Snapshot and restore the provider registry so a test that registers a
    throwaway provider does not leak it into other tests."""
    saved = {cap: dict(names) for cap, names in registry._REGISTRY.items()}
    try:
        yield
    finally:
        registry._REGISTRY.clear()
        registry._REGISTRY.update(saved)

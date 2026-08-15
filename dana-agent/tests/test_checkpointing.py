"""Brain memory checkpointer: memory vs sqlite (in-memory and file-durable)."""

from __future__ import annotations

from pathlib import Path

from langgraph.checkpoint.memory import InMemorySaver

from dana_agent.actions.gate import ActionGate
from dana_agent.brain import Brain, build_chat_model, open_checkpointer
from dana_agent.config import ActionsConfig, LLMConfig, MemoryConfig
from dana_agent.providers.registry import build_actions
from tests.conftest import collect

SYSTEM = "You are DaNa."


def make_brain(checkpointer: object) -> Brain:
    gate = ActionGate(build_actions(ActionsConfig()), policy="standard")
    return Brain(
        build_chat_model(LLMConfig()),
        gate,
        system_prompt=SYSTEM,
        checkpointer=checkpointer,
    )


async def test_memory_backend_returns_in_memory_saver() -> None:
    saver, close = await open_checkpointer(MemoryConfig())
    assert isinstance(saver, InMemorySaver)
    await close()


async def test_sqlite_in_memory_persists_within_process() -> None:
    saver, close = await open_checkpointer(MemoryConfig(backend="sqlite", sqlite_path=":memory:"))
    try:
        brain = make_brain(saver)
        await collect(brain.run_turn("first", thread_id="t1"))
        await collect(brain.run_turn("second", thread_id="t1"))
        assert await brain.user_messages("t1") == ["first", "second"]
    finally:
        await close()


async def test_sqlite_file_is_durable_across_reopen(tmp_path: Path) -> None:
    db = str(tmp_path / "brain.sqlite")
    cfg = MemoryConfig(backend="sqlite", sqlite_path=db)

    saver1, close1 = await open_checkpointer(cfg)
    brain1 = make_brain(saver1)
    await collect(brain1.run_turn("remember me", thread_id="t1"))
    await close1()

    # A fresh connection to the same file must still see the earlier turn.
    saver2, close2 = await open_checkpointer(cfg)
    brain2 = make_brain(saver2)
    try:
        assert await brain2.user_messages("t1") == ["remember me"]
    finally:
        await close2()


async def test_sqlite_file_backend_creates_parent_dir(tmp_path: Path) -> None:
    db = str(tmp_path / "nested" / "dir" / "brain.sqlite")
    saver, close = await open_checkpointer(MemoryConfig(backend="sqlite", sqlite_path=db))
    await close()
    assert Path(db).parent.is_dir()

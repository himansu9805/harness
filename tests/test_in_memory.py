"""Tests for the InMemoryMemory conversation store."""

import pytest

from harness.agent.memory.in_memory import InMemoryMemory
from harness.schemas.chat import ChatMessage


@pytest.fixture
def memory() -> InMemoryMemory:
    return InMemoryMemory()


async def test_empty_history_for_unknown_session(memory):
    assert await memory.get_history("nope") == []


async def test_append_and_get_preserves_order(memory):
    await memory.append("s1", ChatMessage(role="user", content="first"))
    await memory.append("s1", ChatMessage(role="assistant", content="second"))
    history = await memory.get_history("s1")
    assert [m.content for m in history] == ["first", "second"]


async def test_sessions_are_isolated(memory):
    await memory.append("a", ChatMessage(role="user", content="x"))
    assert await memory.get_history("b") == []


async def test_get_history_returns_copy(memory):
    await memory.append("s1", ChatMessage(role="user", content="x"))
    history = await memory.get_history("s1")
    history.append(ChatMessage(role="user", content="mutated"))
    assert len(await memory.get_history("s1")) == 1


async def test_clear_removes_history(memory):
    await memory.append("s1", ChatMessage(role="user", content="x"))
    await memory.clear("s1")
    assert await memory.get_history("s1") == []


async def test_clear_unknown_session_is_noop(memory):
    await memory.clear("ghost")  # should not raise

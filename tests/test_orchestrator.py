"""Tests for the AgentOrchestrator plan-act loop."""

import pytest

from harness.agent.memory.in_memory import InMemoryMemory
from harness.agent.orchestrator.orchestrator import (
    AgentOrchestrator,
    _preview,
)
from harness.schemas.chat import ChatMessage
from harness.tools.registry import ToolRegistry
from tests.conftest import FakeMCPRegistry, ScriptedPlanner


@pytest.fixture
def tool_registry(skill_registry, echo_tool):
    reg = ToolRegistry(mcp_registry=FakeMCPRegistry(), skill_registry=skill_registry)
    reg.register_builtin_tool(echo_tool)
    return reg


def _orchestrator(settings, tool_registry, steps, memory=None):
    return AgentOrchestrator(
        settings=settings,
        tool_registry=tool_registry,
        planner=ScriptedPlanner(steps),
        memory=memory or InMemoryMemory(),
    )


async def test_immediate_response(settings, tool_registry):
    orch = _orchestrator(
        settings, tool_registry, [{"action": "respond", "content": "hello"}]
    )
    resp = await orch.run_turn(
        session_id=None, messages=[ChatMessage(role="user", content="hi")]
    )
    assert resp.message.role == "assistant"
    assert resp.message.content == "hello"
    assert resp.tool_calls == []
    assert resp.session_id  # a new id was generated


async def test_provided_session_id_is_preserved(settings, tool_registry):
    orch = _orchestrator(
        settings, tool_registry, [{"action": "respond", "content": "hi"}]
    )
    resp = await orch.run_turn(
        session_id="fixed-session",
        messages=[ChatMessage(role="user", content="hi")],
    )
    assert resp.session_id == "fixed-session"


async def test_tool_call_then_respond(settings, tool_registry):
    steps = [
        {
            "action": "call_tools",
            "assistant_content": "let me run that",
            "tool_calls": [
                {"id": "c1", "name": "echo", "arguments": {"value": "hi"}}
            ],
        },
        {"action": "respond", "content": "done"},
    ]
    orch = _orchestrator(settings, tool_registry, steps)
    resp = await orch.run_turn(
        session_id="s", messages=[ChatMessage(role="user", content="go")]
    )
    assert resp.message.content == "done"
    assert len(resp.tool_calls) == 1
    call = resp.tool_calls[0]
    assert call.name == "echo"
    assert call.source == "builtin"
    assert call.arguments == {"value": "hi"}


async def test_string_arguments_are_json_parsed(settings, tool_registry):
    steps = [
        {
            "action": "call_tools",
            "assistant_content": None,
            "tool_calls": [
                {"id": "c1", "name": "echo", "arguments": '{"value": "x"}'}
            ],
        },
        {"action": "respond", "content": "ok"},
    ]
    orch = _orchestrator(settings, tool_registry, steps)
    resp = await orch.run_turn(session_id="s", messages=[])
    assert resp.tool_calls[0].arguments == {"value": "x"}


async def test_history_is_persisted(settings, tool_registry):
    memory = InMemoryMemory()
    orch = _orchestrator(
        settings, tool_registry, [{"action": "respond", "content": "hi"}], memory
    )
    await orch.run_turn(
        session_id="s", messages=[ChatMessage(role="user", content="hi")]
    )
    history = await memory.get_history("s")
    # user message + assistant reply
    assert [m.role for m in history] == ["user", "assistant"]


async def test_tool_messages_added_to_history(settings, tool_registry):
    memory = InMemoryMemory()
    steps = [
        {
            "action": "call_tools",
            "assistant_content": None,
            "tool_calls": [{"id": "c1", "name": "echo", "arguments": {}}],
        },
        {"action": "respond", "content": "done"},
    ]
    orch = _orchestrator(settings, tool_registry, steps, memory)
    await orch.run_turn(
        session_id="s", messages=[ChatMessage(role="user", content="go")]
    )
    roles = [m.role for m in await memory.get_history("s")]
    assert roles == ["user", "assistant", "tool", "assistant"]


def _distinct_call(i):
    return {
        "action": "call_tools",
        "assistant_content": None,
        "tool_calls": [{"id": f"c{i}", "name": "echo", "arguments": {"i": i}}],
    }


def _same_call():
    return {
        "action": "call_tools",
        "assistant_content": None,
        "tool_calls": [{"id": "c", "name": "echo", "arguments": {"same": True}}],
    }


async def test_budget_can_be_overridden_per_turn(settings, tool_registry):
    # 3 distinct tool calls then a forced summary once the budget is spent.
    steps = [_distinct_call(i) for i in range(3)]
    steps.append({"action": "respond", "content": "here is what I found"})
    orch = _orchestrator(settings, tool_registry, steps)
    resp = await orch.run_turn(session_id="s", messages=[], max_iterations=3)
    assert resp.message.content == "here is what I found"
    assert len(resp.tool_calls) == 3


async def test_stall_detection_ends_turn_early(settings, tool_registry):
    # Same call repeated: should stop after max_repeated_tool_calls (default 3),
    # having executed it only twice, then summarise.
    steps = [_same_call(), _same_call(), _same_call()]
    steps.append({"action": "respond", "content": "stopping, no progress"})
    orch = _orchestrator(settings, tool_registry, steps)
    resp = await orch.run_turn(session_id="s", messages=[], max_iterations=20)
    assert resp.message.content == "stopping, no progress"
    assert len(resp.tool_calls) == 2


async def test_distinct_calls_do_not_trip_stall_guard(settings, tool_registry):
    # Many different calls in a row must not be mistaken for a stall.
    steps = [_distinct_call(i) for i in range(5)]
    steps.append({"action": "respond", "content": "done"})
    orch = _orchestrator(settings, tool_registry, steps)
    resp = await orch.run_turn(session_id="s", messages=[], max_iterations=10)
    assert resp.message.content == "done"
    assert len(resp.tool_calls) == 5


async def test_forced_summary_falls_back_when_model_wont_answer(
    settings, tool_registry
):
    # Budget spent, and the forced final call still asks for tools (no answer):
    # fall back to the static message.
    steps = [_distinct_call(0), _distinct_call(1)]
    orch = _orchestrator(settings, tool_registry, steps)
    resp = await orch.run_turn(session_id="s", messages=[], max_iterations=1)
    assert "wasn't able to fully complete" in resp.message.content
    assert len(resp.tool_calls) == 1


def test_tool_call_signature_normalises_dict_and_json_string():
    dict_sig = AgentOrchestrator._tool_call_signature(
        [{"name": "echo", "arguments": {"a": 1, "b": 2}}]
    )
    json_sig = AgentOrchestrator._tool_call_signature(
        [{"name": "echo", "arguments": '{"b": 2, "a": 1}'}]
    )
    assert dict_sig == json_sig


def test_tool_call_signature_is_order_independent():
    a = AgentOrchestrator._tool_call_signature(
        [{"name": "x", "arguments": {}}, {"name": "y", "arguments": {}}]
    )
    b = AgentOrchestrator._tool_call_signature(
        [{"name": "y", "arguments": {}}, {"name": "x", "arguments": {}}]
    )
    assert a == b


def test_to_openai_tool_schema_wraps_definitions():
    from harness.schemas.tool import ToolDefinition

    defs = [
        ToolDefinition(
            name="echo",
            description="d",
            parameters={"type": "object"},
            source="builtin",
        )
    ]
    schema = AgentOrchestrator._to_openai_tool_schema(defs)
    assert schema[0]["type"] == "function"
    assert schema[0]["function"]["name"] == "echo"
    assert schema[0]["function"]["parameters"] == {"type": "object"}


def test_to_openai_tool_schema_defaults_empty_parameters():
    from harness.schemas.tool import ToolDefinition

    defs = [ToolDefinition(name="x", description="d", source="skill")]
    schema = AgentOrchestrator._to_openai_tool_schema(defs)
    assert schema[0]["function"]["parameters"] == {"type": "object", "properties": {}}


def test_preview_truncates_long_text():
    out = _preview("a " * 500, limit=10)
    assert out.endswith("chars)")
    assert "…" in out


def test_preview_collapses_whitespace():
    assert _preview("a\n\n  b\tc") == "a b c"

"""Tests for the unified ToolRegistry over builtin, MCP and skill tools."""

import pytest

from harness.tools.registry import ToolRegistry
from tests.conftest import FakeMCPRegistry, KeywordSkill


@pytest.fixture
def mcp_registry():
    return FakeMCPRegistry(
        tools=[
            {
                "name": "remote_fetch",
                "description": "Fetch a URL",
                "input_schema": {"type": "object"},
                "server_label": "web",
            }
        ]
    )


@pytest.fixture
def registry(mcp_registry, skill_registry, echo_tool):
    skill_registry.register(KeywordSkill(name="summarize", keyword="summ"))
    reg = ToolRegistry(mcp_registry=mcp_registry, skill_registry=skill_registry)
    reg.register_builtin_tool(echo_tool)
    return reg


async def test_list_tools_merges_all_sources(registry):
    defs = await registry.list_tools()
    by_name = {d.name: d for d in defs}
    assert by_name["echo"].source == "builtin"
    assert by_name["remote_fetch"].source == "mcp"
    assert by_name["remote_fetch"].server_label == "web"
    assert by_name["summarize"].source == "skill"


async def test_builtin_call_returns_output(registry):
    result = await registry.call(name="echo", arguments={"value": "hi"})
    assert result.is_error is False
    assert result.output == {"echoed": {"value": "hi"}}
    assert result.duration_ms is not None and result.duration_ms >= 0


async def test_builtin_takes_precedence_over_mcp(registry, mcp_registry):
    await registry.call(name="echo", arguments={}, server_label="web")
    assert mcp_registry.calls == []


async def test_mcp_call_routes_by_server_label(registry, mcp_registry):
    result = await registry.call(
        name="remote_fetch", arguments={"url": "x"}, server_label="web"
    )
    assert result.is_error is False
    assert mcp_registry.calls == [("web", "remote_fetch", {"url": "x"})]


async def test_unresolvable_tool_returns_error_result(registry):
    result = await registry.call(name="ghost", arguments={})
    assert result.is_error is True
    assert "ghost" in result.output


async def test_tool_exception_is_captured_not_raised(mcp_registry, skill_registry,
                                                     boom_tool):
    reg = ToolRegistry(mcp_registry=mcp_registry, skill_registry=skill_registry)
    reg.register_builtin_tool(boom_tool)
    result = await reg.call(name="boom", arguments={})
    assert result.is_error is True
    assert "kaboom" in result.output
    assert result.duration_ms is not None

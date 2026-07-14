"""Tests for SessionMCPClient's transport-agnostic session operations."""

from types import SimpleNamespace

import pytest

from harness.core.exceptions import MCPConnectionError, ToolExecutionError
from harness.mcp.clients.session import SessionMCPClient


class _FakeSession:
    def __init__(self, *, tools=None, call_result=None, call_raises=None):
        self._tools = tools or []
        self._call_result = call_result
        self._call_raises = call_raises
        self.calls = []

    async def list_tools(self):
        return SimpleNamespace(tools=self._tools)

    async def call_tool(self, name, arguments):
        self.calls.append((name, arguments))
        if self._call_raises is not None:
            raise self._call_raises
        return self._call_result


def _tool(name, description=None, schema=None):
    return SimpleNamespace(name=name, description=description, inputSchema=schema)


def _result(*, is_error=False, structured=None, content=None):
    return SimpleNamespace(
        isError=is_error, structuredContent=structured, content=content or []
    )


class _ConcreteSessionClient(SessionMCPClient):
    """Minimal concrete subclass so the ABC can be instantiated in tests."""

    async def connect(self) -> None:  # pragma: no cover - not exercised here
        raise NotImplementedError


@pytest.fixture
def client():
    return _ConcreteSessionClient(server_label="srv")


def test_require_session_raises_when_not_connected(client):
    with pytest.raises(MCPConnectionError):
        client._require_session()


async def test_list_tools_maps_fields(client):
    client._session = _FakeSession(
        tools=[_tool("fetch", "Fetch a URL", {"type": "object"})]
    )
    tools = await client.list_tools()
    assert tools == [
        {
            "name": "fetch",
            "description": "Fetch a URL",
            "input_schema": {"type": "object"},
        }
    ]


async def test_list_tools_defaults_missing_fields(client):
    client._session = _FakeSession(tools=[_tool("bare", None, None)])
    tools = await client.list_tools()
    assert tools[0]["description"] == ""
    assert tools[0]["input_schema"] == {}


async def test_call_tool_prefers_structured_content(client):
    client._session = _FakeSession(
        call_result=_result(structured={"answer": 42})
    )
    assert await client.call_tool("t", {}) == {"answer": 42}


async def test_call_tool_concatenates_text_blocks(client):
    blocks = [
        SimpleNamespace(text="hello "),
        SimpleNamespace(text="world"),
        SimpleNamespace(),  # no text attribute -> skipped
    ]
    client._session = _FakeSession(call_result=_result(content=blocks))
    assert await client.call_tool("t", {}) == "hello world"


async def test_call_tool_error_result_raises(client):
    client._session = _FakeSession(call_result=_result(is_error=True, content="bad"))
    with pytest.raises(ToolExecutionError):
        await client.call_tool("t", {})


async def test_call_tool_wraps_session_exception(client):
    client._session = _FakeSession(call_raises=RuntimeError("network"))
    with pytest.raises(ToolExecutionError) as excinfo:
        await client.call_tool("t", {})
    assert "network" in str(excinfo.value)


async def test_disconnect_without_connection_is_safe(client):
    await client.disconnect()  # no exit stack; must not raise
    assert client._session is None

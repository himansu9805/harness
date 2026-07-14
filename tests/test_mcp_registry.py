"""Tests for MCPRegistry config loading, client building and routing."""

from typing import Any

import pytest

from harness.core.config import Settings
from harness.core.exceptions import MCPConnectionError
from harness.mcp.clients.http_client import HTTPMCPClient
from harness.mcp.clients.stdio_client import StdioMCPClient
from harness.mcp.registry.registry import MCPRegistry


@pytest.fixture
def registry(settings):
    return MCPRegistry(settings=settings)


def _registry_with_config(tmp_path, yaml_text) -> MCPRegistry:
    cfg = tmp_path / "mcp.yaml"
    cfg.write_text(yaml_text)
    settings = Settings(_env_file=None, mcp_servers_config_path=str(cfg))
    return MCPRegistry(settings=settings)


def test_load_configs_missing_file_returns_empty(tmp_path):
    settings = Settings(
        _env_file=None, mcp_servers_config_path=str(tmp_path / "absent.yaml")
    )
    assert MCPRegistry(settings=settings)._load_server_configs() == {}


def test_load_configs_reads_servers(tmp_path):
    reg = _registry_with_config(
        tmp_path,
        "mcpServers:\n"
        "  playwright:\n"
        "    command: npx\n"
        "    args: ['@playwright/mcp@latest']\n",
    )
    configs = reg._load_server_configs()
    assert configs["playwright"]["command"] == "npx"


def test_load_configs_without_mcpservers_key(tmp_path):
    reg = _registry_with_config(tmp_path, "other: {}\n")
    assert reg._load_server_configs() == {}


def test_build_stdio_client(registry):
    client = registry._build_client(
        "pw", {"command": "npx", "args": ["x"], "cwd": "/tmp"}
    )
    assert isinstance(client, StdioMCPClient)
    assert client.server_label == "pw"
    assert client.command == "npx"
    assert client.args == ["x"]


def test_build_http_client(registry):
    client = registry._build_client("web", {"url": "http://localhost:8080/mcp"})
    assert isinstance(client, HTTPMCPClient)
    assert client.server_url == "http://localhost:8080/mcp"


def test_build_client_requires_transport(registry):
    with pytest.raises(MCPConnectionError):
        registry._build_client("bad", {"foo": "bar"})


class _FakeClient:
    def __init__(self, label, *, fail=False, tools=None):
        self.server_label = label
        self._fail = fail
        self._tools = tools or []
        self.disconnected = False
        self.calls = []

    async def connect(self):
        if self._fail:
            raise MCPConnectionError(message="nope")

    async def disconnect(self):
        self.disconnected = True

    async def list_tools(self):
        return list(self._tools)

    async def call_tool(self, name, arguments):
        self.calls.append((name, arguments))
        return {"ok": name}


async def test_initialize_skips_failing_servers(registry, monkeypatch):
    good = _FakeClient("good")
    bad = _FakeClient("bad", fail=True)
    monkeypatch.setattr(
        registry,
        "_load_server_configs",
        lambda: {"good": {"command": "x"}, "bad": {"command": "y"}},
    )
    monkeypatch.setattr(
        registry,
        "_build_client",
        lambda label, cfg: good if label == "good" else bad,
    )
    await registry.initialize()
    assert registry.get_client("good") is good
    with pytest.raises(KeyError):
        registry.get_client("bad")


async def test_shutdown_disconnects_and_clears(registry):
    client = _FakeClient("a")
    registry._clients["a"] = client
    await registry.shutdown()
    assert client.disconnected is True
    assert registry._clients == {}


async def test_list_all_tools_tags_server_label(registry):
    registry._clients["web"] = _FakeClient(
        "web", tools=[{"name": "fetch", "description": "d"}]
    )
    tools = await registry.list_all_tools()
    assert tools == [{"name": "fetch", "description": "d", "server_label": "web"}]


async def test_call_tool_routes_to_client(registry):
    client = _FakeClient("web")
    registry._clients["web"] = client
    result: Any = await registry.call_tool("web", "fetch", {"url": "x"})
    assert result == {"ok": "fetch"}
    assert client.calls == [("fetch", {"url": "x"})]

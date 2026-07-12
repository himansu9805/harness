"""Registry that connects to and multiplexes configured MCP servers."""

from typing import Any

import yaml

from harness.core.config import Settings
from harness.core.exceptions import MCPConnectionError
from harness.core.logging import get_logger
from harness.mcp.clients.base import BaseMCPClient
from harness.mcp.clients.http_client import HTTPMCPClient
from harness.mcp.clients.stdio_client import StdioMCPClient

logger = get_logger(__name__)


class MCPRegistry:
    """Central registry that owns all configured MCP servers."""

    def __init__(self, settings: Settings):
        """Store ``settings`` and start with no connected clients."""
        self._settings = settings
        self._clients: dict[str, BaseMCPClient] = {}

    def _load_server_configs(self) -> dict[str, dict[str, Any]]:
        """Read the ``mcpServers`` mapping from the YAML config.

        Supports the standard MCP config shape::

            mcpServers:
              playwright:
                command: npx
                args: ["@playwright/mcp@latest"]
              remote:
                url: http://localhost:8080/mcp

        Returns:
            A mapping of server label to its config, or ``{}`` when the file is
            missing or defines no servers.
        """
        try:
            with open(
                self._settings.mcp_servers_config_path, "r", encoding="utf-8"
            ) as fh:
                data = yaml.safe_load(fh) or {}
        except FileNotFoundError:
            logger.warning(
                "No MCP server config found at %s",
                self._settings.mcp_servers_config_path,
            )
            return {}
        return data.get("mcpServers", {}) or {}

    def _build_client(self, label: str, cfg: dict[str, Any]) -> BaseMCPClient:
        """Construct the right client for ``cfg`` based on its transport.

        A ``command`` selects the stdio transport (a spawned subprocess); a
        ``url`` selects streamable HTTP/SSE.

        Raises:
            MCPConnectionError: If the config specifies neither transport.
        """
        timeout = self._settings.mcp_request_timeout_seconds
        if "command" in cfg:
            return StdioMCPClient(
                server_label=label,
                command=cfg["command"],
                args=cfg.get("args", []),
                env=cfg.get("env"),
                cwd=cfg.get("cwd"),
                timeout_seconds=timeout,
            )
        if "url" in cfg:
            return HTTPMCPClient(
                server_label=label,
                server_url=cfg["url"],
                timeout_seconds=timeout,
            )
        raise MCPConnectionError(
            message=(
                f"MCP server '{label}' must define either 'command' (stdio) "
                f"or 'url' (http)"
            )
        )

    async def initialize(self) -> None:
        """Connect to every configured server, skipping any that fail."""
        for label, cfg in self._load_server_configs().items():
            try:
                client = self._build_client(label, cfg)
                await client.connect()
            except MCPConnectionError:
                logger.exception("Skipping MCP server '%s': connection failed", label)
                continue
            except Exception:
                logger.exception(
                    "Skipping MCP server '%s': unexpected error during connect",
                    label,
                )
                continue
            self._clients[client.server_label] = client

    async def shutdown(self) -> None:
        """Disconnect every client and clear the registry."""
        for client in self._clients.values():
            await client.disconnect()
        self._clients.clear()

    def get_client(self, server_label: str) -> BaseMCPClient:
        """Return the connected client registered under ``server_label``."""
        return self._clients[server_label]

    async def list_all_tools(self) -> list[dict[str, Any]]:
        """Return every server's tools, each tagged with its ``server_label``."""
        tools: list[dict[str, Any]] = []
        for client in self._clients.values():
            server_tools = await client.list_tools()
            for tool in server_tools:
                tools.append({**tool, "server_label": client.server_label})
        return tools

    async def call_tool(
        self, server_label: str, name: str, arguments: dict[str, Any]
    ) -> Any:
        """Invoke tool ``name`` on the server identified by ``server_label``."""
        return await self.get_client(server_label).call_tool(name, arguments)

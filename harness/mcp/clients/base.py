"""Base inteface every MCP server connection must implement."""

from abc import ABC, abstractmethod
from typing import Any


class BaseMCPClient(ABC):
    """Abstraction over single MCP server connection (stdio, HTTP/SSE)"""

    server_label: str

    @abstractmethod
    async def connect(self) -> None:
        """Establish the connection/session with MCP server."""

    @abstractmethod
    async def disconnect(self) -> None:
        """Tear down the connection."""

    @abstractmethod
    async def list_tools(self) -> list[dict[str, Any]]:
        """Return the tool definitions exposed by this MCP server."""

    @abstractmethod
    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        """Invoke a tool on the MCP server and return its result."""

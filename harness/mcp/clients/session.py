"""Shared base for MCP clients that own an ``mcp.ClientSession``.

The MCP handshake differs per transport (HTTP/SSE vs. stdio subprocess), but
once a :class:`ClientSession` exists every operation on it - listing tools,
calling a tool, tearing down - is transport-agnostic. Subclasses implement
:meth:`connect` and hand the established session to :meth:`_activate`.
"""

from contextlib import AsyncExitStack
from typing import Any

from mcp import ClientSession

from harness.core.exceptions import MCPConnectionError, ToolExecutionError
from harness.core.logging import get_logger
from harness.mcp.clients.base import BaseMCPClient

logger = get_logger(__name__)


class SessionMCPClient(BaseMCPClient):
    """A :class:`BaseMCPClient` backed by an ``mcp.ClientSession``."""

    def __init__(self, server_label: str, timeout_seconds: int = 30):
        """Initialise shared connection state; no session is opened yet."""
        self.server_label = server_label
        self.timeout_seconds = timeout_seconds
        self._session: ClientSession | None = None
        self._exit_stack: AsyncExitStack | None = None

    def _activate(self, exit_stack: AsyncExitStack, session: ClientSession) -> None:
        """Adopt an established exit stack and session (called from connect)."""
        self._exit_stack = exit_stack
        self._session = session
        logger.info("Connected to MCP server '%s'", self.server_label)

    async def disconnect(self) -> None:
        """Close the session and release all associated resources."""
        if self._exit_stack is not None:
            logger.info("Disconnecting from MCP server '%s'", self.server_label)
            await self._exit_stack.aclose()
        self._exit_stack = None
        self._session = None

    def _require_session(self) -> ClientSession:
        """Return the active session or raise if not connected.

        Raises:
            MCPConnectionError: If :meth:`connect` has not succeeded.
        """
        if self._session is None:
            raise MCPConnectionError(
                message=f"MCP server '{self.server_label}' is not connected"
            )
        return self._session

    async def list_tools(self) -> list[dict[str, Any]]:
        """Return the server's tools as name/description/input_schema dicts."""
        session = self._require_session()
        result = await session.list_tools()
        return [
            {
                "name": tool.name,
                "description": tool.description or "",
                "input_schema": tool.inputSchema or {},
            }
            for tool in result.tools
        ]

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        """Invoke tool ``name`` with ``arguments`` and return its result.

        Returns:
            The tool's structured content if present, otherwise its
            concatenated text blocks.

        Raises:
            ToolExecutionError: If the call fails or the tool reports an error.
        """
        session = self._require_session()
        try:
            result = await session.call_tool(name=name, arguments=arguments)
        except Exception as exc:
            raise ToolExecutionError(
                message=(
                    f"MCP tool '{name}' call failed on '{self.server_label}': {exc}"
                )
            ) from exc
        if result.isError:
            raise ToolExecutionError(
                message=f"MCP tool '{name}' returned an error: {result.content}"
            )
        if result.structuredContent is not None:
            return result.structuredContent
        return "".join(
            block.text  # type: ignore[attr-defined]
            for block in result.content
            if hasattr(block, "text")
        )

"""MCP client for servers exposed over streamable HTTP/SSE."""

from contextlib import AsyncExitStack
from urllib.parse import urlparse

import anyio
import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from harness.core.exceptions import MCPConnectionError
from harness.core.logging import get_logger
from harness.mcp.clients.session import SessionMCPClient

logger = get_logger(__name__)


class HTTPMCPClient(SessionMCPClient):
    """MCP client that talks to a server over streamable HTTP/SSE."""

    def __init__(self, server_label: str, server_url: str, timeout_seconds: int = 30):
        """Configure the client without opening a connection yet.

        Args:
            server_label: Human-readable id used in logs and errors.
            server_url: Base URL of the MCP server's HTTP endpoint.
            timeout_seconds: Connect/handshake timeout in seconds.
        """
        super().__init__(server_label=server_label, timeout_seconds=timeout_seconds)
        self.server_url = server_url

    async def _preflight_check(self) -> None:
        """Fast fail with a clean error if the MCP server is unreachable."""
        parsed = urlparse(url=self.server_url)
        host = parsed.hostname
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        if not host:
            raise MCPConnectionError(
                message=(
                    f"Invalid MCP server URL for '{self.server_label}': "
                    f"{self.server_url}"
                )
            )
        try:
            with anyio.fail_after(delay=self.timeout_seconds):
                stream = await anyio.connect_tcp(remote_host=host, remote_port=port)
            await stream.aclose()
        except (TimeoutError, OSError) as exc:
            raise MCPConnectionError(
                message=(
                    f"MCP server '{self.server_label}' at {host}:{port} "
                    f"is unreachable: {exc}"
                )
            ) from exc

    async def connect(self) -> None:
        """Open the HTTP session and complete the MCP handshake.

        Raises:
            MCPConnectionError: If the server is unreachable, the handshake
                times out, or setup otherwise fails.
        """
        logger.info(
            "Connecting to MCP server '%s' at '%s'", self.server_label, self.server_url
        )
        await self._preflight_check()

        exit_stack = AsyncExitStack()
        try:
            http_client = await exit_stack.enter_async_context(
                httpx.AsyncClient(
                    timeout=httpx.Timeout(timeout=self.timeout_seconds, read=300),
                    follow_redirects=True,
                )
            )
            read_stream, write_stream, _get_session_id = (
                await exit_stack.enter_async_context(
                    streamable_http_client(url=self.server_url, http_client=http_client)
                )
            )
            session = await exit_stack.enter_async_context(
                ClientSession(
                    read_stream=read_stream,
                    write_stream=write_stream,
                )
            )
            with anyio.fail_after(delay=self.timeout_seconds):
                await session.initialize()
        except TimeoutError as exc:
            await exit_stack.aclose()
            raise MCPConnectionError(
                message=(
                    f"Timed out waiting for MCP handshake with "
                    f"'{self.server_label}' after '{self.timeout_seconds}'"
                )
            ) from exc
        except Exception as exc:
            await exit_stack.aclose()
            raise MCPConnectionError(
                message=f"Failed to connect to MCP server '{self.server_label}': {exc}"
            ) from exc

        self._activate(exit_stack, session)

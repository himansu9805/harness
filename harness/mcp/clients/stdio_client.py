"""MCP client for servers launched locally and spoken to over stdio.

This is the transport used by the official ``command``/``args`` config style,
e.g. running ``npx @playwright/mcp@latest`` as a child process.
"""

from contextlib import AsyncExitStack

import anyio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from harness.core.exceptions import MCPConnectionError
from harness.core.logging import get_logger
from harness.mcp.clients.session import SessionMCPClient

logger = get_logger(__name__)


class StdioMCPClient(SessionMCPClient):
    """MCP client that spawns a local process and speaks MCP over stdio."""

    def __init__(
        self,
        server_label: str,
        command: str,
        *,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
        timeout_seconds: int = 30,
    ):
        """Configure the subprocess to launch without starting it yet.

        Args:
            server_label: Human-readable id used in logs and errors.
            command: Executable to run (e.g. ``npx``).
            args: Arguments passed to ``command``.
            env: Extra environment variables for the child process. When
                ``None`` the SDK provides a minimal default environment that
                includes ``PATH`` (so ``npx`` and friends resolve).
            cwd: Working directory for the child process.
            timeout_seconds: Handshake timeout in seconds.
        """
        super().__init__(server_label=server_label, timeout_seconds=timeout_seconds)
        self.command = command
        self.args = args or []
        self.env = env
        self.cwd = cwd

    async def connect(self) -> None:
        """Spawn the server process and complete the MCP handshake.

        Raises:
            MCPConnectionError: If the process cannot be started or the
                handshake times out or otherwise fails.
        """
        logger.info(
            "Starting MCP server '%s': %s %s",
            self.server_label,
            self.command,
            " ".join(self.args),
        )
        server_params = StdioServerParameters(
            command=self.command,
            args=self.args,
            env=self.env,
            cwd=self.cwd,
        )

        exit_stack = AsyncExitStack()
        try:
            read_stream, write_stream = await exit_stack.enter_async_context(
                stdio_client(server_params)
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
                    f"'{self.server_label}' after '{self.timeout_seconds}s' "
                    f"(first run may need to download the server package)"
                )
            ) from exc
        except Exception as exc:
            await exit_stack.aclose()
            raise MCPConnectionError(
                message=(
                    f"Failed to start MCP server '{self.server_label}' "
                    f"('{self.command}'): {exc}"
                )
            ) from exc

        self._activate(exit_stack, session)

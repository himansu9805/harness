"""Built-in tools that ship with the harness."""

import asyncio
from typing import Any

from harness.core.exceptions import ToolExecutionError
from harness.core.logging import get_logger
from harness.tools.base import BaseTool

logger = get_logger(__name__)

DEFAULT_TIMEOUT = 120
MAX_TIMEOUT = 600
MAX_OUTPUT_CHARS = 30_000


class ShellTool(BaseTool):
    """Execute a shell command and return its stdout, stderr and exit code."""

    name = "shell"
    description = (
        "Execute a shell command in the local environment and return its "
        "stdout, stderr and exit code. Commands run through the system shell, "
        "so pipes, redirection and shell built-ins are supported. Use for "
        "running scripts, inspecting the filesystem and interacting with CLI "
        "tools."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The shell command to execute.",
            },
            "workdir": {
                "type": "string",
                "description": (
                    "Optional working directory to run the command in. "
                    "Defaults to the current process directory."
                ),
            },
            "timeout": {
                "type": "integer",
                "description": (
                    "Optional timeout in seconds before the command is killed. "
                    f"Defaults to {DEFAULT_TIMEOUT}, capped at {MAX_TIMEOUT}."
                ),
            },
        },
        "required": ["command"],
    }

    async def execute(self, arguments: dict[str, Any]) -> dict[str, Any]:
        command = arguments.get("command")
        if not command or not isinstance(command, str):
            raise ToolExecutionError(
                message="'command' is required and must be a non-empty string"
            )

        workdir = arguments.get("workdir")
        raw_timeout = arguments.get("timeout")
        timeout = DEFAULT_TIMEOUT if raw_timeout is None else int(raw_timeout)
        # Keep the timeout within a sane, positive range.
        timeout = max(1, min(timeout, MAX_TIMEOUT))

        logger.info("Executing shell command: %s", command)

        try:
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=workdir,
            )
        except OSError as exc:
            raise ToolExecutionError(
                message=f"Failed to start command: {exc}", cause=exc
            ) from exc

        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                process.communicate(), timeout=timeout
            )
        except asyncio.TimeoutError as exc:
            process.kill()
            await process.wait()
            raise ToolExecutionError(
                message=f"Command timed out after {timeout} seconds",
                cause=exc,
            ) from exc

        stdout = _decode_and_truncate(stdout_bytes)
        stderr = _decode_and_truncate(stderr_bytes)

        return {
            "exit_code": process.returncode,
            "stdout": stdout,
            "stderr": stderr,
        }


def _decode_and_truncate(data: bytes) -> str:
    text = data.decode("utf-8", errors="replace")
    if len(text) > MAX_OUTPUT_CHARS:
        omitted = len(text) - MAX_OUTPUT_CHARS
        text = text[:MAX_OUTPUT_CHARS] + f"\n... [truncated {omitted} characters]"
    return text

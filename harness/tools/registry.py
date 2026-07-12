"""Unified registry over built-in, MCP and skill tools."""

import time
from typing import Any

from harness.core.exceptions import ToolExecutionError
from harness.core.logging import get_logger
from harness.mcp.registry.registry import MCPRegistry
from harness.schemas.tool import ToolDefinition, ToolInvocationResult
from harness.skills.registry.registry import SkillRegistry
from harness.tools.base import BaseTool

logger = get_logger(__name__)


class ToolRegistry:
    """A registry for tools"""

    def __init__(self, mcp_registry: MCPRegistry, skill_registry: SkillRegistry):
        """Wire the registry to its MCP and skill sources.

        Args:
            mcp_registry: Source of tools exposed by MCP servers.
            skill_registry: Source of skill-backed tools.
        """
        self._mcp_registry = mcp_registry
        self._skill_registry = skill_registry
        self._builtin_tools: dict[str, BaseTool] = {}

    def register_builtin_tool(self, tool: BaseTool) -> None:
        """Register a built-in ``tool`` under its name."""
        self._builtin_tools[tool.name] = tool

    async def list_tools(self) -> list[ToolDefinition]:
        """Return definitions for all built-in, MCP and skill tools."""
        definitions = [
            ToolDefinition(
                name=tool.name,
                description=tool.description,
                parameters=tool.parameters,
                source="builtin",
            )
            for tool in self._builtin_tools.values()
        ]
        for tool in await self._mcp_registry.list_all_tools():
            definitions.append(
                ToolDefinition(
                    name=tool["name"],
                    description=tool.get("description", ""),
                    parameters=tool.get("input_schema", {}),
                    source="mcp",
                    server_label=tool.get("server_label"),
                )
            )
        for manifest in self._skill_registry.list_maifests():
            definitions.append(
                ToolDefinition(
                    name=manifest.name,
                    description=manifest.description,
                    source="skill",
                )
            )
        return definitions

    async def call(
        self, name: str, arguments: dict[str, Any], server_label: str | None = None
    ) -> ToolInvocationResult:
        """Invoke a tool by name and wrap its output or error.

        Built-in tools take precedence; otherwise ``server_label`` routes the
        call to an MCP server. Any exception is captured and returned as a
        result with ``is_error=True`` rather than propagated.

        Args:
            name: The tool's name.
            arguments: Arguments passed to the tool.
            server_label: MCP server to route to when the tool is not built-in.

        Returns:
            The invocation result, flagged as an error if the call failed.
        """
        started = time.perf_counter()
        try:
            if name in self._builtin_tools:
                output = await self._builtin_tools[name].execute(arguments=arguments)
            elif server_label:
                output = await self._mcp_registry.call_tool(
                    server_label=server_label, name=name, arguments=arguments
                )
            else:
                raise ToolExecutionError(
                    message=f"Unable to resolve tool '{name}' to a handler"
                )
            duration_ms = (time.perf_counter() - started) * 1000
            return ToolInvocationResult(
                name=name, output=output, duration_ms=duration_ms
            )
        except Exception as exc:
            duration_ms = (time.perf_counter() - started) * 1000
            logger.exception("Tool '%s' execution failed", name)
            return ToolInvocationResult(
                name=name, output=str(exc), is_error=True, duration_ms=duration_ms
            )

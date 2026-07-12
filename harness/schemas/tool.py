"""Pydantic models describing tools and their invocation requests/results."""

from typing import Any, Literal

from pydantic import BaseModel, Field


class ToolDefinition(BaseModel):
    """A tool exposed to the model, from a builtin, skill or MCP server."""

    name: str
    description: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    source: Literal["mcp", "skill", "builtin"]
    server_label: str | None = None


class ToolInvocationRequest(BaseModel):
    """A request to invoke a named tool with the given arguments."""

    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolInvocationResult(BaseModel):
    """The outcome of a tool invocation, including any error flag."""

    name: str
    output: Any
    is_error: bool = False
    duration_ms: float | None = None

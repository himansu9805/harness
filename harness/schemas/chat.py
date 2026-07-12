"""Pydantic models for chat requests, responses and messages."""

from typing import Any, Literal

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    """A single message in a conversation (system, user, assistant or tool)."""

    role: Literal["system", "user", "assistant", "tool"]
    content: str | None = None
    name: str | None = None
    tool_call_id: str | None = Field(
        default=None,
        description=(
            "Required on 'tool' messages: id of the assistant tool_call "
            "being answered"
        ),
    )
    tool_calls: list[dict[str, Any]] | None = Field(
        default=None,
        description="Present on 'assistant' messages that require tool invocations",
    )


class ChatRequest(BaseModel):
    """An incoming request to run a conversation turn."""

    session_id: str | None = Field(
        default=None, description="Existing conversation/session id"
    )
    messages: list[ChatMessage]
    model: str | None = None
    stream: bool = False
    tool_choice: Literal["auto", "none", "required"] = "auto"
    metadata: dict[str, Any] = Field(default_factory=dict)


class ToolCall(BaseModel):
    """A record of a single tool invocation made during a turn."""

    id: str
    name: str
    arguments: dict[str, Any]
    source: Literal["mcp", "skill", "builtin"] = "builtin"


class ChatResponse(BaseModel):
    """The result of a completed conversation turn."""

    session_id: str
    message: ChatMessage
    tool_calls: list[ToolCall] = Field(default_factory=list)
    usage: dict[str, int] = Field(default_factory=dict)

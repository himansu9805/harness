"""Orchestrator containing the core loop that ties the harness together.

Wires the planner, tool registry and memory into a single agent turn: it
repeatedly asks the planner for the next step, executes any requested tool
calls, and returns once the planner produces a final response.
"""

import json
import uuid
from typing import Any

from harness.agent.memory.base import BaseMemory
from harness.agent.memory.in_memory import InMemoryMemory
from harness.agent.planners.base import BasePlanner
from harness.agent.planners.function_calling import FunctionCallingPlanner
from harness.core.config import Settings
from harness.core.logging import get_logger
from harness.schemas.chat import ChatMessage, ChatResponse, ToolCall
from harness.schemas.tool import ToolDefinition
from harness.tools.registry import ToolRegistry

logger = get_logger(__name__)

MAX_TOOL_ITERATIONS = 8
_PREVIEW_LIMIT = 200


def _preview(value: Any, limit: int = _PREVIEW_LIMIT) -> str:
    """Render ``value`` as a single-line, length-capped string for logs."""
    text = " ".join(str(value).split())
    if len(text) > limit:
        return f"{text[:limit]}… (+{len(text) - limit} chars)"
    return text


class AgentOrchestrator:
    """Runs the plan-act loop for a conversation turn."""

    def __init__(
        self,
        settings: Settings,
        tool_registry: ToolRegistry,
        planner: BasePlanner | None = None,
        memory: BaseMemory | None = None,
    ):
        """Wire up the orchestrator's collaborators.

        Args:
            settings: Runtime configuration.
            tool_registry: Registry used to list and invoke tools.
            planner: Strategy for choosing the next step; defaults to
                :class:`FunctionCallingPlanner`.
            memory: History store; defaults to :class:`InMemoryMemory`.
        """
        self._settings = settings
        self._tool_registry = tool_registry
        self._planner = planner or FunctionCallingPlanner(settings=settings)
        self._memory = memory or InMemoryMemory()

    async def run_turn(
        self, session_id: str | None, messages: list[ChatMessage]
    ) -> ChatResponse:
        """Run one conversation turn to completion.

        Appends ``messages`` to history, then loops (up to
        ``MAX_TOOL_ITERATIONS`` times) planning and executing tool calls until
        the planner responds or the budget is exhausted.

        Args:
            session_id: Existing session id, or ``None`` to start a new one.
            messages: New user/system messages for this turn.

        Returns:
            The assistant's final response along with any tool calls made.
        """
        session_id = session_id or str(uuid.uuid4())
        sid = session_id[:8]
        history = await self._memory.get_history(session_id=session_id)
        conversation = history + messages
        for message in messages:
            await self._memory.append(session_id=session_id, message=message)

        tool_defs = await self._tool_registry.list_tools()
        openai_tools = self._to_openai_tool_schema(tool_defs=tool_defs)
        logger.info(
            "[%s] turn start: %d new message(s), %d prior, %d tool(s) available",
            sid,
            len(messages),
            len(history),
            len(tool_defs),
        )

        tool_calls_made: list[ToolCall] = []
        for iteration in range(1, MAX_TOOL_ITERATIONS + 1):
            logger.info(
                "[%s] planning (step %d/%d)", sid, iteration, MAX_TOOL_ITERATIONS
            )
            step = await self._planner.next_step(
                messages=conversation, available_tools=openai_tools
            )

            if step["action"] == "respond":
                logger.info(
                    "[%s] model responded after %d tool call(s): %s",
                    sid,
                    len(tool_calls_made),
                    _preview(step["content"] or ""),
                )
                final_message = ChatMessage(
                    role="assistant", content=step["content"] or ""
                )
                await self._memory.append(session_id=session_id, message=final_message)
                return ChatResponse(
                    session_id=session_id,
                    message=final_message,
                    tool_calls=tool_calls_made,
                )

            if step["action"] == "call_tools":
                raw_tool_calls = step["tool_calls"]
                logger.info(
                    "[%s] model requested %d tool call(s): %s",
                    sid,
                    len(raw_tool_calls),
                    ", ".join(call["name"] for call in raw_tool_calls),
                )
                assistant_message = ChatMessage(
                    role="assistant",
                    content=step.get("assistant_content"),
                    tool_calls=[
                        {
                            "id": call["id"],
                            "type": "function",
                            "function": {
                                "name": call["name"],
                                "arguments": call["arguments"],
                            },
                        }
                        for call in raw_tool_calls
                    ],
                )
                conversation.append(assistant_message)
                await self._memory.append(
                    session_id=session_id, message=assistant_message
                )

                for call in raw_tool_calls:
                    arguments = call["arguments"]
                    if isinstance(arguments, str):
                        arguments = json.loads(arguments or "{}")

                    tool_def = next(
                        (tool for tool in tool_defs if tool.name == call["name"]), None
                    )
                    source = tool_def.source if tool_def else "builtin"
                    logger.info(
                        "[%s] → tool '%s' (%s) args=%s",
                        sid,
                        call["name"],
                        source,
                        _preview(arguments),
                    )
                    result = await self._tool_registry.call(
                        name=call["name"],
                        arguments=arguments,
                        server_label=tool_def.server_label if tool_def else None,
                    )
                    logger.info(
                        "[%s] ← tool '%s' %s in %.0fms: %s",
                        sid,
                        call["name"],
                        "ERROR" if result.is_error else "ok",
                        result.duration_ms or 0.0,
                        _preview(result.output),
                    )
                    tool_calls_made.append(
                        ToolCall(
                            id=call["id"],
                            name=call["name"],
                            arguments=arguments,
                            source=source,
                        )
                    )

                    tool_message = ChatMessage(
                        role="tool",
                        name=call["name"],
                        tool_call_id=call["id"],
                        content=str(result.output),
                    )
                    conversation.append(tool_message)
                    await self._memory.append(
                        session_id=session_id, message=tool_message
                    )
                continue
            break

        logger.warning(
            "[%s] gave up after %d iteration(s) without a final response",
            sid,
            MAX_TOOL_ITERATIONS,
        )
        fallback = ChatMessage(
            role="assistant", content="I wasn't able to complete this request in time."
        )
        return ChatResponse(
            session_id=session_id, message=fallback, tool_calls=tool_calls_made
        )

    @staticmethod
    def _to_openai_tool_schema(tool_defs: list[ToolDefinition]) -> list[dict[str, Any]]:
        """Convert internal tool definitions to OpenAI function-call schemas."""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters
                    or {"type": "object", "properties": {}},
                },
            }
            for tool in tool_defs
        ]

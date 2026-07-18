"""Orchestrator containing the core loop that ties the harness together.

Wires the planner, tool registry and memory into a single agent turn: it
repeatedly asks the planner for the next step, executes any requested tool
calls, and returns once the planner produces a final response.

The loop is bounded by two independent guards rather than a single step
counter:

- **Budget** (:attr:`_max_iterations`) — the most plan-act iterations a turn is
  allowed. Sized generously for legitimately long tasks (e.g. driving a browser
  via MCP) and configurable per deployment or per call.
- **Progress** (:attr:`_max_repeated_tool_calls`) — a stall guard that ends the
  turn early when the model requests the *identical* set of tool calls several
  times in a row, which signals it is stuck rather than making headway.

When either guard trips, the turn does not simply bail with a canned message;
it asks the model once more, with tools disabled, to summarise what it managed
to accomplish.
"""

import json
import uuid
from typing import Any

import aiofiles

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

# Fallbacks used only when a caller supplies neither an explicit value nor a
# Settings object carrying one.
DEFAULT_MAX_TOOL_ITERATIONS = 50
DEFAULT_MAX_REPEATED_TOOL_CALLS = 3
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
        *,
        max_iterations: int | None = None,
        max_repeated_tool_calls: int | None = None,
    ):
        """Wire up the orchestrator's collaborators.

        Args:
            settings: Runtime configuration.
            tool_registry: Registry used to list and invoke tools.
            planner: Strategy for choosing the next step; defaults to
                :class:`FunctionCallingPlanner`.
            memory: History store; defaults to :class:`InMemoryMemory`.
            max_iterations: Override for the per-turn iteration budget. Falls
                back to ``settings.max_tool_iterations``.
            max_repeated_tool_calls: Override for the stall threshold. Falls
                back to ``settings.max_repeated_tool_calls``.
        """
        self._settings = settings
        self._tool_registry = tool_registry
        self._planner = planner or FunctionCallingPlanner(settings=settings)
        self._memory = memory or InMemoryMemory()
        self._max_iterations = max_iterations or getattr(
            settings, "max_tool_iterations", DEFAULT_MAX_TOOL_ITERATIONS
        )
        self._max_repeated_tool_calls = max_repeated_tool_calls or getattr(
            settings, "max_repeated_tool_calls", DEFAULT_MAX_REPEATED_TOOL_CALLS
        )

    async def _read_system_prompt(self) -> str:
        """Read system prompt."""
        async with aiofiles.open(
            self._settings.system_prompt_path, mode="r", encoding="utf-8"
        ) as file:
            return await file.read()

    async def run_turn(
        self,
        session_id: str | None,
        messages: list[ChatMessage],
        *,
        max_iterations: int | None = None,
    ) -> ChatResponse:
        """Run one conversation turn to completion.

        Appends ``messages`` to history, then loops (up to the iteration
        budget) planning and executing tool calls until the planner responds,
        the budget is exhausted, or the model stalls.

        Args:
            session_id: Existing session id, or ``None`` to start a new one.
            messages: New user/system messages for this turn.
            max_iterations: Optional per-turn override of the iteration budget,
                useful for a single tool-heavy request.

        Returns:
            The assistant's final response along with any tool calls made.
        """
        budget = max_iterations or self._max_iterations
        session_id = session_id or str(uuid.uuid4())
        sid = session_id[:8]
        history = await self._memory.get_history(session_id=session_id)
        if not history:
            history = [
                ChatMessage(
                    role="system",
                    content=await self._read_system_prompt(),
                )
            ]
        conversation = history + messages
        for message in messages:
            await self._memory.append(session_id=session_id, message=message)

        tool_defs = await self._tool_registry.list_tools()
        openai_tools = self._to_openai_tool_schema(tool_defs=tool_defs)
        logger.info(
            "[%s] turn start: %d new message(s), %d prior, %d tool(s), budget=%d",
            sid,
            len(messages),
            len(history),
            len(tool_defs),
            budget,
        )

        tool_calls_made: list[ToolCall] = []
        last_signature: tuple[tuple[str, str], ...] | None = None
        repeat_count = 0
        for iteration in range(1, budget + 1):
            logger.info("[%s] planning (step %d/%d)", sid, iteration, budget)
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

                signature = self._tool_call_signature(raw_tool_calls)
                if signature == last_signature:
                    repeat_count += 1
                else:
                    last_signature = signature
                    repeat_count = 1
                if repeat_count >= self._max_repeated_tool_calls:
                    logger.warning(
                        "[%s] stalled: identical tool call(s) requested %d times "
                        "in a row",
                        sid,
                        repeat_count,
                    )
                    return await self._forced_final_response(
                        session_id=session_id,
                        conversation=conversation,
                        tool_calls_made=tool_calls_made,
                        sid=sid,
                        reason="kept repeating the same step without progress",
                    )

                await self._execute_tool_batch(
                    session_id=session_id,
                    sid=sid,
                    step=step,
                    tool_defs=tool_defs,
                    conversation=conversation,
                    tool_calls_made=tool_calls_made,
                )
                continue
            break

        return await self._forced_final_response(
            session_id=session_id,
            conversation=conversation,
            tool_calls_made=tool_calls_made,
            sid=sid,
            reason=f"reached the {budget}-step limit for this turn",
        )

    async def _execute_tool_batch(
        self,
        *,
        session_id: str,
        sid: str,
        step: dict[str, Any],
        tool_defs: list[ToolDefinition],
        conversation: list[ChatMessage],
        tool_calls_made: list[ToolCall],
    ) -> None:
        """Record the assistant's tool request, run each call, and log results.

        Appends the assistant message and one ``tool`` message per call to both
        the working ``conversation`` and memory, and extends ``tool_calls_made``
        with a record of each invocation.
        """
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
        await self._memory.append(session_id=session_id, message=assistant_message)

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
            await self._memory.append(session_id=session_id, message=tool_message)

    async def _forced_final_response(
        self,
        *,
        session_id: str,
        conversation: list[ChatMessage],
        tool_calls_made: list[ToolCall],
        sid: str,
        reason: str,
    ) -> ChatResponse:
        """End the turn by asking the model to answer with tools disabled.

        Passing no tools forces the planner to produce a textual reply, so the
        user gets the model's best summary of the work so far instead of a
        canned give-up message. If that final call fails or comes back empty, a
        static fallback is used.
        """
        logger.warning(
            "[%s] ending turn (%s) after %d tool call(s); asking model to summarise",
            sid,
            reason,
            len(tool_calls_made),
        )
        content: str | None = None
        try:
            step = await self._planner.next_step(
                messages=conversation, available_tools=[]
            )
            if step.get("action") == "respond":
                content = step.get("content")
        except Exception:
            logger.exception("[%s] forced final response failed", sid)

        if not content:
            content = (
                "I wasn't able to fully complete this request because I "
                f"{reason}. Let me know if you'd like me to keep going."
            )

        final_message = ChatMessage(role="assistant", content=content)
        await self._memory.append(session_id=session_id, message=final_message)
        return ChatResponse(
            session_id=session_id, message=final_message, tool_calls=tool_calls_made
        )

    @staticmethod
    def _tool_call_signature(
        raw_tool_calls: list[dict[str, Any]],
    ) -> tuple[tuple[str, str], ...]:
        """Return an order-independent fingerprint of a batch of tool calls.

        Two batches with the same tool names and arguments share a signature,
        letting the loop notice when the model keeps asking for the exact same
        thing. Arguments may arrive as a raw JSON string or a parsed dict, and
        key ordering may differ between calls; both are normalised to a
        canonical form so equivalent calls compare equal.
        """
        parts: list[tuple[str, str]] = []
        for call in raw_tool_calls:
            arguments = call.get("arguments")
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments or "{}")
                except json.JSONDecodeError:
                    parts.append((call.get("name", ""), arguments))
                    continue
            try:
                canonical = json.dumps(arguments, sort_keys=True, default=str)
            except TypeError:
                canonical = str(arguments)
            parts.append((call.get("name", ""), canonical))
        return tuple(sorted(parts))

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

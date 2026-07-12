"""Planner backed by an LLM's native function-calling API (via litellm)."""

import time
from typing import Any

import litellm

from harness.agent.planners.base import BasePlanner
from harness.core.config import Settings
from harness.core.exceptions import LLMProviderError
from harness.core.logging import get_logger
from harness.schemas.chat import ChatMessage

logger = get_logger(__name__)


class FunctionCallingPlanner(BasePlanner):
    """Ask the configured LLM to either answer or request tool calls."""

    def __init__(self, settings: Settings):
        """Store ``settings`` used to select the model and API endpoint."""
        self._settings = settings

    async def next_step(
        self, messages: list[ChatMessage], available_tools: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Query the LLM and translate its reply into a plan step.

        Args:
            messages: The conversation so far, oldest first.
            available_tools: Tool schemas in OpenAI function-calling format.

        Returns:
            ``{"action": "call_tools", ...}`` if the model requested tools,
            otherwise ``{"action": "respond", "content": ...}``.

        Raises:
            LLMProviderError: If the underlying completion call fails.
        """
        model = self._settings.ollama_model
        logger.debug(
            "requesting completion from '%s' (%d messages, %d tools)",
            model,
            len(messages),
            len(available_tools),
        )
        started = time.perf_counter()
        try:
            response = await litellm.acompletion(
                model=model,
                messages=[
                    message.model_dump(exclude_none=True) for message in messages
                ],
                tools=available_tools or None,
                tool_choice="auto" if available_tools else None,
                api_base=self._settings.ollama_base_url,
            )
        except Exception as exc:
            raise LLMProviderError(message=f"LLM call failed: {exc}") from exc

        elapsed_ms = (time.perf_counter() - started) * 1000
        usage = getattr(response, "usage", None)
        logger.info(
            "LLM '%s' replied in %.0fms (prompt=%s, completion=%s tokens)",
            model,
            elapsed_ms,
            getattr(usage, "prompt_tokens", "?"),
            getattr(usage, "completion_tokens", "?"),
        )

        choice = response.choices[0]  # type: ignore
        if choice.message.tool_calls:
            return {
                "action": "call_tools",
                "assistant_content": choice.message.content,
                "tool_calls": [
                    {
                        "id": call.id,
                        "name": call.function.name,
                        "arguments": call.function.arguments,
                    }
                    for call in choice.message.tool_calls
                ],
            }
        return {"action": "respond", "content": choice.message.content}

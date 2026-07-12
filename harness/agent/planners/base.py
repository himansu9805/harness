"""Planner interface."""

from abc import ABC, abstractmethod
from typing import Any

from harness.schemas.chat import ChatMessage


class BasePlanner(ABC):
    """Decides the agent's next action given the conversation and tools."""

    @abstractmethod
    async def next_step(
        self,
        messages: list[ChatMessage],
        available_tools: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Decide the next step for the agent loop.

        Args:
            messages: The conversation so far, oldest first.
            available_tools: Tool schemas in OpenAI function-calling format.

        Returns:
            A step dict, either ``{"action": "respond", "content": ...}`` or
            ``{"action": "call_tools", "tool_calls": [...]}``.
        """

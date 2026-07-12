"""Base interface every built-in tool must implement."""

from abc import ABC, abstractmethod
from typing import Any


class BaseTool(ABC):
    """A callable tool with a name, description and JSON-schema parameters."""

    name: str
    description: str
    parameters: dict[str, Any] = {}

    @abstractmethod
    async def execute(self, arguments: dict[str, Any]) -> Any:
        """Run the tool with the given arguments and return the result."""

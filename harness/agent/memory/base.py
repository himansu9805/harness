"""Conversation memory abstraction for storing per-session chat history."""

from abc import ABC, abstractmethod

from harness.schemas.chat import ChatMessage


class BaseMemory(ABC):
    """Interface for a store that persists chat history keyed by session id."""

    @abstractmethod
    async def get_history(self, session_id: str) -> list[ChatMessage]:
        """Return the ordered message history for ``session_id`` (empty if none)."""

    @abstractmethod
    async def append(self, session_id: str, message: ChatMessage) -> None:
        """Append ``message`` to the history of ``session_id``."""

    @abstractmethod
    async def clear(self, session_id: str) -> None:
        """Remove all stored history for ``session_id``."""

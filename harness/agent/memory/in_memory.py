"""In-process, non-persistent implementation of :class:`BaseMemory`."""

from collections import defaultdict

from harness.agent.memory.base import BaseMemory
from harness.schemas.chat import ChatMessage


class InMemoryMemory(BaseMemory):
    """Store chat history in a dict; state is lost when the process exits."""

    def __init__(self):
        """Initialize an empty per-session message store."""
        self._store: dict[str, list[ChatMessage]] = defaultdict(list)

    async def get_history(self, session_id: str) -> list[ChatMessage]:
        """Return a shallow copy of the messages stored for ``session_id``."""
        return list(self._store[session_id])

    async def append(self, session_id: str, message: ChatMessage) -> None:
        """Append ``message`` to the history of ``session_id``."""
        self._store[session_id].append(message)

    async def clear(self, session_id: str) -> None:
        """Discard any stored history for ``session_id``."""
        self._store.pop(session_id, None)

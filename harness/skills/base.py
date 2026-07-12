"""Base interface every skill must implement."""

from abc import ABC, abstractmethod
from typing import Any

from harness.schemas.skill import SkillManifest


class BaseSkill(ABC):
    """A self-contained capability activated by matching user input."""

    manifest: SkillManifest

    @abstractmethod
    async def can_handle(self, user_input) -> bool:
        """Return true if this skill should be activated for the given input."""

    @abstractmethod
    async def run(self, user_input: str, context: dict[str, Any]) -> str:
        """Execute the skill and return its textual result."""

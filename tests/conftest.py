"""Shared fixtures and lightweight fakes for the harness test suite."""

from typing import Any

import pytest

from harness.agent.planners.base import BasePlanner
from harness.core.config import Settings
from harness.mcp.registry.registry import MCPRegistry
from harness.schemas.chat import ChatMessage
from harness.schemas.skill import SkillManifest
from harness.skills.base import BaseSkill
from harness.skills.registry.registry import SkillRegistry
from harness.tools.base import BaseTool


@pytest.fixture
def settings() -> Settings:
    """A Settings instance built without touching the real environment/.env."""
    return Settings(_env_file=None)


class FakeMCPRegistry(MCPRegistry):
    """MCPRegistry with the transport layer stubbed out for tests."""

    def __init__(self, tools: list[dict[str, Any]] | None = None):
        self._tools = tools or []
        self.calls: list[tuple[str, str, dict[str, Any]]] = []

    async def list_all_tools(self) -> list[dict[str, Any]]:
        return list(self._tools)

    async def call_tool(
        self, server_label: str, name: str, arguments: dict[str, Any]
    ) -> Any:
        self.calls.append((server_label, name, arguments))
        return {"echo": arguments}


class EchoTool(BaseTool):
    """A trivial built-in tool that echoes its arguments back."""

    name = "echo"
    description = "Echo the given arguments."
    parameters = {
        "type": "object",
        "properties": {"value": {"type": "string"}},
    }

    async def execute(self, arguments: dict[str, Any]) -> dict[str, Any]:
        return {"echoed": arguments}


class BoomTool(BaseTool):
    """A built-in tool that always raises, for error-path testing."""

    name = "boom"
    description = "Always fails."

    async def execute(self, arguments: dict[str, Any]) -> Any:
        raise RuntimeError("kaboom")


class KeywordSkill(BaseSkill):
    """A skill that activates when a keyword appears in the input."""

    def __init__(self, name: str, keyword: str, enabled: bool = True):
        self.manifest = SkillManifest(
            name=name, description=f"Handles {keyword}", enabled=enabled
        )
        self._keyword = keyword

    async def can_handle(self, user_input: str) -> bool:
        return self._keyword in user_input

    async def run(self, user_input: str, context: dict[str, Any]) -> str:
        return f"{self.manifest.name} handled: {user_input}"


class ScriptedPlanner(BasePlanner):
    """A planner that returns a predefined sequence of steps."""

    def __init__(self, steps: list[dict[str, Any]]):
        self._steps = list(steps)
        self.calls: list[list[ChatMessage]] = []

    async def next_step(
        self, messages: list[ChatMessage], available_tools: list[dict[str, Any]]
    ) -> dict[str, Any]:
        self.calls.append(list(messages))
        return self._steps.pop(0)


@pytest.fixture
def echo_tool() -> EchoTool:
    return EchoTool()


@pytest.fixture
def boom_tool() -> BoomTool:
    return BoomTool()


@pytest.fixture
def skill_registry() -> SkillRegistry:
    return SkillRegistry()

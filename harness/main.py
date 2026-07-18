"""Executable entry point that wires the harness together and runs a turn."""

import asyncio

from harness.agent.orchestrator.orchestrator import AgentOrchestrator
from harness.core.config import get_settings
from harness.core.logging import configure_logging, get_logger
from harness.mcp.registry.registry import MCPRegistry
from harness.schemas.chat import ChatMessage
from harness.skills.registry.registry import SkillRegistry
from harness.tools.builtin import ShellTool
from harness.tools.registry import ToolRegistry

logger = get_logger(__name__)


async def main():
    """Entry point for harness."""

    configure_logging()
    settings = get_settings()

    mcp_registry = MCPRegistry(settings=settings)
    await mcp_registry.initialize()
    skill_registry = SkillRegistry()
    tool_registry = ToolRegistry(
        mcp_registry=mcp_registry, skill_registry=skill_registry
    )
    tool_registry.register_builtin_tool(ShellTool())
    orchestrator = AgentOrchestrator(settings=settings, tool_registry=tool_registry)

    try:
        result = await orchestrator.run_turn(
            session_id=None,
            messages=[
                ChatMessage(
                    role="user",
                    content=("latest news related to manchester city"),
                )
            ],
        )
        print(result)
    finally:
        await mcp_registry.shutdown()


if __name__ == "__main__":
    asyncio.run(main())

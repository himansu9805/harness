"""Pydantic models describing skills and their invocation results."""

from pydantic import BaseModel, Field


class SkillManifest(BaseModel):
    """Metadata declaring a skill and what activates it."""

    name: str
    version: str = "0.1.0"
    description: str
    triggers: list[str] = Field(
        default_factory=list, description="Keywords/intents that activate this skill"
    )
    required_tools: list[str] = Field(default_factory=list)
    enabled: bool = True


class SkillInvocationResult(BaseModel):
    """The output produced by running a skill."""

    skill_name: str
    output: str
    tool_calls_made: list[str] = Field(default_factory=list)

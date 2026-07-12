"""Registry that holds available skills and matches them to user input."""

from harness.core.exceptions import SkillNotFoundError
from harness.schemas.skill import SkillManifest
from harness.skills.base import BaseSkill


class SkillRegistry:
    """In-memory catalog of skills keyed by name."""

    def __init__(self):
        """Create an empty registry and load the built-in skills."""
        self._skills: dict[str, BaseSkill] = {}
        self._register_builtin_skills()

    def _register_builtin_skills(self) -> None:
        """Register skills that ship with the harness (currently none)."""
        # Import the skill and pass it to the register method here:
        # self.register(SomeSkill())

    def register(self, skill: BaseSkill) -> None:
        """Add ``skill`` to the registry under its manifest name."""
        self._skills[skill.manifest.name] = skill

    def get(self, name: str) -> BaseSkill:
        """Return the skill registered as ``name``.

        Raises:
            SkillNotFoundError: If no skill is registered under ``name``.
        """
        try:
            return self._skills[name]
        except KeyError as exc:
            raise SkillNotFoundError(f"Skill '{name}' is not registered") from exc

    def list_maifests(self) -> list[SkillManifest]:
        """Return the manifests of all enabled skills."""
        return [
            skill.manifest for skill in self._skills.values() if skill.manifest.enabled
        ]

    async def find_matching(self, user_input: str) -> list[BaseSkill]:
        """Return all enabled skills that can handle ``user_input``."""
        matches = []
        for skill in self._skills.values():
            if skill.manifest.enabled and await skill.can_handle(user_input=user_input):
                matches.append(skill)
        return matches

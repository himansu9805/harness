"""Tests for the SkillRegistry catalog and matching logic."""

import pytest

from harness.core.exceptions import SkillNotFoundError
from harness.skills.registry.registry import SkillRegistry
from tests.conftest import KeywordSkill


def test_new_registry_has_no_manifests():
    assert SkillRegistry().list_maifests() == []


def test_register_and_get(skill_registry):
    skill = KeywordSkill(name="alpha", keyword="a")
    skill_registry.register(skill)
    assert skill_registry.get("alpha") is skill


def test_get_unknown_raises(skill_registry):
    with pytest.raises(SkillNotFoundError):
        skill_registry.get("missing")


def test_list_manifests_excludes_disabled(skill_registry):
    skill_registry.register(KeywordSkill(name="on", keyword="x"))
    skill_registry.register(KeywordSkill(name="off", keyword="y", enabled=False))
    names = {m.name for m in skill_registry.list_maifests()}
    assert names == {"on"}


async def test_find_matching_returns_activating_skills(skill_registry):
    skill_registry.register(KeywordSkill(name="weather", keyword="weather"))
    skill_registry.register(KeywordSkill(name="news", keyword="news"))
    matches = await skill_registry.find_matching("what is the weather today")
    assert [s.manifest.name for s in matches] == ["weather"]


async def test_find_matching_skips_disabled(skill_registry):
    skill_registry.register(
        KeywordSkill(name="weather", keyword="weather", enabled=False)
    )
    assert await skill_registry.find_matching("weather please") == []


async def test_find_matching_empty_when_no_match(skill_registry):
    skill_registry.register(KeywordSkill(name="weather", keyword="weather"))
    assert await skill_registry.find_matching("hello there") == []

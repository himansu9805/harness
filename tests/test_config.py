"""Tests for Settings loading and the cached get_settings accessor."""

from harness.core import config
from harness.core.config import Settings, get_settings


def test_defaults(settings):
    assert settings.app_name == "Harness"
    assert settings.debug is False
    assert settings.mcp_request_timeout_seconds == 30
    assert settings.log_level == "INFO"
    assert settings.ollama_base_url == "http://localhost:11434"


def test_env_overrides(monkeypatch):
    monkeypatch.setenv("APP_NAME", "Custom")
    monkeypatch.setenv("DEBUG", "true")
    monkeypatch.setenv("MCP_REQUEST_TIMEOUT_SECONDS", "5")
    loaded = Settings(_env_file=None)
    assert loaded.app_name == "Custom"
    assert loaded.debug is True
    assert loaded.mcp_request_timeout_seconds == 5


def test_extra_env_vars_are_ignored(monkeypatch):
    monkeypatch.setenv("SOMETHING_UNRELATED", "value")
    # Should not raise despite the unknown variable.
    Settings(_env_file=None)


def test_get_settings_is_cached():
    config.get_settings.cache_clear()
    first = get_settings()
    second = get_settings()
    assert first is second
    config.get_settings.cache_clear()

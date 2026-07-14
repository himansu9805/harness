"""Tests for logging helpers."""

import logging

from harness.core import config
from harness.core.logging import configure_logging, get_logger


def test_get_logger_returns_named_logger():
    logger = get_logger("harness.test")
    assert isinstance(logger, logging.Logger)
    assert logger.name == "harness.test"


def test_get_logger_is_idempotent():
    assert get_logger("harness.same") is get_logger("harness.same")


def test_configure_logging_applies_configured_level(monkeypatch):
    monkeypatch.setenv("LOG_LEVEL", "WARNING")
    config.get_settings.cache_clear()
    try:
        configure_logging()
        assert logging.getLogger().level == logging.WARNING
    finally:
        config.get_settings.cache_clear()

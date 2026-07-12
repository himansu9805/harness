"""Structured logging setup."""

import logging
import sys

from harness.core.config import get_settings


def configure_logging() -> None:
    """Configure root logging using the level from :func:`get_settings`."""
    settings = get_settings()
    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=sys.stdout,
    )


def get_logger(name: str) -> logging.Logger:
    """Return a module-scoped logger for ``name`` (typically ``__name__``)."""
    return logging.getLogger(name)

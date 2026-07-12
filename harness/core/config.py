"""Harness settings loaded from environment."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings sourced from environment variables and ``.env``."""

    model_config = SettingsConfigDict(
        env_file=".env", env_nested_delimiter="__", extra="ignore"
    )

    # Application Configuration
    app_name: str = "Harness"
    debug: bool = False

    # LLM Provider
    ollama_model: str = "ollama_chat/qwen3.5:9b-mlx"
    ollama_base_url: str = "http://localhost:11434"

    # MCP Configuration
    mcp_servers_config_path: str = "configs/mcp_servers.yaml"
    mcp_request_timeout_seconds: int = 30

    # Memory and Persistance
    database_url: str = "sqlite+aiosqlite:///./data/app.db"
    redis_url: str = ""

    # Observability
    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide :class:`Settings`, constructed once and cached."""
    return Settings()

"""Exceptions classes."""

from enum import Enum


class ErrorCode(Enum):
    """Enums for Error Codes."""

    UNKNOWN = (1000, False, False)
    VALIDATION_ERROR = (1001, False, True)
    NOT_FOUND = (1002, False, True)
    UNAUTHORIZED = (1003, False, False)
    FORBIDDEN = (1004, False, False)
    TIMEOUT = (1005, True, True)
    RATE_LIMITED = (1006, True, False)
    UNSUPPORTED = (1007, False, True)
    CONFIG_INVALID = (1100, False, False)
    CONFIG_NOT_FOUND = (1101, False, False)
    LLM_PROVIDER_ERROR = (2000, True, False)
    LLM_TIMEOUT = (2001, True, False)
    LLM_RATE_LIMITED = (2002, True, False)
    LLM_RESPONSE_INVALID = (2003, True, False)  # unparseable / malformed reply
    MODEL_NOT_FOUND = (2100, False, False)  # not pulled locally
    MODEL_UNSUPPORTED = (2101, False, False)  # lacks native tool support
    OLLAMA_UNREACHABLE = (2102, True, False)  # daemon down / wrong host
    TOOL_NOT_FOUND = (3000, False, True)  # hallucinated tool name
    TOOL_VALIDATION_ERROR = (3001, False, True)  # args failed the schema
    TOOL_EXECUTION_ERROR = (3002, False, True)  # tool raised at runtime
    TOOL_TIMEOUT = (3003, True, True)
    MCP_CONNECTION_ERROR = (4000, True, False)  # can't reach the server
    MCP_PROTOCOL_ERROR = (4001, False, False)  # bad handshake / protocol
    MCP_TOOL_ERROR = (4002, False, True)  # remote tool call failed
    MCP_TIMEOUT = (4003, True, True)
    SKILL_NOT_FOUND = (5000, False, True)
    SKILL_LOAD_ERROR = (5001, False, False)  # found but failed to parse
    HOOK_BLOCKED = (6000, False, True)  # policy denial -> tell the model why
    HOOK_ERROR = (6001, False, False)  # a hook itself raised
    MAX_STEPS_EXCEEDED = (7000, False, False)
    BUDGET_EXCEEDED = (7001, False, False)
    NO_PROGRESS = (7002, False, False)  # stalled: identical calls
    WORKSPACE_ERROR = (8000, False, False)
    IO_ERROR = (8001, True, False)
    INTERNAL_ERROR = (9000, False, False)

    def __init__(self, code: int, retryable: bool, surface: bool):
        self.code = code
        self.retryable = retryable
        self.surface_to_model = surface

    def __str__(self) -> str:  # e.g. "TOOL_EXECUTION_ERROR (3002)"
        return f"{self.name} ({self.code})"


class HarnessError(Exception):
    """Base for all harness errors.

    Raise the base directly for one-off cases:
        raise HarnessError("bad thing", code=ErrorCode.VALIDATION_ERROR)
        or use a domain subclass below, which sets the code for you.
    """

    code: ErrorCode = ErrorCode.UNKNOWN

    def __init__(
        self,
        message: str = "",
        *,
        code: ErrorCode | None = None,
        cause: Exception | None = None,
    ):
        self.error_code = code or self.code
        self.message = message or self.error_code.name

        super().__init__(f"[{self.error_code}] {self.message}")
        if cause is not None:
            self.__cause__ = cause

    @property
    def retryable(self) -> bool:
        """Whether retrying the failed operation may succeed."""
        return self.error_code.retryable

    @property
    def surface_to_model(self) -> bool:
        """Whether this error should be fed back to the model as an observation."""
        return self.error_code.surface_to_model

    def as_observation(self) -> str:
        """The string to feed back into the loop when surface_to_model is True."""
        return f"ERROR [{self.error_code.name}]: {self.message}"


class ValidationError(HarnessError):
    """Input failed validation."""

    code = ErrorCode.VALIDATION_ERROR


class NotFoundError(HarnessError):
    """A requested resource does not exist."""

    code = ErrorCode.NOT_FOUND


class UnauthorizedError(HarnessError):
    """The caller is not authorized for the requested action."""

    code = ErrorCode.UNAUTHORIZED


class ConfigError(HarnessError):
    """Configuration is missing or invalid."""

    code = ErrorCode.CONFIG_INVALID


class LLMProviderError(HarnessError):
    """The LLM provider call failed."""

    code = ErrorCode.LLM_PROVIDER_ERROR


class ModelNotFoundError(HarnessError):
    """The requested model is not available locally."""

    code = ErrorCode.MODEL_NOT_FOUND


class ModelUnsupportedError(HarnessError):
    """The model lacks a capability the harness requires (e.g. tool calls)."""

    code = ErrorCode.MODEL_UNSUPPORTED


class OllamaUnreachableError(HarnessError):
    """The Ollama daemon is unreachable."""

    code = ErrorCode.OLLAMA_UNREACHABLE


class ToolNotFoundError(HarnessError):
    """A tool with the requested name is not registered."""

    code = ErrorCode.TOOL_NOT_FOUND


class ToolValidationError(HarnessError):
    """Tool arguments failed schema validation."""

    code = ErrorCode.TOOL_VALIDATION_ERROR


class ToolExecutionError(HarnessError):
    """A tool raised an error while running."""

    code = ErrorCode.TOOL_EXECUTION_ERROR


class MCPConnectionError(HarnessError):
    """Could not connect to an MCP server."""

    code = ErrorCode.MCP_CONNECTION_ERROR


class MCPProtocolError(HarnessError):
    """An MCP handshake or protocol exchange was invalid."""

    code = ErrorCode.MCP_PROTOCOL_ERROR


class MCPToolError(HarnessError):
    """A remote MCP tool call returned an error."""

    code = ErrorCode.MCP_TOOL_ERROR


class SkillNotFoundError(HarnessError):
    """A skill with the requested name is not registered."""

    code = ErrorCode.SKILL_NOT_FOUND


class SkillLoadError(HarnessError):
    """A skill was found but could not be loaded or parsed."""

    code = ErrorCode.SKILL_LOAD_ERROR


class HookBlocked(HarnessError):
    """A policy hook blocked the requested action."""

    code = ErrorCode.HOOK_BLOCKED

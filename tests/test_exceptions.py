"""Tests for the error-code taxonomy and HarnessError hierarchy."""

import pytest

from harness.core.exceptions import (
    ConfigError,
    ErrorCode,
    HarnessError,
    HookBlocked,
    MCPConnectionError,
    NotFoundError,
    SkillNotFoundError,
    ToolExecutionError,
    ValidationError,
)


def test_error_code_carries_metadata():
    assert ErrorCode.TIMEOUT.code == 1005
    assert ErrorCode.TIMEOUT.retryable is True
    assert ErrorCode.TIMEOUT.surface_to_model is True


def test_error_code_str_includes_name_and_code():
    assert str(ErrorCode.TOOL_EXECUTION_ERROR) == "TOOL_EXECUTION_ERROR (3002)"


def test_error_codes_are_unique():
    codes = [member.code for member in ErrorCode]
    assert len(codes) == len(set(codes))


def test_base_error_defaults_to_unknown_code():
    err = HarnessError("something broke")
    assert err.error_code is ErrorCode.UNKNOWN
    assert err.message == "something broke"
    assert str(err) == "[UNKNOWN (1000)] something broke"


def test_base_error_blank_message_falls_back_to_code_name():
    err = HarnessError(code=ErrorCode.RATE_LIMITED)
    assert err.message == "RATE_LIMITED"


def test_explicit_code_overrides_class_default():
    err = HarnessError("bad", code=ErrorCode.VALIDATION_ERROR)
    assert err.error_code is ErrorCode.VALIDATION_ERROR


def test_subclass_sets_its_own_code():
    assert ValidationError().error_code is ErrorCode.VALIDATION_ERROR
    assert NotFoundError().error_code is ErrorCode.NOT_FOUND
    assert ConfigError().error_code is ErrorCode.CONFIG_INVALID
    assert ToolExecutionError().error_code is ErrorCode.TOOL_EXECUTION_ERROR
    assert MCPConnectionError().error_code is ErrorCode.MCP_CONNECTION_ERROR
    assert SkillNotFoundError().error_code is ErrorCode.SKILL_NOT_FOUND
    assert HookBlocked().error_code is ErrorCode.HOOK_BLOCKED


def test_retryable_and_surface_properties_read_from_code():
    err = ToolExecutionError("x")
    assert err.retryable is ErrorCode.TOOL_EXECUTION_ERROR.retryable
    assert err.surface_to_model is ErrorCode.TOOL_EXECUTION_ERROR.surface_to_model


def test_as_observation_formats_message():
    err = ToolExecutionError("tool blew up")
    assert err.as_observation() == "ERROR [TOOL_EXECUTION_ERROR]: tool blew up"


def test_cause_is_chained():
    original = ValueError("root")
    err = ToolExecutionError("wrapped", cause=original)
    assert err.__cause__ is original


def test_harness_errors_are_exceptions():
    with pytest.raises(HarnessError):
        raise NotFoundError("nope")

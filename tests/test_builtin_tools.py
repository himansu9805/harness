"""Tests for the built-in ShellTool and output truncation helper."""

import pytest

from harness.core.exceptions import ToolExecutionError
from harness.tools.builtin import (
    MAX_OUTPUT_CHARS,
    ShellTool,
    _decode_and_truncate,
)


@pytest.fixture
def shell() -> ShellTool:
    return ShellTool()


async def test_runs_command_and_captures_stdout(shell):
    result = await shell.execute({"command": "echo hello"})
    assert result["exit_code"] == 0
    assert result["stdout"].strip() == "hello"
    assert result["stderr"] == ""


async def test_captures_stderr_and_nonzero_exit(shell):
    result = await shell.execute({"command": "echo oops >&2; exit 3"})
    assert result["exit_code"] == 3
    assert result["stderr"].strip() == "oops"


async def test_respects_workdir(shell, tmp_path):
    result = await shell.execute({"command": "pwd", "workdir": str(tmp_path)})
    # macOS may prefix /private; compare the resolved tail.
    assert result["stdout"].strip().endswith(str(tmp_path).lstrip("/"))


async def test_missing_command_raises(shell):
    with pytest.raises(ToolExecutionError):
        await shell.execute({})


async def test_non_string_command_raises(shell):
    with pytest.raises(ToolExecutionError):
        await shell.execute({"command": 123})


async def test_timeout_kills_command(shell):
    with pytest.raises(ToolExecutionError) as excinfo:
        await shell.execute({"command": "sleep 5", "timeout": 1})
    assert "timed out" in str(excinfo.value).lower()


def test_decode_and_truncate_passes_short_output():
    assert _decode_and_truncate(b"short") == "short"


def test_decode_and_truncate_caps_long_output():
    raw = b"x" * (MAX_OUTPUT_CHARS + 100)
    out = _decode_and_truncate(raw)
    assert out.startswith("x" * MAX_OUTPUT_CHARS)
    assert "truncated 100 characters" in out


def test_decode_and_truncate_replaces_invalid_bytes():
    assert _decode_and_truncate(b"\xff\xfe") != ""

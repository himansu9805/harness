"""Tests for the Pydantic schema models (chat, tool, skill)."""

import pytest
from pydantic import ValidationError as PydanticValidationError

from harness.schemas.chat import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    ToolCall,
)
from harness.schemas.skill import SkillInvocationResult, SkillManifest
from harness.schemas.tool import (
    ToolDefinition,
    ToolInvocationRequest,
    ToolInvocationResult,
)


def test_chat_message_defaults():
    msg = ChatMessage(role="user", content="hi")
    assert msg.name is None
    assert msg.tool_call_id is None
    assert msg.tool_calls is None


def test_chat_message_rejects_bad_role():
    with pytest.raises(PydanticValidationError):
        ChatMessage(role="robot", content="hi")


def test_chat_message_exclude_none_dump():
    dumped = ChatMessage(role="user", content="hi").model_dump(exclude_none=True)
    assert dumped == {"role": "user", "content": "hi"}


def test_chat_request_defaults():
    req = ChatRequest(messages=[ChatMessage(role="user", content="hi")])
    assert req.session_id is None
    assert req.stream is False
    assert req.tool_choice == "auto"
    assert req.metadata == {}


def test_chat_request_rejects_bad_tool_choice():
    with pytest.raises(PydanticValidationError):
        ChatRequest(
            messages=[ChatMessage(role="user", content="hi")],
            tool_choice="maybe",
        )


def test_tool_call_defaults_to_builtin_source():
    call = ToolCall(id="1", name="shell", arguments={"command": "ls"})
    assert call.source == "builtin"


def test_chat_response_defaults():
    resp = ChatResponse(
        session_id="s1", message=ChatMessage(role="assistant", content="ok")
    )
    assert resp.tool_calls == []
    assert resp.usage == {}


def test_tool_definition_requires_valid_source():
    with pytest.raises(PydanticValidationError):
        ToolDefinition(name="x", description="d", source="nope")


def test_tool_definition_defaults():
    td = ToolDefinition(name="x", description="d", source="builtin")
    assert td.parameters == {}
    assert td.server_label is None


def test_tool_invocation_request_default_arguments():
    req = ToolInvocationRequest(name="shell")
    assert req.arguments == {}


def test_tool_invocation_result_defaults():
    res = ToolInvocationResult(name="shell", output="ok")
    assert res.is_error is False
    assert res.duration_ms is None


def test_skill_manifest_defaults():
    manifest = SkillManifest(name="s", description="d")
    assert manifest.version == "0.1.0"
    assert manifest.triggers == []
    assert manifest.required_tools == []
    assert manifest.enabled is True


def test_skill_invocation_result_defaults():
    res = SkillInvocationResult(skill_name="s", output="done")
    assert res.tool_calls_made == []

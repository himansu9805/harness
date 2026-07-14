"""Tests for FunctionCallingPlanner, with litellm mocked out."""

from types import SimpleNamespace

import pytest

from harness.agent.planners import function_calling
from harness.agent.planners.function_calling import FunctionCallingPlanner
from harness.core.exceptions import LLMProviderError
from harness.schemas.chat import ChatMessage


def _response(*, content=None, tool_calls=None):
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    choice = SimpleNamespace(message=message)
    usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5)
    return SimpleNamespace(choices=[choice], usage=usage)


@pytest.fixture
def planner(settings):
    return FunctionCallingPlanner(settings=settings)


@pytest.fixture
def messages():
    return [ChatMessage(role="user", content="hi")]


async def test_respond_step_when_no_tool_calls(monkeypatch, planner, messages):
    async def fake_acompletion(**kwargs):
        return _response(content="the answer")

    monkeypatch.setattr(function_calling.litellm, "acompletion", fake_acompletion)
    step = await planner.next_step(messages=messages, available_tools=[])
    assert step == {"action": "respond", "content": "the answer"}


async def test_call_tools_step_maps_tool_calls(monkeypatch, planner, messages):
    tool_call = SimpleNamespace(
        id="call_1",
        function=SimpleNamespace(name="shell", arguments='{"command": "ls"}'),
    )

    async def fake_acompletion(**kwargs):
        return _response(content="thinking", tool_calls=[tool_call])

    monkeypatch.setattr(function_calling.litellm, "acompletion", fake_acompletion)
    step = await planner.next_step(messages=messages, available_tools=[{"x": 1}])
    assert step["action"] == "call_tools"
    assert step["assistant_content"] == "thinking"
    assert step["tool_calls"] == [
        {"id": "call_1", "name": "shell", "arguments": '{"command": "ls"}'}
    ]


async def test_passes_tools_and_tool_choice(monkeypatch, planner, messages):
    captured = {}

    async def fake_acompletion(**kwargs):
        captured.update(kwargs)
        return _response(content="ok")

    monkeypatch.setattr(function_calling.litellm, "acompletion", fake_acompletion)
    tools = [{"type": "function"}]
    await planner.next_step(messages=messages, available_tools=tools)
    assert captured["tools"] == tools
    assert captured["tool_choice"] == "auto"
    assert captured["model"] == planner._settings.ollama_model
    assert captured["messages"] == [{"role": "user", "content": "hi"}]


async def test_no_tools_sends_none(monkeypatch, planner, messages):
    captured = {}

    async def fake_acompletion(**kwargs):
        captured.update(kwargs)
        return _response(content="ok")

    monkeypatch.setattr(function_calling.litellm, "acompletion", fake_acompletion)
    await planner.next_step(messages=messages, available_tools=[])
    assert captured["tools"] is None
    assert captured["tool_choice"] is None


async def test_provider_failure_wrapped(monkeypatch, planner, messages):
    async def fake_acompletion(**kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(function_calling.litellm, "acompletion", fake_acompletion)
    with pytest.raises(LLMProviderError):
        await planner.next_step(messages=messages, available_tools=[])

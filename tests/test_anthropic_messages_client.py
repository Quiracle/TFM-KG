import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.tfmkg.adapters.llm.anthropic_messages import AnthropicMessagesClient
from src.tfmkg.domain.ports.llm import LLMMessage


class _FakeResponse:
    def __init__(self, text: str):
        self.type = "text"
        self.text = text


class _FakeUsage:
    def __init__(self, input_tokens: int, output_tokens: int):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class _FakeMessageResponse:
    def __init__(self):
        self.content = [_FakeResponse("Final grounded answer.")]
        self.usage = _FakeUsage(11, 7)


class _FakeMessagesAPI:
    def __init__(self, recorder: dict):
        self._recorder = recorder

    def create(self, **kwargs):
        self._recorder["payload"] = kwargs
        return _FakeMessageResponse()


class _FakeAnthropicClient:
    def __init__(self, recorder: dict):
        self.messages = _FakeMessagesAPI(recorder)


def test_anthropic_messages_generate_maps_messages_and_usage(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict = {}

    def _fake_anthropic(*, api_key: str, timeout: int, max_retries: int):
        captured["api_key"] = api_key
        captured["timeout"] = timeout
        captured["max_retries"] = max_retries
        return _FakeAnthropicClient(captured)

    monkeypatch.setattr("src.tfmkg.adapters.llm.anthropic_messages.Anthropic", _fake_anthropic)

    client = AnthropicMessagesClient(
        api_key="test-anthropic-key",
        model="claude-sonnet-4-5",
        timeout_s=30,
    )

    result = client.generate(
        [
            LLMMessage(role="system", content="Use only provided evidence."),
            LLMMessage(role="user", content="Who painted Guernica?"),
        ],
        temperature=0.0,
        max_output_tokens=120,
    )

    assert captured["api_key"] == "test-anthropic-key"
    assert captured["timeout"] == 30
    assert captured["max_retries"] == 0
    assert captured["payload"]["model"] == "claude-sonnet-4-5"
    assert captured["payload"]["max_tokens"] == 120
    assert captured["payload"]["temperature"] == 0.0
    assert captured["payload"]["system"] == "Use only provided evidence."
    assert captured["payload"]["messages"] == [{"role": "user", "content": "Who painted Guernica?"}]
    assert result.text == "Final grounded answer."
    assert result.prompt_tokens == 11
    assert result.completion_tokens == 7


def test_anthropic_messages_generate_requires_user_or_assistant_messages() -> None:
    client = AnthropicMessagesClient(
        api_key="test-anthropic-key",
        model="claude-sonnet-4-5",
        timeout_s=30,
    )

    with pytest.raises(RuntimeError, match="at least one user/assistant message"):
        client.generate([LLMMessage(role="system", content="System only message.")])

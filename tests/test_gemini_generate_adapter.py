import sys
from types import SimpleNamespace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.tfmkg.adapters.llm import gemini_generate
from src.tfmkg.adapters.llm.gemini_generate import GeminiGenerateAdapter
from src.tfmkg.domain.ports.llm import LLMMessage


class _FakeUsage:
    def __init__(self, prompt_tokens: int, completion_tokens: int):
        self.prompt_token_count = prompt_tokens
        self.candidates_token_count = completion_tokens


class _FakeGenerateResponse:
    text = "Final grounded answer."
    usage_metadata = _FakeUsage(13, 8)


class _FakeModelsAPI:
    def __init__(self, recorder: dict):
        self._recorder = recorder

    def generate_content(self, **kwargs):
        self._recorder["payload"] = kwargs
        return _FakeGenerateResponse()


class _FakeGeminiClient:
    def __init__(self, recorder: dict):
        self.models = _FakeModelsAPI(recorder)


class _FakeAPIError(Exception):
    code = 500


class _FakeHttpOptions:
    def __init__(self, *, timeout: int, retry_options: object):
        self.timeout = timeout
        self.retry_options = retry_options

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, _FakeHttpOptions)
            and other.timeout == self.timeout
            and other.retry_options == self.retry_options
        )


class _FakeHttpRetryOptions:
    def __init__(self, *, attempts: int):
        self.attempts = attempts

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _FakeHttpRetryOptions) and other.attempts == self.attempts


def test_gemini_generate_maps_messages_config_and_usage(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict = {}

    def _fake_client(*, api_key: str, http_options: dict):
        captured["api_key"] = api_key
        captured["http_options"] = http_options
        return _FakeGeminiClient(captured)

    monkeypatch.setattr(gemini_generate, "genai", SimpleNamespace(Client=_fake_client))
    monkeypatch.setattr(gemini_generate, "errors", SimpleNamespace(APIError=_FakeAPIError))
    monkeypatch.setattr(
        gemini_generate,
        "types",
        SimpleNamespace(HttpOptions=_FakeHttpOptions, HttpRetryOptions=_FakeHttpRetryOptions),
    )

    client = GeminiGenerateAdapter(
        api_key="test-gemini-key",
        model="gemini-2.5-flash",
        timeout_s=30,
    )

    result = client.generate(
        [
            LLMMessage(role="system", content="Use only provided evidence."),
            LLMMessage(role="user", content="Who painted Guernica?"),
            LLMMessage(role="assistant", content="Previous answer."),
        ],
        temperature=0.0,
        max_output_tokens=120,
    )

    assert captured["api_key"] == "test-gemini-key"
    assert captured["http_options"] == _FakeHttpOptions(
        timeout=30000,
        retry_options=_FakeHttpRetryOptions(attempts=1),
    )
    assert captured["payload"]["model"] == "gemini-2.5-flash"
    assert captured["payload"]["config"] == {
        "temperature": 0.0,
        "max_output_tokens": 120,
        "system_instruction": "Use only provided evidence.",
    }
    assert captured["payload"]["contents"] == [
        {"role": "user", "parts": [{"text": "Who painted Guernica?"}]},
        {"role": "model", "parts": [{"text": "Previous answer."}]},
    ]
    assert result.text == "Final grounded answer."
    assert result.prompt_tokens == 13
    assert result.completion_tokens == 8


def test_gemini_generate_requires_user_or_assistant_messages(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gemini_generate, "genai", SimpleNamespace(Client=lambda **_: _FakeGeminiClient({})))
    monkeypatch.setattr(gemini_generate, "errors", SimpleNamespace(APIError=_FakeAPIError))
    monkeypatch.setattr(
        gemini_generate,
        "types",
        SimpleNamespace(HttpOptions=_FakeHttpOptions, HttpRetryOptions=_FakeHttpRetryOptions),
    )

    client = GeminiGenerateAdapter(
        api_key="test-gemini-key",
        model="gemini-2.5-flash",
        timeout_s=30,
    )

    with pytest.raises(RuntimeError, match="at least one user/assistant message"):
        client.generate([LLMMessage(role="system", content="System only message.")])

from dataclasses import dataclass
from typing import Literal, Protocol


@dataclass(frozen=True)
class LLMMessage:
    role: Literal["system", "user", "assistant"]
    content: str


@dataclass(frozen=True)
class LLMResult:
    text: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class LLMClientPort(Protocol):
    model_name: str
    provider_name: str

    def generate(
        self,
        messages: list[LLMMessage],
        *,
        temperature: float = 0.2,
        max_output_tokens: int = 300,
    ) -> LLMResult: ...

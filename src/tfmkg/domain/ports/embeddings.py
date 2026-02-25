from typing import Protocol


class EmbeddingModelPort(Protocol):
    model_name: str
    provider_name: str

    def embed_texts(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...

from typing import Protocol


class TextProvider(Protocol):
    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 64,
        temperature: float = 0.7,
        seed: int = 42,
    ) -> str:
        ...

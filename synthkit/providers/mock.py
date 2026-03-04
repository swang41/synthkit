from __future__ import annotations

from synthkit.providers.base import TextProvider


class MockProvider(TextProvider):
    """Deterministic provider used for local examples/tests without model downloads."""

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 64,
        temperature: float = 0.7,
        seed: int = 42,
    ) -> str:
        del max_new_tokens, temperature
        return f"seed={seed} :: {prompt}"[:220]

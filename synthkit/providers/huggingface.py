from __future__ import annotations

from synthkit.providers.base import TextProvider


class HuggingFaceProvider(TextProvider):
    """HF Inference API backed provider (lightweight dependency via huggingface_hub)."""

    def __init__(self, model: str, token: str | None = None) -> None:
        self.model = model
        self._token = token
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from huggingface_hub import InferenceClient
            except ImportError as exc:
                raise RuntimeError(
                    "huggingface_hub is not installed. Install with: pip install 'synthkit[hf]'"
                ) from exc
            self._client = InferenceClient(model=self.model, token=self._token)
        return self._client

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 64,
        temperature: float = 0.7,
        seed: int = 42,
    ) -> str:
        client = self._get_client()
        output = client.text_generation(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            seed=seed,
        )
        return output.strip()

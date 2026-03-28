from __future__ import annotations

from threading import Lock

from synthkit.providers.base import TextProvider


class AnthropicProvider(TextProvider):
    """Anthropic Claude provider using the Anthropic SDK."""

    def __init__(
        self,
        model: str = "claude-haiku-4-5-20251001",
        api_key: str | None = None,
    ) -> None:
        self.model = model
        self._api_key = api_key
        self._client = None
        self._lock = Lock()

    def _get_client(self):
        if self._client is not None:
            return self._client
        with self._lock:
            if self._client is not None:
                return self._client
            try:
                from anthropic import Anthropic
            except ImportError as exc:
                raise RuntimeError(
                    "anthropic is not installed. Install with: pip install 'synthkit[anthropic]'"
                ) from exc
            kwargs: dict = {}
            if self._api_key:
                kwargs["api_key"] = self._api_key
            self._client = Anthropic(**kwargs)
            return self._client

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 64,
        temperature: float = 0.7,
        seed: int = 42,
    ) -> str:
        # Note: Anthropic does not expose a seed parameter; reproducibility
        # is not guaranteed for the same inputs across calls.
        client = self._get_client()
        response = client.messages.create(
            model=self.model,
            max_tokens=max_new_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        if not response.content:
            return ""
        return response.content[0].text.strip()

from __future__ import annotations

import json
from urllib import request

from synthkit.providers.base import TextProvider


class RemoteLLMProvider(TextProvider):
    """OpenAI-compatible remote endpoint provider for hosted/self-hosted LLMs."""

    def __init__(
        self,
        endpoint_url: str,
        model: str,
        api_key: str | None = None,
        timeout_s: float = 60.0,
    ) -> None:
        self.endpoint_url = endpoint_url
        self.model = model
        self.api_key = api_key
        self.timeout_s = timeout_s

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 64,
        temperature: float = 0.7,
        seed: int = 42,
    ) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_new_tokens,
            "temperature": temperature,
            "seed": seed,
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        req = request.Request(
            self.endpoint_url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )

        with request.urlopen(req, timeout=self.timeout_s) as response:
            data = json.loads(response.read().decode("utf-8"))

        choices = data.get("choices") or []
        if not choices:
            return ""

        message = choices[0].get("message") or {}
        content = message.get("content")
        if isinstance(content, str):
            return content.strip()

        if isinstance(content, list):
            text_chunks: list[str] = []
            for part in content:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    text_chunks.append(part["text"])
            return "".join(text_chunks).strip()

        return ""

from __future__ import annotations

from threading import Lock

from synthkit.providers.base import TextProvider


class HuggingFaceProvider(TextProvider):
    """Run open-source Hugging Face models locally via transformers."""

    def __init__(self, model: str, token: str | None = None, device: str = "auto") -> None:
        self.model = model
        self.device = device
        self._token = token
        self._pipeline = None
        self._lock = Lock()

    def _get_pipeline(self):
        if self._pipeline is not None:
            return self._pipeline

        with self._lock:
            if self._pipeline is not None:
                return self._pipeline

            try:
                from transformers import pipeline
            except ImportError as exc:
                raise RuntimeError(
                    "transformers is not installed. Install with: pip install 'synthkit[local]'"
                ) from exc

            model_kwargs = {}
            if self._token:
                model_kwargs["token"] = self._token

            self._pipeline = pipeline(
                task="text-generation",
                model=self.model,
                device_map=self.device,
                **model_kwargs,
            )
            return self._pipeline

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 64,
        temperature: float = 0.7,
        seed: int = 42,
    ) -> str:
        pipe = self._get_pipeline()
        output = pipe(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            seed=seed,
            return_full_text=False,
        )
        if not output:
            return ""
        return output[0]["generated_text"].strip()

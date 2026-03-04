from __future__ import annotations

from synthkit.config import SynthKitConfig
from synthkit.providers.base import TextProvider


def augment_texts(
    texts: list[str],
    *,
    operation: str,
    provider: TextProvider,
    config: SynthKitConfig,
) -> list[dict]:
    outputs: list[dict] = []
    for i, text in enumerate(texts):
        prompt = f"Operation: {operation}\nText: {text}\nResult:"
        generated = provider.generate(
            prompt,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            seed=config.seed + i,
        )
        outputs.append(
            {
                "id": i,
                "source_text": text,
                "operation": operation,
                "augmented_text": generated,
                "seed": config.seed + i,
            }
        )
    return outputs

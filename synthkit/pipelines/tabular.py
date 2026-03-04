from __future__ import annotations

from synthkit.config import SynthKitConfig
from synthkit.providers.base import TextProvider


def synthesize_rows(
    schema: dict[str, str],
    *,
    num_rows: int,
    provider: TextProvider,
    config: SynthKitConfig,
) -> list[dict]:
    rows: list[dict] = []
    schema_text = ", ".join(f"{k}:{v}" for k, v in schema.items())
    for i in range(num_rows):
        prompt = f"Generate one row with schema [{schema_text}] as key=value pairs. Row:"
        raw = provider.generate(
            prompt,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            seed=config.seed + i,
        )
        rows.append({"row_id": i, "schema": schema, "raw": raw})
    return rows

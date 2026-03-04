from __future__ import annotations

from synthkit.config import SynthKitConfig
from synthkit.providers.base import TextProvider


def generate_retrieval_validation_set(
    chunks: list[str],
    *,
    provider: TextProvider,
    config: SynthKitConfig,
) -> list[dict]:
    records: list[dict] = []
    for i, chunk in enumerate(chunks):
        q_prompt = (
            "Write 1 retrieval-validation query that can be answered from this chunk only.\n"
            f"Chunk: {chunk}\nQuery:"
        )
        a_prompt = f"Answer this query using only the chunk.\nChunk: {chunk}\nAnswer:"

        query = provider.generate(
            q_prompt,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            seed=config.seed + i,
        )
        answer = provider.generate(
            a_prompt,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            seed=config.seed + 1000 + i,
        )

        records.append(
            {
                "chunk_id": i,
                "query": query,
                "reference_answer": answer,
                "source_chunk": chunk,
            }
        )
    return records

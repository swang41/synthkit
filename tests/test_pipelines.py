from synthkit import (
    SynthKitConfig,
    augment_texts,
    generate_retrieval_validation_set,
    synthesize_rows,
)
from synthkit.providers.mock import MockProvider


def test_text_augmentation_shape():
    out = augment_texts(
        ["hello"],
        operation="paraphrase",
        provider=MockProvider(),
        config=SynthKitConfig(seed=1),
    )
    assert out[0]["operation"] == "paraphrase"
    assert "augmented_text" in out[0]


def test_rag_validation_shape():
    out = generate_retrieval_validation_set(
        ["chunk"], provider=MockProvider(), config=SynthKitConfig(seed=1)
    )
    assert out[0]["chunk_id"] == 0
    assert "reference_answer" in out[0]


def test_tabular_shape():
    out = synthesize_rows(
        {"a": "int"}, num_rows=2, provider=MockProvider(), config=SynthKitConfig(seed=1)
    )
    assert len(out) == 2
    assert "raw" in out[0]

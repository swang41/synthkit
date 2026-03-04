# synthkit
__SynthKit__ is a lightweight Python toolkit for synthetic data generation with a Hugging Face-first approach. v0 focuses on the core: text augmentation for train/validation data, RAG retrieval-validation data generation, and tabular synthesis.

## Install
```bash
pip install -e .
pip install -e '.[hf]'
```

## Quickstart
```python
from synthkit import SynthKitConfig, augment_texts
from synthkit.providers.mock import MockProvider

provider = MockProvider()  # swap for HuggingFaceProvider(model="...") when ready
config = SynthKitConfig(seed=42)

records = augment_texts(
    ["Synthetic data helps bootstrap ML projects."],
    operation="paraphrase",
    provider=provider,
    config=config,
)
print(records[0])
```


Default model in `SynthKitConfig` is set to `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (Llama-family, HF-hosted).

## Examples
- `examples/text_augmentation_example.py`
- `examples/rag_validation_example.py`
- `examples/tabular_example.py`

## Planning
- Version 0 implementation plan: [`VERSION0_PLAN.md`](./VERSION0_PLAN.md)


Tabular synthesis supports optional prompt constraints via `requirements=`.

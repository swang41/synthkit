# synthkit
__SynthKit__ is a lightweight Python toolkit for synthetic data generation with local open-source model support and optional remote LLM endpoint support. v0 focuses on the core: text augmentation for train/validation data, RAG retrieval-validation data generation, and tabular synthesis.

## Install
```bash
pip install -e .
pip install -e '.[local]'
```

If you're in a restricted network where build isolation cannot download from PyPI, use:
```bash
pip install --no-build-isolation -e .
poetry install --only-root
```

## Quickstart
```python
from synthkit import SynthKitConfig, HuggingFaceProvider, augment_texts
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


Default model in `SynthKitConfig` is set to `TinyLlama/TinyLlama-1.1B-Chat-v1.0`.

`HuggingFaceProvider` now runs models locally with `transformers`.

For remote LLM endpoints that expose an OpenAI-compatible `/v1/chat/completions` API:

```python
from synthkit import RemoteLLMProvider

provider = RemoteLLMProvider(
    endpoint_url="http://localhost:8000/v1/chat/completions",
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    api_key="optional",
)
```

## Examples
- `examples/text_augmentation_example.py`
- `examples/rag_validation_example.py`
- `examples/tabular_example.py`
- `examples/gradio_demo.py`


## Gradio quick demo
```bash
pip install -e '.[ui]'
python examples/gradio_demo.py
```

The demo includes tabs for text augmentation, RAG validation, and tabular synthesis. Use `mock` for an instant local demo, `huggingface_local` to run open-source models locally, or `remote` for OpenAI-compatible endpoints.


## Planning
- Version 0 implementation plan: [`VERSION0_PLAN.md`](./VERSION0_PLAN.md)


Tabular synthesis supports optional prompt constraints via `requirements=`.

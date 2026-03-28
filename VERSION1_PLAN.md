# SynthKit Version 1 Plan — Data Quality & Benchmarking

## v1 objective
Add provider choice (Anthropic, OpenAI) and a fully local evaluation layer so users can
compare generation quality across providers without incurring extra API costs during scoring.

## Core principles (carried forward from v0)
1. Lightweight: optional extras keep the base install zero-dependency.
2. Focused: evaluation runs 100 % locally — no API calls during scoring.
3. Practical: clear, runnable examples; reproducible outputs where possible.

## What's new in v1

### 1) AnthropicProvider (`synthkit/providers/anthropic.py`)
- Matches the existing `TextProvider` protocol.
- Lazy client initialisation (thread-safe, identical pattern to `HuggingFaceProvider`).
- Wraps `anthropic.Anthropic.messages.create`.
- Default model: `claude-haiku-4-5-20251001` (fast, cheap for bulk generation).
- Note: Anthropic does not expose a `seed` parameter; exact reproducibility is not guaranteed.
- Install: `pip install 'synthkit[anthropic]'`

### 2) OpenAIProvider (`synthkit/providers/openai.py`)
- Matches the existing `TextProvider` protocol.
- Lazy client initialisation (thread-safe).
- Wraps `openai.OpenAI.chat.completions.create`.
- Supports `seed` for deterministic outputs (where the API honours it).
- Supports `base_url` for any OpenAI-compatible endpoint (e.g. Ollama, vLLM, Together AI).
- Default model: `gpt-4o-mini`.
- Install: `pip install 'synthkit[openai]'`

### 3) LocalQualityScorer (`synthkit/quality/scorer.py`)
All model loading is lazy and thread-safe. No API calls are ever made.

| Metric | Implementation | Range | Better direction |
|---|---|---|---|
| `semantic_similarity` | `all-MiniLM-L6-v2` cosine similarity | −1 → 1 | higher |
| `perplexity` | distilgpt2 token-level perplexity | > 0 | lower |
| `ngram_diversity` | unique-bigram ratio | 0 → 1 | higher |
| `fluency` | `1 / (1 + log(perplexity))` | 0 → 1 | higher |
| `composite` | mean(clip(sim,0), diversity, fluency) | 0 → 1 | higher |

Public API:
```python
scorer = LocalQualityScorer()                  # default models
scorer.semantic_similarity(original, generated) # float
scorer.perplexity(text)                         # float
scorer.ngram_diversity(text, n=2)               # float
scorer.score(original, generated)               # dict[str, float]
```

Install: `pip install 'synthkit[quality]'`

### 4) BenchmarkRunner (`synthkit/benchmark/runner.py`)
Orchestrates a multi-provider comparison on the same text-augmentation task.

```python
from synthkit.benchmark import BenchmarkRunner
from synthkit.providers.anthropic import AnthropicProvider
from synthkit.providers.openai import OpenAIProvider

runner = BenchmarkRunner()          # creates default scorer + config
results = runner.run(
    providers={
        "claude": AnthropicProvider(api_key="..."),
        "gpt":    OpenAIProvider(api_key="..."),
    },
    texts=["Synthetic data helps bootstrap ML projects."],
    operation="paraphrase",
)
for name, res in results.items():
    print(name, res.summary)
```

`ProviderResult` fields: `provider_name`, `records`, `scores`, `summary`.
`summary` keys: `mean_semantic_similarity`, `mean_perplexity`, `mean_ngram_diversity`,
`mean_fluency`, `mean_composite`.

Install: `pip install 'synthkit[benchmark]'`

### 5) New optional extras in `pyproject.toml`

| Extra | Installs |
|---|---|
| `anthropic` | `anthropic>=0.25.0` |
| `openai` | `openai>=1.0.0` |
| `quality` | `sentence-transformers`, `transformers`, `torch` |
| `benchmark` | same as `quality` |

## Out of scope (v1)
- Streaming / async generation.
- Evaluation of RAG or tabular pipelines (only text augmentation is benchmarked for now).
- Dashboards or HTML reports.
- Fine-grained attribution / per-token analysis.
- Non-local scoring (no GPT-as-judge or API-based metrics).

## Acceptance criteria
1. `AnthropicProvider` and `OpenAIProvider` install cleanly and satisfy `TextProvider`.
2. `LocalQualityScorer` runs with no internet access once models are cached locally.
3. `BenchmarkRunner.run()` returns a `ProviderResult` per provider with all five metrics.
4. All 37 tests pass (`pytest tests/`).
5. New providers integrate transparently with existing v0 pipelines.

## Milestones

### Milestone 1: Providers
- `AnthropicProvider` + tests.
- `OpenAIProvider` + tests.

### Milestone 2: Local scorer
- `LocalQualityScorer` with semantic similarity, perplexity, n-gram diversity.
- Full test coverage (no real model loading in CI).

### Milestone 3: Benchmark runner
- `BenchmarkRunner` wiring scorer + providers.
- Tests with `MockProvider` + mocked scorer.

### Milestone 4: Docs & release
- `VERSION1_PLAN.md`.
- `pyproject.toml` extras and version bump to `0.2.0`.
- README update noting v1 features.

## Implementation notes
- `semantic_similarity` uses pure Python arithmetic over the embedding vectors,
  eliminating a hard numpy dependency in the scorer itself.
- `perplexity` imports `torch` lazily (after `_get_lm` guarantees it is available),
  so tests can inject mock models without having torch installed.
- `AnthropicProvider` silently ignores the `seed` parameter (Anthropic API does not
  expose one); callers should be aware outputs may vary across calls.

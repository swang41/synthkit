from __future__ import annotations

from dataclasses import dataclass, field

from synthkit.config import SynthKitConfig
from synthkit.pipelines.text_augmentation import augment_texts
from synthkit.providers.base import TextProvider
from synthkit.quality.scorer import LocalQualityScorer

_SUMMARY_KEYS = (
    "semantic_similarity",
    "perplexity",
    "ngram_diversity",
    "fluency",
    "composite",
)


@dataclass
class ProviderResult:
    """Benchmark results for a single provider."""

    provider_name: str
    records: list[dict] = field(default_factory=list)
    scores: list[dict] = field(default_factory=list)
    summary: dict[str, float] = field(default_factory=dict)


class BenchmarkRunner:
    """Compare multiple providers on the same text-augmentation task.

    All scoring is performed locally via :class:`~synthkit.quality.scorer.LocalQualityScorer`
    — no API calls are made during evaluation.

    Example::

        from synthkit.benchmark import BenchmarkRunner
        from synthkit.providers.anthropic import AnthropicProvider
        from synthkit.providers.openai import OpenAIProvider
        from synthkit.quality import LocalQualityScorer

        runner = BenchmarkRunner()
        results = runner.run(
            providers={
                "claude": AnthropicProvider(api_key="..."),
                "gpt": OpenAIProvider(api_key="..."),
            },
            texts=["Synthetic data helps bootstrap ML projects."],
            operation="paraphrase",
        )
        for name, res in results.items():
            print(name, res.summary)
    """

    def __init__(
        self,
        scorer: LocalQualityScorer | None = None,
        config: SynthKitConfig | None = None,
    ) -> None:
        self.scorer = scorer if scorer is not None else LocalQualityScorer()
        self.config = config if config is not None else SynthKitConfig()

    def run(
        self,
        providers: dict[str, TextProvider],
        texts: list[str],
        operation: str = "paraphrase",
    ) -> dict[str, ProviderResult]:
        """Run all providers on *texts* and score outputs locally.

        Args:
            providers: Mapping of display name to :class:`~synthkit.providers.base.TextProvider`.
            texts: Source texts to augment.
            operation: Augmentation operation name (e.g. ``"paraphrase"``).

        Returns:
            Mapping of provider name to :class:`ProviderResult`.
        """
        results: dict[str, ProviderResult] = {}

        for name, provider in providers.items():
            records = augment_texts(
                texts,
                operation=operation,
                provider=provider,
                config=self.config,
            )

            scores: list[dict] = []
            for record in records:
                metrics = self.scorer.score(record["source_text"], record["augmented_text"])
                metrics["id"] = record["id"]
                scores.append(metrics)

            summary: dict[str, float] = {}
            for key in _SUMMARY_KEYS:
                vals = [s[key] for s in scores if key in s]
                summary[f"mean_{key}"] = round(sum(vals) / len(vals), 4) if vals else 0.0

            results[name] = ProviderResult(
                provider_name=name,
                records=records,
                scores=scores,
                summary=summary,
            )

        return results

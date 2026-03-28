"""Tests for BenchmarkRunner — uses MockProvider and a mocked scorer."""
from unittest.mock import MagicMock

import pytest

from synthkit.benchmark.runner import BenchmarkRunner, ProviderResult
from synthkit.config import SynthKitConfig
from synthkit.providers.mock import MockProvider
from synthkit.quality.scorer import LocalQualityScorer


def _make_mock_scorer(fixed_scores: dict | None = None) -> LocalQualityScorer:
    """Return a LocalQualityScorer whose .score() always returns fixed_scores."""
    default = {
        "semantic_similarity": 0.8,
        "perplexity": 50.0,
        "ngram_diversity": 0.9,
        "fluency": 0.75,
        "composite": 0.82,
    }
    scores = fixed_scores or default
    scorer = MagicMock(spec=LocalQualityScorer)
    scorer.score.return_value = scores
    return scorer


# ---------------------------------------------------------------------------
# Basic structure
# ---------------------------------------------------------------------------


def test_run_returns_result_for_each_provider():
    runner = BenchmarkRunner(
        scorer=_make_mock_scorer(),
        config=SynthKitConfig(seed=1),
    )
    results = runner.run(
        providers={"mock_a": MockProvider(), "mock_b": MockProvider()},
        texts=["Hello world."],
        operation="paraphrase",
    )
    assert set(results.keys()) == {"mock_a", "mock_b"}
    assert all(isinstance(r, ProviderResult) for r in results.values())


def test_provider_result_fields_populated():
    runner = BenchmarkRunner(
        scorer=_make_mock_scorer(),
        config=SynthKitConfig(seed=1),
    )
    results = runner.run(
        providers={"mock": MockProvider()},
        texts=["Alpha.", "Beta."],
        operation="simplify",
    )
    res = results["mock"]
    assert res.provider_name == "mock"
    assert len(res.records) == 2
    assert len(res.scores) == 2
    for score in res.scores:
        assert "id" in score
        assert "composite" in score


def test_summary_contains_mean_keys():
    runner = BenchmarkRunner(
        scorer=_make_mock_scorer(),
        config=SynthKitConfig(seed=1),
    )
    results = runner.run(
        providers={"mock": MockProvider()},
        texts=["Test sentence."],
    )
    summary = results["mock"].summary
    expected_keys = {
        "mean_semantic_similarity",
        "mean_perplexity",
        "mean_ngram_diversity",
        "mean_fluency",
        "mean_composite",
    }
    assert expected_keys.issubset(summary.keys())


def test_summary_mean_matches_fixed_score():
    fixed = {
        "semantic_similarity": 0.6,
        "perplexity": 30.0,
        "ngram_diversity": 0.5,
        "fluency": 0.7,
        "composite": 0.6,
    }
    runner = BenchmarkRunner(
        scorer=_make_mock_scorer(fixed),
        config=SynthKitConfig(seed=1),
    )
    results = runner.run(
        providers={"mock": MockProvider()},
        texts=["One.", "Two.", "Three."],
    )
    summary = results["mock"].summary
    assert summary["mean_composite"] == round(fixed["composite"], 4)
    assert summary["mean_semantic_similarity"] == round(fixed["semantic_similarity"], 4)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_run_empty_providers_returns_empty_dict():
    runner = BenchmarkRunner(scorer=_make_mock_scorer())
    results = runner.run(providers={}, texts=["hi"])
    assert results == {}


def test_run_empty_texts_returns_empty_records():
    runner = BenchmarkRunner(scorer=_make_mock_scorer(), config=SynthKitConfig(seed=1))
    results = runner.run(providers={"mock": MockProvider()}, texts=[])
    assert results["mock"].records == []
    assert results["mock"].scores == []


def test_default_scorer_is_created_when_none():
    runner = BenchmarkRunner()
    assert isinstance(runner.scorer, LocalQualityScorer)


def test_default_config_is_created_when_none():
    runner = BenchmarkRunner()
    assert isinstance(runner.config, SynthKitConfig)

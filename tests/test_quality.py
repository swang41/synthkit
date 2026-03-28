"""Tests for LocalQualityScorer — all mocked, no real model loading required."""
import math
import sys
from unittest.mock import MagicMock, patch

import pytest

from synthkit.quality.scorer import LocalQualityScorer


# ---------------------------------------------------------------------------
# Helpers: build scorers with pre-injected lightweight mocks
# (no numpy/torch import needed in helpers themselves)
# ---------------------------------------------------------------------------


def _make_st_model(vectors: list[list[float]]) -> MagicMock:
    """Return a mock SentenceTransformer whose .encode() returns a list of lists."""
    mock = MagicMock()
    mock.encode.return_value = vectors  # plain Python lists — no numpy needed
    return mock


def _make_mock_numpy() -> MagicMock:
    """Return a MagicMock simulating the numpy functions used by semantic_similarity.

    sentence-transformers requires numpy as a transitive dependency, so numpy is
    always present in production. In tests we inject this mock so the CI environment
    does not need numpy installed.
    """
    mock_np = MagicMock()
    mock_np.linalg = MagicMock()
    mock_np.linalg.norm = lambda v: sum(float(x) ** 2 for x in v) ** 0.5
    mock_np.dot = lambda a, b: sum(float(x) * float(y) for x, y in zip(a, b))
    return mock_np


def _make_lm(loss: float):
    """Return (mock_model, mock_tokenizer) that simulate distilgpt2 with a fixed loss."""
    fake_output = MagicMock()
    fake_output.loss = MagicMock()
    fake_output.loss.item.return_value = loss

    mock_model = MagicMock(return_value=fake_output)

    # Tokenizer returns an object with an input_ids attribute that has shape (1, N)
    fake_ids = MagicMock()
    fake_ids.shape = (1, 3)  # non-empty
    mock_tokenizer = MagicMock(return_value=MagicMock(input_ids=fake_ids))

    return mock_model, mock_tokenizer


def _make_lm_empty():
    """Return (mock_model, mock_tokenizer) that simulate an empty token sequence."""
    fake_ids = MagicMock()
    fake_ids.shape = (1, 0)  # empty
    mock_tokenizer = MagicMock(return_value=MagicMock(input_ids=fake_ids))
    mock_model = MagicMock()
    return mock_model, mock_tokenizer


# ---------------------------------------------------------------------------
# semantic_similarity
# ---------------------------------------------------------------------------


def test_semantic_similarity_identical_vectors():
    scorer = LocalQualityScorer()
    scorer._st_model = _make_st_model([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    with patch.dict(sys.modules, {"numpy": _make_mock_numpy()}):
        sim = scorer.semantic_similarity("hello", "hello")
    assert abs(sim - 1.0) < 1e-4


def test_semantic_similarity_orthogonal_vectors():
    scorer = LocalQualityScorer()
    scorer._st_model = _make_st_model([[1.0, 0.0], [0.0, 1.0]])
    with patch.dict(sys.modules, {"numpy": _make_mock_numpy()}):
        sim = scorer.semantic_similarity("foo", "bar")
    assert abs(sim - 0.0) < 1e-4


def test_semantic_similarity_zero_vector_returns_zero():
    scorer = LocalQualityScorer()
    scorer._st_model = _make_st_model([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    with patch.dict(sys.modules, {"numpy": _make_mock_numpy()}):
        result = scorer.semantic_similarity("a", "b")
    assert result == 0.0


def test_semantic_similarity_partial_overlap():
    scorer = LocalQualityScorer()
    # [1, 1] and [1, 0] → cos = 1/sqrt(2) ≈ 0.7071
    scorer._st_model = _make_st_model([[1.0, 1.0], [1.0, 0.0]])
    with patch.dict(sys.modules, {"numpy": _make_mock_numpy()}):
        sim = scorer.semantic_similarity("x", "y")
    assert abs(sim - (1.0 / (2 ** 0.5))) < 1e-4


def test_semantic_similarity_requires_sentence_transformers():
    scorer = LocalQualityScorer()
    scorer._st_model = None
    with patch.dict(sys.modules, {"sentence_transformers": None, "numpy": _make_mock_numpy()}):
        with pytest.raises(RuntimeError, match="sentence-transformers is not installed"):
            scorer.semantic_similarity("a", "b")


# ---------------------------------------------------------------------------
# perplexity
# ---------------------------------------------------------------------------


def _make_mock_torch() -> MagicMock:
    """Return a MagicMock that acts as the torch module for perplexity tests."""
    mock_torch = MagicMock()
    ctx = MagicMock()
    ctx.__enter__ = MagicMock(return_value=None)
    ctx.__exit__ = MagicMock(return_value=False)
    mock_torch.no_grad.return_value = ctx
    return mock_torch


def test_perplexity_returns_exp_loss():
    loss = 2.0
    scorer = LocalQualityScorer()
    scorer._lm_model, scorer._lm_tokenizer = _make_lm(loss)

    with patch.dict(sys.modules, {"torch": _make_mock_torch()}):
        ppl = scorer.perplexity("some text")

    assert abs(ppl - math.exp(loss)) < 1e-2


def test_perplexity_high_loss_returns_inf():
    """Loss > 709 would overflow math.exp; should return float('inf')."""
    scorer = LocalQualityScorer()
    scorer._lm_model, scorer._lm_tokenizer = _make_lm(710.0)

    with patch.dict(sys.modules, {"torch": _make_mock_torch()}):
        ppl = scorer.perplexity("some text")

    assert ppl == float("inf")


def test_perplexity_empty_sequence_returns_inf():
    scorer = LocalQualityScorer()
    scorer._lm_model, scorer._lm_tokenizer = _make_lm_empty()

    with patch.dict(sys.modules, {"torch": _make_mock_torch()}):
        ppl = scorer.perplexity("")

    assert ppl == float("inf")


def test_perplexity_requires_transformers():
    scorer = LocalQualityScorer()
    scorer._lm_model = None
    scorer._lm_tokenizer = None
    # Patch both absent so _get_lm raises our helpful RuntimeError
    with patch.dict(sys.modules, {"torch": None, "transformers": None}):
        with pytest.raises(RuntimeError, match="transformers is not installed"):
            scorer._get_lm()


# ---------------------------------------------------------------------------
# ngram_diversity
# ---------------------------------------------------------------------------


def test_ngram_diversity_all_unique():
    scorer = LocalQualityScorer()
    # "the cat sat on the mat" bigrams: (the,cat),(cat,sat),(sat,on),(on,the),(the,mat) — 5 unique/5 total
    div = scorer.ngram_diversity("the cat sat on the mat", n=2)
    assert div == 1.0


def test_ngram_diversity_all_repeated():
    scorer = LocalQualityScorer()
    # "a a a a" bigrams: (a,a),(a,a),(a,a) → 1 unique / 3 total
    div = scorer.ngram_diversity("a a a a", n=2)
    assert abs(div - 1 / 3) < 1e-6


def test_ngram_diversity_short_text_returns_zero():
    scorer = LocalQualityScorer()
    assert scorer.ngram_diversity("hello", n=2) == 0.0
    assert scorer.ngram_diversity("", n=2) == 0.0


def test_ngram_diversity_unigrams():
    scorer = LocalQualityScorer()
    # "a b a" → tokens: a, b, a → unigrams (1-gram): (a,),(b,),(a,) → 2 unique / 3 total
    div = scorer.ngram_diversity("a b a", n=1)
    assert abs(div - 2 / 3) < 1e-6


def test_ngram_diversity_trigrams():
    scorer = LocalQualityScorer()
    # "a b c a b c" trigrams: (a,b,c),(b,c,a),(c,a,b),(a,b,c) → 3 unique / 4 total
    div = scorer.ngram_diversity("a b c a b c", n=3)
    assert abs(div - 3 / 4) < 1e-6


# ---------------------------------------------------------------------------
# score (composite) — mock everything heavy
# ---------------------------------------------------------------------------


def test_score_returns_all_keys():
    scorer = LocalQualityScorer()
    scorer._st_model = _make_st_model([[1.0, 0.0], [0.9, 0.1]])

    with patch.dict(sys.modules, {"numpy": _make_mock_numpy()}):
        with patch.object(scorer, "perplexity", return_value=50.0):
            result = scorer.score("hello world", "hello world again here")

    assert set(result.keys()) == {
        "semantic_similarity",
        "perplexity",
        "ngram_diversity",
        "fluency",
        "composite",
    }
    assert result["perplexity"] == 50.0


def test_score_composite_in_unit_range():
    scorer = LocalQualityScorer()
    scorer._st_model = _make_st_model([[1.0, 0.0], [0.0, 1.0]])

    with patch.dict(sys.modules, {"numpy": _make_mock_numpy()}):
        with patch.object(scorer, "perplexity", return_value=100.0):
            res = scorer.score("quick brown fox", "lazy dog jumps over fence now")

    assert 0.0 <= res["composite"] <= 1.0
    assert 0.0 <= res["fluency"] <= 1.0


def test_score_fluency_formula():
    scorer = LocalQualityScorer()
    scorer._st_model = _make_st_model([[1.0], [1.0]])

    ppl = 20.0
    with patch.dict(sys.modules, {"numpy": _make_mock_numpy()}):
        with patch.object(scorer, "perplexity", return_value=ppl):
            res = scorer.score("test", "test sentence here now extra")

    expected_fluency = round(1.0 / (1.0 + math.log(ppl)), 4)
    assert res["fluency"] == expected_fluency


def test_score_composite_formula():
    scorer = LocalQualityScorer()
    scorer._st_model = _make_st_model([[1.0, 0.0], [1.0, 0.0]])
    text = "one two three four five"

    ppl = 10.0
    with patch.dict(sys.modules, {"numpy": _make_mock_numpy()}):
        with patch.object(scorer, "perplexity", return_value=ppl):
            res = scorer.score("anything", text)

    sim = res["semantic_similarity"]
    div = res["ngram_diversity"]
    fluency = res["fluency"]
    expected_composite = round((max(sim, 0.0) + div + fluency) / 3.0, 4)
    assert res["composite"] == expected_composite

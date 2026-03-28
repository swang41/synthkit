from __future__ import annotations

import math
from threading import Lock


class LocalQualityScorer:
    """Local quality scorer using sentence-transformers, distilgpt2, and n-gram diversity.

    All model loading is lazy and thread-safe. No API calls are made.

    Metrics returned by ``score()``:
    - ``semantic_similarity``: cosine similarity between sentence embeddings (higher = more similar)
    - ``perplexity``: distilgpt2 token-level perplexity (lower = more fluent)
    - ``ngram_diversity``: unique bigram ratio (higher = more diverse)
    - ``fluency``: 1 / (1 + log(perplexity)) normalised to [0, 1] (higher = more fluent)
    - ``composite``: mean of semantic_similarity, ngram_diversity, and fluency
    """

    def __init__(
        self,
        similarity_model: str = "all-MiniLM-L6-v2",
        perplexity_model: str = "distilgpt2",
    ) -> None:
        self._similarity_model_name = similarity_model
        self._perplexity_model_name = perplexity_model
        self._st_model = None
        self._lm_model = None
        self._lm_tokenizer = None
        self._st_lock = Lock()
        self._lm_lock = Lock()

    # ------------------------------------------------------------------
    # Lazy model loaders
    # ------------------------------------------------------------------

    def _get_st_model(self):
        if self._st_model is not None:
            return self._st_model
        with self._st_lock:
            if self._st_model is not None:
                return self._st_model
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError(
                    "sentence-transformers is not installed. "
                    "Install with: pip install 'synthkit[quality]'"
                ) from exc
            self._st_model = SentenceTransformer(self._similarity_model_name)
            return self._st_model

    def _get_lm(self):
        if self._lm_model is not None:
            return self._lm_model, self._lm_tokenizer
        with self._lm_lock:
            if self._lm_model is not None:
                return self._lm_model, self._lm_tokenizer
            try:
                import torch as _torch  # noqa: F401 — required to run transformers models
                from transformers import AutoModelForCausalLM, AutoTokenizer
            except ImportError as exc:
                raise RuntimeError(
                    "transformers is not installed. "
                    "Install with: pip install 'synthkit[quality]'"
                ) from exc
            # Load into locals first so partial failure leaves state consistent.
            tok = AutoTokenizer.from_pretrained(self._perplexity_model_name)
            mdl = AutoModelForCausalLM.from_pretrained(self._perplexity_model_name)
            mdl.eval()
            self._lm_tokenizer = tok
            self._lm_model = mdl
            return self._lm_model, self._lm_tokenizer

    # ------------------------------------------------------------------
    # Individual metrics
    # ------------------------------------------------------------------

    def semantic_similarity(self, text1: str, text2: str) -> float:
        """Cosine similarity between sentence embeddings. Range: -1 to 1."""
        import numpy as np  # always available: sentence-transformers requires numpy

        model = self._get_st_model()
        emb = model.encode([text1, text2], convert_to_numpy=True)
        a, b = emb[0], emb[1]
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0.0:
            return 0.0
        return float(np.dot(a, b) / denom)

    def perplexity(self, text: str) -> float:
        """Token-level perplexity using distilgpt2. Lower = more fluent."""
        model, tokenizer = self._get_lm()
        # torch is guaranteed importable after _get_lm() succeeds
        import torch

        encodings = tokenizer(text, return_tensors="pt")
        input_ids = encodings.input_ids
        if input_ids.shape[1] == 0:
            return float("inf")
        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss.item()
        # Guard against OverflowError on nonsensical/very short text (loss > ~709)
        if loss > 709.0:
            return float("inf")
        return math.exp(loss)

    def ngram_diversity(self, text: str, n: int = 2) -> float:
        """Ratio of unique n-grams to total n-grams. Range: 0 to 1."""
        tokens = text.lower().split()
        if len(tokens) < n:
            return 0.0
        ngrams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
        return len(set(ngrams)) / len(ngrams)

    # ------------------------------------------------------------------
    # Composite score
    # ------------------------------------------------------------------

    def score(self, original: str, generated: str) -> dict[str, float]:
        """Compute all quality metrics for a (original, generated) pair.

        Returns a dict with keys:
        ``semantic_similarity``, ``perplexity``, ``ngram_diversity``,
        ``fluency``, ``composite``.
        """
        sim = self.semantic_similarity(original, generated)
        ppl = self.perplexity(generated)
        div = self.ngram_diversity(generated)

        # Map perplexity to [0, 1]: fluency = 1 / (1 + log(max(ppl, 1)))
        fluency = 1.0 / (1.0 + math.log(max(ppl, 1.0)))

        composite = (max(sim, 0.0) + div + fluency) / 3.0

        return {
            "semantic_similarity": round(sim, 4),
            "perplexity": round(ppl, 4),
            "ngram_diversity": round(div, 4),
            "fluency": round(fluency, 4),
            "composite": round(composite, 4),
        }

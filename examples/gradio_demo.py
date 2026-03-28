from __future__ import annotations

import os
from typing import Callable

import gradio as gr

from synthkit import (
    SynthKitConfig,
    augment_texts,
    generate_retrieval_validation_set,
    synthesize_rows,
)
from synthkit.providers.huggingface import HuggingFaceProvider
from synthkit.providers.mock import MockProvider
from synthkit.providers.remote import RemoteLLMProvider


def _build_provider(provider_name: str, model: str, token: str, endpoint_url: str):
    if provider_name == "mock":
        return MockProvider()
    if provider_name == "remote":
        return RemoteLLMProvider(
            endpoint_url=endpoint_url,
            model=model,
            api_key=token or None,
        )
    return HuggingFaceProvider(model=model, token=token or None)


def _split_nonempty_lines(blob: str) -> list[str]:
    return [line.strip() for line in blob.splitlines() if line.strip()]


def _parse_schema(schema_blob: str) -> dict[str, str]:
    schema: dict[str, str] = {}
    for line in _split_nonempty_lines(schema_blob):
        key, _, value = line.partition(":")
        if not key or not value:
            raise ValueError("Schema must be one field per line in key:type format.")
        schema[key.strip()] = value.strip()
    if not schema:
        raise ValueError("Schema cannot be empty.")
    return schema


def _json_output(fn: Callable[[], list[dict]]) -> list[dict] | dict[str, str]:
    try:
        return fn()
    except Exception as exc:  # pragma: no cover - UI guardrail
        return {"error": str(exc)}


def run_text_augmentation(
    texts_blob: str,
    operation: str,
    label: str,
    provider_name: str,
    model: str,
    token: str,
    endpoint_url: str,
    max_new_tokens: int,
    temperature: float,
    seed: int,
):
    texts = _split_nonempty_lines(texts_blob)
    if not texts:
        return {"error": "Please provide at least one source text."}

    provider = _build_provider(provider_name, model, token, endpoint_url)
    config = SynthKitConfig(
        model=model,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        seed=int(seed),
    )

    return _json_output(
        lambda: augment_texts(
            texts,
            operation=operation,
            provider=provider,
            config=config,
            label=label or None,
        )
    )


def run_rag_validation(
    chunks_blob: str,
    provider_name: str,
    model: str,
    token: str,
    endpoint_url: str,
    max_new_tokens: int,
    temperature: float,
    seed: int,
):
    chunks = _split_nonempty_lines(chunks_blob)
    if not chunks:
        return {"error": "Please provide at least one knowledge chunk."}

    provider = _build_provider(provider_name, model, token, endpoint_url)
    config = SynthKitConfig(
        model=model,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        seed=int(seed),
    )

    return _json_output(
        lambda: generate_retrieval_validation_set(
            chunks,
            provider=provider,
            config=config,
        )
    )


def run_tabular(
    schema_blob: str,
    num_rows: int,
    requirements: str,
    provider_name: str,
    model: str,
    token: str,
    endpoint_url: str,
    max_new_tokens: int,
    temperature: float,
    seed: int,
):
    provider = _build_provider(provider_name, model, token, endpoint_url)
    config = SynthKitConfig(
        model=model,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        seed=int(seed),
    )

    return _json_output(
        lambda: synthesize_rows(
            _parse_schema(schema_blob),
            num_rows=num_rows,
            requirements=requirements or None,
            provider=provider,
            config=config,
        )
    )


# ---------------------------------------------------------------------------
# Quality scoring helpers
# ---------------------------------------------------------------------------

def _score_single_pair(original: str, generated: str) -> tuple[dict, object]:
    """Score one (original, generated) pair and return (scores_dict, bar_chart_fig)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from synthkit.quality.scorer import LocalQualityScorer
    except ImportError as exc:
        return {"error": str(exc)}, None

    scorer = LocalQualityScorer()
    try:
        scores = scorer.score(original.strip(), generated.strip())
    except Exception as exc:
        return {"error": str(exc)}, None

    # Build a simple bar chart
    display_keys = ["semantic_similarity", "ngram_diversity", "fluency", "composite"]
    labels = ["Semantic\nSimilarity", "N-gram\nDiversity", "Fluency", "Composite"]
    values = [scores[k] for k in display_keys]
    colors = ["#4c72b0", "#55a868", "#c44e52", "#8172b2"]

    fig, ax = plt.subplots(figsize=(7, 3.5))
    bars = ax.bar(labels, values, color=colors, edgecolor="white", width=0.5)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score (0–1)")
    ax.set_title(f"Quality metrics  |  perplexity = {scores['perplexity']:.1f}")
    ax.axhline(0.5, color="grey", linewidth=0.8, linestyle="--", alpha=0.6)
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    fig.tight_layout()
    return scores, fig


def run_quality_score(original: str, generated: str):
    if not original.strip() or not generated.strip():
        return {"error": "Both original and generated text are required."}, None
    return _score_single_pair(original, generated)


def run_batch_quality_score(pairs_blob: str):
    """Score multiple original|||generated pairs (one per line)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from synthkit.quality.scorer import LocalQualityScorer
    except ImportError as exc:
        return {"error": str(exc)}, None

    lines = [l.strip() for l in pairs_blob.splitlines() if l.strip()]
    if not lines:
        return {"error": "No pairs provided."}, None

    pairs: list[tuple[str, str]] = []
    for line in lines:
        if "|||" not in line:
            return {"error": f"Line missing '|||' separator: {line!r}"}, None
        orig, _, gen = line.partition("|||")
        pairs.append((orig.strip(), gen.strip()))

    scorer = LocalQualityScorer()
    results: list[dict] = []
    try:
        for i, (orig, gen) in enumerate(pairs):
            scores = scorer.score(orig, gen)
            scores["pair"] = i + 1
            scores["original"] = orig[:60] + ("…" if len(orig) > 60 else "")
            results.append(scores)
    except Exception as exc:
        return {"error": str(exc)}, None

    # Plot composite scores per pair
    pair_ids = [f"Pair {r['pair']}" for r in results]
    composite_scores = [r["composite"] for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].bar(pair_ids, composite_scores, color="#4c72b0", edgecolor="white")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("Composite score")
    axes[0].set_title("Composite quality per pair")
    axes[0].axhline(0.5, color="grey", linewidth=0.8, linestyle="--", alpha=0.6)
    for x, y in zip(pair_ids, composite_scores):
        axes[0].text(x, y + 0.02, f"{y:.3f}", ha="center", fontsize=8)

    metric_keys = ["semantic_similarity", "ngram_diversity", "fluency"]
    metric_labels = ["Semantic sim.", "N-gram div.", "Fluency"]
    colors = ["#4c72b0", "#55a868", "#c44e52"]
    x = range(len(pairs))
    width = 0.25
    for i, (key, label, color) in enumerate(zip(metric_keys, metric_labels, colors)):
        offset = (i - 1) * width
        vals = [r[key] for r in results]
        axes[1].bar([xi + offset for xi in x], vals, width=width, label=label, color=color, edgecolor="white")
    axes[1].set_xticks(list(x))
    axes[1].set_xticklabels(pair_ids)
    axes[1].set_ylim(0, 1.15)
    axes[1].set_ylabel("Score")
    axes[1].set_title("Metrics breakdown per pair")
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    return results, fig


# ---------------------------------------------------------------------------
# Provider benchmark helpers
# ---------------------------------------------------------------------------

def run_provider_benchmark(
    texts_blob: str,
    operation: str,
    provider_names: list[str],
    anthropic_key: str,
    openai_key: str,
    openai_base_url: str,
    anthropic_model: str,
    openai_model: str,
    max_new_tokens: int,
    temperature: float,
    seed: int,
):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from synthkit.benchmark.runner import BenchmarkRunner
    except ImportError as exc:
        return {"error": str(exc)}, None

    texts = _split_nonempty_lines(texts_blob)
    if not texts:
        return {"error": "Please provide at least one source text."}, None
    if not provider_names:
        return {"error": "Select at least one provider."}, None

    providers: dict = {}
    try:
        if "mock" in provider_names:
            providers["mock"] = MockProvider()
        if "anthropic" in provider_names:
            from synthkit.providers.anthropic import AnthropicProvider
            providers["anthropic"] = AnthropicProvider(
                model=anthropic_model,
                api_key=anthropic_key or None,
            )
        if "openai" in provider_names:
            from synthkit.providers.openai import OpenAIProvider
            providers["openai"] = OpenAIProvider(
                model=openai_model,
                api_key=openai_key or None,
                base_url=openai_base_url or None,
            )
    except Exception as exc:
        return {"error": f"Provider setup failed: {exc}"}, None

    config = SynthKitConfig(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        seed=int(seed),
    )

    try:
        runner = BenchmarkRunner(config=config)
        results = runner.run(providers=providers, texts=texts, operation=operation)
    except Exception as exc:
        return {"error": str(exc)}, None

    summaries = {name: res.summary for name, res in results.items()}

    # Plot summary metrics side-by-side
    metric_keys = ["mean_semantic_similarity", "mean_ngram_diversity", "mean_fluency", "mean_composite"]
    metric_labels = ["Semantic sim.", "N-gram div.", "Fluency", "Composite"]
    provider_list = list(summaries.keys())
    x = range(len(metric_keys))
    width = 0.8 / max(len(provider_list), 1)
    palette = ["#4c72b0", "#55a868", "#c44e52", "#8172b2", "#ccb974"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for i, (name, summary) in enumerate(summaries.items()):
        offset = (i - len(provider_list) / 2 + 0.5) * width
        vals = [summary.get(k, 0.0) for k in metric_keys]
        bars = axes[0].bar(
            [xi + offset for xi in x],
            vals,
            width=width,
            label=name,
            color=palette[i % len(palette)],
            edgecolor="white",
        )
        for bar, val in zip(bars, vals):
            axes[0].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.2f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )

    axes[0].set_xticks(list(x))
    axes[0].set_xticklabels(metric_labels, fontsize=9)
    axes[0].set_ylim(0, 1.15)
    axes[0].set_ylabel("Mean score")
    axes[0].set_title("Quality metrics by provider")
    axes[0].legend(fontsize=9)

    # Perplexity (lower is better) — separate axis
    ppl_vals = [summaries[n].get("mean_perplexity", 0.0) for n in provider_list]
    axes[1].bar(provider_list, ppl_vals, color=[palette[i % len(palette)] for i in range(len(provider_list))], edgecolor="white")
    axes[1].set_ylabel("Mean perplexity (lower = more fluent)")
    axes[1].set_title("Perplexity by provider")
    for xi, val in enumerate(ppl_vals):
        axes[1].text(xi, val + 0.5, f"{val:.1f}", ha="center", fontsize=9)

    fig.tight_layout()
    return summaries, fig


def create_demo() -> gr.Blocks:
    with gr.Blocks(title="SynthKit Demo") as demo:
        gr.Markdown(
            "# SynthKit — Synthetic Data & Quality Validation\n"
            "Generate synthetic examples and validate quality interactively."
        )

        with gr.Accordion("Shared generation settings", open=True):
            provider_name = gr.Radio(
                ["mock", "huggingface_local", "remote"],
                value="mock",
                label="Provider",
            )
            model = gr.Textbox(
                value="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                label="Model (local or remote provider)",
            )
            token = gr.Textbox(
                value=os.environ.get("HF_TOKEN", ""),
                type="password",
                label="Token/API key (optional)",
            )
            endpoint_url = gr.Textbox(
                value=os.environ.get("LLM_ENDPOINT_URL", "http://localhost:8000/v1/chat/completions"),
                label="Remote endpoint URL (OpenAI-compatible)",
            )
            max_new_tokens = gr.Slider(16, 512, value=64, step=8, label="max_new_tokens")
            temperature = gr.Slider(0.0, 1.5, value=0.7, step=0.1, label="temperature")
            seed = gr.Number(value=42, precision=0, label="seed")

        with gr.Tab("Text augmentation"):
            texts_blob = gr.Textbox(lines=5, label="One source text per line")
            operation = gr.Textbox(value="paraphrase", label="Operation")
            label = gr.Textbox(value="train", label="Label (optional)")
            out_aug = gr.JSON(label="Augmented records")
            gr.Button("Generate").click(
                run_text_augmentation,
                inputs=[
                    texts_blob,
                    operation,
                    label,
                    provider_name,
                    model,
                    token,
                    endpoint_url,
                    max_new_tokens,
                    temperature,
                    seed,
                ],
                outputs=out_aug,
            )

        with gr.Tab("RAG validation"):
            chunks_blob = gr.Textbox(lines=6, label="One chunk per line")
            out_rag = gr.JSON(label="Query/answer records")
            gr.Button("Generate").click(
                run_rag_validation,
                inputs=[
                    chunks_blob,
                    provider_name,
                    model,
                    token,
                    endpoint_url,
                    max_new_tokens,
                    temperature,
                    seed,
                ],
                outputs=out_rag,
            )

        with gr.Tab("Tabular synthesis"):
            schema_blob = gr.Textbox(
                lines=6,
                value="customer_id:int\nsegment:str\nchurned:bool",
                label="Schema (key:type per line)",
            )
            num_rows = gr.Slider(1, 50, value=5, step=1, label="Rows")
            requirements = gr.Textbox(
                lines=2,
                value="customer_id should be unique",
                label="Requirements (optional)",
            )
            out_tab = gr.JSON(label="Synthetic rows")
            gr.Button("Generate").click(
                run_tabular,
                inputs=[
                    schema_blob,
                    num_rows,
                    requirements,
                    provider_name,
                    model,
                    token,
                    endpoint_url,
                    max_new_tokens,
                    temperature,
                    seed,
                ],
                outputs=out_tab,
            )

        # ------------------------------------------------------------------
        # Quality scoring tab
        # ------------------------------------------------------------------
        with gr.Tab("Quality scoring"):
            gr.Markdown(
                "### Score a single (original, generated) pair\n"
                "Uses `LocalQualityScorer` — runs **fully locally**, no API calls.\n"
                "Requires `pip install 'synthkit[quality]'`."
            )
            with gr.Row():
                qs_original = gr.Textbox(
                    lines=4,
                    label="Original text",
                    placeholder="The quick brown fox jumps over the lazy dog.",
                )
                qs_generated = gr.Textbox(
                    lines=4,
                    label="Generated / augmented text",
                    placeholder="A swift auburn fox leaps across the sleepy hound.",
                )
            qs_score_btn = gr.Button("Score", variant="primary")
            with gr.Row():
                qs_out_json = gr.JSON(label="Scores")
                qs_out_chart = gr.Plot(label="Metric chart")
            qs_score_btn.click(
                run_quality_score,
                inputs=[qs_original, qs_generated],
                outputs=[qs_out_json, qs_out_chart],
            )

            gr.Markdown("---")
            gr.Markdown(
                "### Batch scoring\n"
                "One pair per line: `original text ||| generated text`"
            )
            qs_batch_input = gr.Textbox(
                lines=6,
                label="Pairs (original ||| generated)",
                placeholder=(
                    "The cat sat on the mat. ||| A feline rested upon the rug.\n"
                    "Synthetic data is useful. ||| Artificially generated data proves helpful."
                ),
            )
            qs_batch_btn = gr.Button("Score batch")
            with gr.Row():
                qs_batch_json = gr.JSON(label="All scores")
                qs_batch_chart = gr.Plot(label="Batch chart")
            qs_batch_btn.click(
                run_batch_quality_score,
                inputs=[qs_batch_input],
                outputs=[qs_batch_json, qs_batch_chart],
            )

        # ------------------------------------------------------------------
        # Provider benchmark tab
        # ------------------------------------------------------------------
        with gr.Tab("Provider benchmark"):
            gr.Markdown(
                "### Compare providers on the same texts\n"
                "Runs augmentation with each selected provider and scores outputs locally with `BenchmarkRunner`.\n"
                "Requires `pip install 'synthkit[benchmark]'`."
            )
            with gr.Row():
                bm_texts = gr.Textbox(
                    lines=4,
                    label="Source texts (one per line)",
                    placeholder="Synthetic data helps bootstrap ML projects.\nData augmentation improves model robustness.",
                )
                bm_operation = gr.Textbox(value="paraphrase", label="Operation")

            bm_providers = gr.CheckboxGroup(
                choices=["mock", "anthropic", "openai"],
                value=["mock"],
                label="Providers to compare",
            )

            with gr.Accordion("Provider credentials & models", open=False):
                with gr.Row():
                    bm_anthropic_key = gr.Textbox(
                        value=os.environ.get("ANTHROPIC_API_KEY", ""),
                        type="password",
                        label="Anthropic API key",
                    )
                    bm_anthropic_model = gr.Textbox(
                        value="claude-haiku-4-5-20251001",
                        label="Anthropic model",
                    )
                with gr.Row():
                    bm_openai_key = gr.Textbox(
                        value=os.environ.get("OPENAI_API_KEY", ""),
                        type="password",
                        label="OpenAI API key",
                    )
                    bm_openai_model = gr.Textbox(value="gpt-4o-mini", label="OpenAI model")
                    bm_openai_base_url = gr.Textbox(
                        value=os.environ.get("OPENAI_BASE_URL", ""),
                        label="OpenAI base URL (optional, e.g. Ollama)",
                    )

            with gr.Row():
                bm_max_new_tokens = gr.Slider(16, 256, value=64, step=8, label="max_new_tokens")
                bm_temperature = gr.Slider(0.0, 1.5, value=0.7, step=0.1, label="temperature")
                bm_seed = gr.Number(value=42, precision=0, label="seed")

            bm_run_btn = gr.Button("Run benchmark", variant="primary")
            with gr.Row():
                bm_out_json = gr.JSON(label="Summary scores per provider")
                bm_out_chart = gr.Plot(label="Comparison chart")
            bm_run_btn.click(
                run_provider_benchmark,
                inputs=[
                    bm_texts,
                    bm_operation,
                    bm_providers,
                    bm_anthropic_key,
                    bm_openai_key,
                    bm_openai_base_url,
                    bm_anthropic_model,
                    bm_openai_model,
                    bm_max_new_tokens,
                    bm_temperature,
                    bm_seed,
                ],
                outputs=[bm_out_json, bm_out_chart],
            )

    return demo


if __name__ == "__main__":
    create_demo().launch()

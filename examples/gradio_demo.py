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


def _build_provider(provider_name: str, model: str, token: str):
    if provider_name == "mock":
        return MockProvider()
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
    max_new_tokens: int,
    temperature: float,
    seed: int,
):
    texts = _split_nonempty_lines(texts_blob)
    if not texts:
        return {"error": "Please provide at least one source text."}

    provider = _build_provider(provider_name, model, token)
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
    max_new_tokens: int,
    temperature: float,
    seed: int,
):
    chunks = _split_nonempty_lines(chunks_blob)
    if not chunks:
        return {"error": "Please provide at least one knowledge chunk."}

    provider = _build_provider(provider_name, model, token)
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
    max_new_tokens: int,
    temperature: float,
    seed: int,
):
    provider = _build_provider(provider_name, model, token)
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


def create_demo() -> gr.Blocks:
    with gr.Blocks(title="SynthKit Demo") as demo:
        gr.Markdown("# SynthKit quick demo\nGenerate synthetic examples with Mock or Hugging Face providers.")

        with gr.Accordion("Shared generation settings", open=True):
            provider_name = gr.Radio(
                ["mock", "huggingface"],
                value="mock",
                label="Provider",
            )
            model = gr.Textbox(
                value="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                label="Model (for Hugging Face provider)",
            )
            token = gr.Textbox(
                value=os.environ.get("HF_TOKEN", ""),
                type="password",
                label="HF token (optional if already set in env)",
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
                    max_new_tokens,
                    temperature,
                    seed,
                ],
                outputs=out_tab,
            )

    return demo


if __name__ == "__main__":
    create_demo().launch()

# SynthKit Version 0 Plan (Core Only)

## v0 objective
Ship a small, usable package for synthetic data generation focused on **Hugging Face only**.

## Core principles
1. Keep it lightweight (minimal dependencies, simple install).
2. Keep it focused (only the must-have workflows).
3. Keep it practical (clear examples and reproducible outputs).

## In scope (v0)

### 1) Core package + HF provider
- Minimal package structure.
- One provider path: Hugging Face.
- Seed support for reproducible generation.

### 2) Text augmentation (MVP)
- Input: seed texts.
- Output: augmented text records with metadata.

### 3) RAG retrieval-validation data generation (MVP)
- Input: document chunks.
- Output: validation queries + reference answers + source chunk IDs.
- Purpose: evaluate retrieval quality.

### 4) Tabular synthesis (MVP)
- Input: schema (+ optional sample rows).
- Output: synthetic rows (Python/CSV/JSON).

### 5) Essential docs
- Quickstart.
- One runnable example per pipeline.
- Known limitations.

## Out of scope (v0)
- Non-Hugging Face providers.
- Advanced constraint solving.
- Benchmarking dashboards.
- Distributed/production orchestration.
- Extra UX features beyond basic usage docs.

## Milestones

### Milestone 1: Foundation
- Package skeleton.
- HF provider working end-to-end.
- Seeded generation support.

### Milestone 2: Core pipelines
- Text augmentation MVP.
- RAG retrieval-validation MVP.
- Tabular synthesis MVP.

### Milestone 3: Stabilization
- Docs cleanup.
- Critical-path tests.
- Release notes + version tag.

## Acceptance criteria
1. Package installs cleanly with a lightweight dependency footprint.
2. All three core pipelines run with Hugging Face from documented examples.
3. RAG pipeline outputs retrieval-validation artifacts (query, answer, source chunk ID).
4. Outputs are reproducible (seeded) and exportable where relevant.
5. v0 docs are sufficient for first-time users.

## Implementation order
1. HF provider + package base.
2. Text augmentation.
3. RAG retrieval-validation.
4. Tabular synthesis.
5. Docs/tests/release prep.

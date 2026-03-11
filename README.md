# Local RAG Pipeline

A local, offline Retrieval-Augmented Generation (RAG) pipeline for pharmaceutical PDF question answering.

The project runs fully on-device with a local GGUF model through `llama-cpp-python`.

## What This Program Does

This repository provides:

- A reusable Python package (`src/rag`) with `RAGPipeline`
- PDF ingestion via PyMuPDF with OCR fallback for scanned pages
- Fixed-size text chunking (`SentenceSplitter`, chunk size 512, overlap 50)
- Hybrid retrieval (vector + BM25) fused with reciprocal rerank
- Optional pharma document/page classification and query-time classification
- Single-file and multi-file PDF indexing
- Query APIs with sources and confidence scoring
- Streaming query responses with source metadata
- Notebook-based Gradio UI demo (`notebooks/rag.ipynb`)
- Unit tests for core pipeline behavior (`tests/test_rag_pipeline.py`)
- Pandas-based regression harness for answer/retrieval drift tracking

Out of scope for this repo:

- Hosted production API/server deployment
- Model training or fine-tuning workflows

## Architecture

```text
PDF(s)
  -> load_pdf (PyMuPDF)
  -> optional OCR fallback for sparse/scanned pages (Tesseract)
  -> optional pharma page classification (keyword + batched LLM)
  -> chunking (SentenceSplitter)
  -> indexing (VectorStoreIndex)
  -> retrieval (VectorIndexRetriever + BM25Retriever + reciprocal rerank)
  -> answer generation (local LlamaCPP GGUF model)
```

## Tech Stack

- LLM: local GGUF model via `llama-cpp-python`
- Retrieval framework: `llama-index`
- Embeddings: HuggingFace sentence-transformers
- PDF parsing: `pymupdf`
- OCR (optional): `pytesseract` + Tesseract binary + `pillow`
- UI demo: `gradio` in notebook

## Repository Layout

```text
RAG/
  src/rag/
    __init__.py
    pipeline.py
    demos/multi_file.py
  tests/
    conftest.py
    test_rag_pipeline.py
  notebooks/
    rag.ipynb
    storage/
  docs/
    *.pdf
  storage/
  requirements.txt
  requirements-dev.txt
  requirements-ci.txt
  pyproject.toml
  .env.example
  README.md
```

## Requirements

- Python 3.10+
- Local GGUF model file (configured with `MODEL_PATH`)
- Optional GPU for faster inference (CPU-only supported)
- Optional OCR runtime:
  - Python packages: `pytesseract`, `pillow`
  - Installed Tesseract executable discoverable on PATH (or standard Windows path)

## Installation

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1

pip install -e .
# or: pip install -r requirements.txt
```

Development dependencies:

```bash
pip install -r requirements-dev.txt
```

## Configuration

Copy `.env.example` to `.env` and set values as needed.

Key variables:

- `MODEL_PATH`
- `N_GPU_LAYERS` (`-1` all layers on GPU, `0` CPU-only)
- `EMBED_MODEL`
- `SIMILARITY_TOP_K`
- `LOG_LEVEL`

## Quick Start

### Single PDF

```python
from rag import RAGPipeline

rag = RAGPipeline()
rag.build("docs/Sample1.pdf", classify_docs=True)

answer = rag.query("What are the storage conditions?", classify=True)
print(answer)
```

### Multiple PDFs

```python
from rag import RAGPipeline

rag = RAGPipeline(persist_dir="./storage")

pdf_files = [
    "docs/Sample1.pdf",
    "docs/Sample2.pdf",
    "docs/Sample3.pdf",
]

rag.build_from_multiple_pdfs(pdf_files, classify_docs=True)
result = rag.query_with_sources("Is there a BSE/TSE declaration?", classify=True)

print(result["answer"])
for s in result["sources"]:
    print(s["file"], s["page"], s["score"])
```

## Public API

Primary class: `rag.RAGPipeline`

- `build(pdf_path, classify_docs=False)`
- `build_from_multiple_pdfs(pdf_paths, classify_docs=False, progress_callback=None)`
- `query(question, expand=False, num_expansions=3, classify=False)`
- `query_with_sources(question, expand=False, num_expansions=3, classify=False)`
- `stream_query_with_sources(question, expand=False, num_expansions=3, classify=False)`
- `expand_query(query, num_expansions=3)`
- `get_stats()`
- `get_document_details()`
- `clear_cache()`

`query_with_sources(...)` returns a dictionary containing:

- `answer`: answer text
- `sources`: chunk entries with `file`, `page`, `score`, `doc_type`, `pharma_doc_type`, and `text`
- `chunk_count`: number of source chunks
- `query_category`: query category when classification is enabled, else `None`

## Pharma Classification Categories

- `cover_letter`
- `certificate_of_quality`
- `packaging_specification`
- `bse_tse_declaration`
- `material_description`
- `supplier_qualification`
- `chain_of_custody`
- `unknown`

When `classify_docs=True` at build time, retrieval can be category-filtered with `classify=True` during queries.

## Notebook Demo

`notebooks/rag.ipynb` provides a Gradio interface for:

- Uploading one or more PDFs
- Building the index
- Asking questions
- Viewing answer sources and index statistics

## Testing and Linting

Run tests:

```bash
pytest
```

Regression harness example:

```python
from rag import RAGPipeline, RAGRegressionHarness

rag = RAGPipeline()
rag.build("docs/Sample1.pdf", classify_docs=True)

suite = RAGRegressionHarness.create_test_suite([
  {
    "test_id": "TC-001",
    "query": "What is the batch number?",
    "expected_query_category": "certificate_of_quality",
    "required_terms": "batch|lot",
    "min_sources": 1,
    "classify": True,
  }
])

harness = RAGRegressionHarness(rag)
results = harness.run(suite)
summary = harness.summarize(results)

print(summary)
results.to_csv("regression_results.csv", index=False)

# Create a visualization dashboard for this run
harness.visualize_results(results, "regression_dashboard.png")
```

If you have a baseline comparison DataFrame from `compare_to_baseline(...)`,
you can generate a drift dashboard:

```python
comparison = harness.compare_to_baseline(current_results=results, baseline_results=baseline)
harness.visualize_comparison(comparison, "baseline_comparison_dashboard.png")
```

Run linting:

```bash
pylint src
```

## Troubleshooting

- `FileNotFoundError: GGUF model not found`

  - Verify `MODEL_PATH` or pass `model_path=` explicitly.

- OCR does not run for scanned pages

  - Ensure Tesseract binary is installed and discoverable.
  - Ensure `pytesseract` and `pillow` are installed.

- `RuntimeError: Pipeline not built`

  - Call `build(...)` or `build_from_multiple_pdfs(...)` before query methods.

- Slow CPU performance
  - Use GPU offload (`N_GPU_LAYERS`) or a smaller quantized GGUF model.

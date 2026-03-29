# Local RAG Pipeline

A local, offline Retrieval-Augmented Generation (RAG) pipeline for pharmaceutical document question answering over PDFs and scanned images.

The project runs fully on-device with a local GGUF model through `llama-cpp-python`.

## What This Program Does

This repository provides:

- A reusable Python package (`src/rag`) with `RAGPipeline`
- PDF ingestion via PyMuPDF with OCR fallback for scanned pages
- Image-folder ingestion via Tesseract OCR (`build_from_images`)
- Fixed-size text chunking (`SentenceSplitter`, chunk size 128, overlap 16)
- Hybrid retrieval (vector + BM25) fused with reciprocal rerank
- Optional pharma document/page classification and query-time classification
- Single-file and multi-file PDF indexing
- Query APIs with sources and confidence scoring
- Streaming query responses with source metadata
- Notebook-based Gradio UI demo (`notebooks/rag.ipynb`)
- Unit tests for core pipeline behavior (`tests/test_rag_pipeline.py`)
- OCR accuracy and classification correctness tests (`tests/test_ocr_accuracy.py`)
- Pipeline classification dynamics tests (`tests/test_embedding_dynamics.py`)
- Pandas-based regression harness for PDF and OCR drift tracking
- Per-file regression test suites covering 13 pharmaceutical documents (SDS, batch protocols, FDA letters)
- Standalone baseline comparison script (`scripts/compare_baseline.py`)

Out of scope for this repo:

- Hosted production API/server deployment
- Model training or fine-tuning workflows

## Architecture

```text
Input
  -> PDF(s): load_pdf (PyMuPDF) + optional OCR fallback for sparse/scanned pages
  -> Image folder(s): load_images (Tesseract OCR)
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
    regression.py
    demos/multi_file.py
  scripts/
    run_regression.py
    compare_baseline.py
  tests/
    conftest.py
    test_rag_pipeline.py
    test_regression_harness.py
    test_embedding_dynamics.py
    test_ocr_accuracy.py
  notebooks/
    rag.ipynb
    rag_colab.ipynb
    storage/
  docs/
    test_1.pdf   — Pfizer-BioNTech COVID-19 Vaccine Safety Data Sheet
    test_2.pdf   — Paracetamol Solution for Infusion Safety Data Sheet
    test_3.pdf   — Zoledronic Acid Injection Safety Data Sheet
    test_4.pdf   — Ciprofloxacin Injection Safety Data Sheet
    test_5.pdf   — Cytiva AKTA ready Flow Kit supplier documents
    test_7.pdf   — BioNTech COVID-19 mRNA Vaccine Electronic Protocol (Lot FE3592)
    test_8.pdf   — BioNTech COVID-19 mRNA Vaccine Corrected Protocol (Lot FD7220)
    test_9.pdf   — BioNTech COVID-19 mRNA Vaccine Protocol (Lot FD7220)
    test_10.pdf  — FDA Response Letter: RNA Integrity / CGE Method (BLA 125742)
    test_11.pdf  — FDA Response Letter: Sterility and Endotoxin Methods (BLA 125742)
    test_12.pdf  — Technical Response: Sterility and Endotoxin Verification (BLA 125742/0)
    test_13.pdf  — FDA Response Letter: Manufacturing and Equipment (BLA 125742)
    test_14.pdf  — Technical Response: Manufacturing and Equipment (BLA 125742/0)
    image_test_1/ — scanned images test_1 (17 pages)
    image_test_2/ — scanned images test_2 (18 pages)
    image_test_3/ — scanned images test_3 (18 pages)
    image_test_4/ — scanned images test_4
    image_test_5/ — scanned images test_5
  artifacts/
    regression_results.csv
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
rag.build("docs/test_1.pdf", classify_docs=True)

answer = rag.query("What are the storage conditions?", classify=True)
print(answer)
```

### Multiple PDFs

```python
from rag import RAGPipeline

rag = RAGPipeline(persist_dir="./storage")

pdf_files = [
    "docs/test_1.pdf",
    "docs/test_2.pdf",
    "docs/test_5.pdf",
]

rag.build_from_multiple_pdfs(pdf_files, classify_docs=True)
result = rag.query_with_sources("Is there a BSE/TSE declaration?", classify=True)

print(result["answer"])
for s in result["sources"]:
    print(s["file"], s["page"], s["score"])
```

### Scanned Image Folder (OCR)

```python
from rag import RAGPipeline

rag = RAGPipeline()
rag.build_from_images("docs/image_test_1", classify_docs=False)

result = rag.query_with_sources("What is the product code?")
print(result["answer"])
```

Notes:

- `build_from_images(...)` expects image files in one folder and processes them in sorted filename order.
- OCR requires both Python packages (`pytesseract`, `pillow`) and a Tesseract executable on PATH (or standard Windows install path).

## Public API

Primary class: `rag.RAGPipeline`

- `build(pdf_path, classify_docs=False)`
- `build_from_multiple_pdfs(pdf_paths, classify_docs=False, progress_callback=None)`
- `load_images(folder_path)`
- `build_from_images(folder_path, classify_docs=False)`
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

`notebooks/rag_colab.ipynb` is a Colab-compatible version of the same notebook.

## Testing

### Test Markers

Tests are tagged with three markers:

| Marker        | Meaning                                                                                                           |
| ------------- | ----------------------------------------------------------------------------------------------------------------- |
| `unit`        | Pure-Python tests using a deterministic fake pipeline; no model or OCR required                                   |
| `integration` | Full pipeline accuracy tests that require a real GGUF model and indexed documents                                 |
| `ocr_scan`    | OCR accuracy/classification tests that require a Tesseract binary; skipped automatically when Tesseract is absent |

### Running Tests

Run all tests (unit tests run without a model or Tesseract):

```bash
pytest
```

Run only unit tests:

```bash
pytest -m unit
```

Run with verbose output:

```bash
pytest -v
```

Run a specific test module:

```bash
pytest tests/test_rag_pipeline.py -v
pytest tests/test_regression_harness.py -v
pytest tests/test_ocr_accuracy.py -v
```

Run the pipeline classification dynamics test with printed output:

```bash
pytest tests/test_embedding_dynamics.py -s
```

This module measures how quickly the classification index builds up coverage and reduces LLM fallback reliance across three independent random orderings of the document corpus. No real model or OCR is needed.

### Test Modules

| Module                       | What it tests                                                                                                       |
| ---------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| `test_rag_pipeline.py`       | Core `RAGPipeline` behavior: build, query, sources, streaming, stats                                                |
| `test_regression_harness.py` | `RAGRegressionHarness` mechanics: test suite creation, pass/fail scoring, summarize, visualize, baseline comparison |
| `test_embedding_dynamics.py` | Classification index growth dynamics across document orderings                                                      |
| `test_ocr_accuracy.py`       | OCR quality metrics, character error rate, and keyword classification for `image_test_*` folders                    |

## Regression Script and Artifacts

### Running the Regression Script

Run the full regression suite across all documents in `docs/`:

```bash
python scripts/run_regression.py
```

Specify directories explicitly:

```bash
python scripts/run_regression.py --docs-dir docs --artifacts-dir artifacts
```

Compare against a saved baseline in the same run:

```bash
python scripts/run_regression.py --docs-dir docs --artifacts-dir artifacts --baseline artifacts/baseline_results.csv
```

Override the GGUF model path:

```bash
python scripts/run_regression.py --model-path /path/to/model.gguf
```

Use persistent per-file indexes (re-creates each index from scratch per run):

```bash
python scripts/run_regression.py --index-dir .index_cache
```

### Regression Script Arguments

| Argument          | Default         | Description                                                 |
| ----------------- | --------------- | ----------------------------------------------------------- |
| `--docs-dir`      | `docs`          | Directory containing source PDFs and `image_test_*` folders |
| `--artifacts-dir` | `artifacts`     | Directory where CSV and PNG outputs are written             |
| `--baseline`      | _(empty)_       | Optional path to a baseline CSV for drift comparison        |
| `--model-path`    | _(env/default)_ | GGUF model path override                                    |
| `--index-dir`     | _(empty)_       | Root directory for per-file persistent indexes              |

### Regression Script Phases

The script runs in three sequential phases:

**Phase 1 — PDF regression**

- Iterates over every `*.pdf` in `--docs-dir`.
- Each file is indexed individually with `classify_docs=True`.
- Runs the per-document test suite with `classify=True` and `expand=True` (full pipeline path).
- Writes combined results to `artifacts/regression_results.csv` and `artifacts/regression_dashboard.png`.
- Exits with code `1` if any test fails.

**Phase 2 — OCR (image) regression**

- Discovers every `image_test_*` folder in `--docs-dir`.
- Builds an index per folder with `build_from_images(..., classify_docs=True)`.
- Runs the matching image suite (e.g. `suite_image_test1` for `image_test_1`).
- Writes combined results to `artifacts/image_regression_results.csv` and `artifacts/image_regression_dashboard.png`.
- Image folders are skipped with a warning if OCR runtime is unavailable.

**Phase 3 — Baseline comparison** _(only when `--baseline` is provided)_

- Loads the baseline CSV.
- Computes per-test drift metrics and flags regressions.
- Writes `artifacts/baseline_comparison.csv` and `artifacts/baseline_comparison_dashboard.png`.
- Exits with code `1` if any regressions are detected.

### Output Artifacts

All outputs are written to the `--artifacts-dir` directory (default: `artifacts/`).

| File                                | Phase    | Description                                                               |
| ----------------------------------- | -------- | ------------------------------------------------------------------------- |
| `regression_results.csv`            | PDF      | Pass/fail metrics for all PDF test cases                                  |
| `regression_dashboard.png`          | PDF      | Visual dashboard of PDF regression results                                |
| `image_regression_results.csv`      | OCR      | Pass/fail metrics for all OCR test cases                                  |
| `image_regression_dashboard.png`    | OCR      | Visual dashboard of OCR regression results                                |
| `baseline_comparison.csv`           | Baseline | Per-test drift vs baseline (confidence, response time, answer similarity) |
| `baseline_comparison_dashboard.png` | Baseline | Drift visualization dashboard                                             |

### Per-Document Test Suites

The regression script includes hand-authored test suites for each document in `docs/`. Each suite contains 6 test cases covering factual retrieval and hallucination resistance:

| Suite          | File          | Document                                                 |
| -------------- | ------------- | -------------------------------------------------------- |
| `suite_test1`  | `test_1.pdf`  | Pfizer-BioNTech COVID-19 Vaccine SDS                     |
| `suite_test2`  | `test_2.pdf`  | Paracetamol Solution for Infusion SDS                    |
| `suite_test3`  | `test_3.pdf`  | Zoledronic Acid Injection SDS                            |
| `suite_test4`  | `test_4.pdf`  | Ciprofloxacin Injection SDS                              |
| `suite_test5`  | `test_5.pdf`  | Cytiva AKTA ready Flow Kit supplier documents            |
| `suite_test7`  | `test_7.pdf`  | COVID-19 Vaccine Electronic Protocol (Lot FE3592)        |
| `suite_test8`  | `test_8.pdf`  | COVID-19 Vaccine Corrected Protocol (Lot FD7220)         |
| `suite_test9`  | `test_9.pdf`  | COVID-19 Vaccine Protocol (Lot FD7220)                   |
| `suite_test10` | `test_10.pdf` | FDA Response: RNA Integrity / CGE Method                 |
| `suite_test11` | `test_11.pdf` | FDA Response: Sterility and Endotoxin Methods            |
| `suite_test12` | `test_12.pdf` | Technical Response: Sterility and Endotoxin Verification |
| `suite_test13` | `test_13.pdf` | FDA Response: Manufacturing and Equipment                |
| `suite_test14` | `test_14.pdf` | Technical Response: Manufacturing and Equipment          |

OCR image suites (`suite_image_test1` through `suite_image_test5`) mirror the first five PDF suites and run against the corresponding `image_test_*` folders.

### Standalone Baseline Comparison

To compare two existing regression CSVs without re-running the full pipeline:

```bash
python scripts/compare_baseline.py \
    --current artifacts/regression_results.csv \
    --baseline artifacts/baseline_results.csv
```

Optional argument:

```bash
python scripts/compare_baseline.py \
    --current artifacts/regression_results.csv \
    --baseline artifacts/baseline_results.csv \
    --artifacts-dir artifacts
```

Outputs `baseline_comparison.csv` and `baseline_comparison_dashboard.png` to `--artifacts-dir`.
Exits with code `1` if any regressions are detected.

### Programmatic Regression Harness

Use `RAGRegressionHarness` directly in your own scripts:

```python
from rag import RAGPipeline, RAGRegressionHarness

rag = RAGPipeline()
rag.build("docs/test_1.pdf", classify_docs=True)

suite = RAGRegressionHarness.create_test_suite([
    {
        "test_id": "TC-001",
        "query": "What is the batch number?",
        "expected_query_category": "certificate_of_quality",
        "required_terms": "batch|lot",
        "min_sources": 1,
        "classify": True,
        "criticality": "high",
    }
])

harness = RAGRegressionHarness(rag)
results = harness.run(suite)
summary = harness.summarize(results)

print(summary)
results.to_csv("artifacts/regression_results.csv", index=False)

# Generate a visual dashboard
harness.visualize_results(results, "artifacts/regression_dashboard.png")
```

To compare results against a saved baseline:

```python
baseline = harness.load_results("artifacts/baseline_results.csv")
comparison = harness.compare_to_baseline(current_results=results, baseline_results=baseline)
harness.visualize_comparison(comparison, "artifacts/baseline_comparison_dashboard.png")
comparison.to_csv("artifacts/baseline_comparison.csv", index=False)
```

### Test Case Schema

Each test case dict passed to `RAGRegressionHarness.create_test_suite(...)` supports the following keys:

| Key                       | Required | Default    | Description                                                  |
| ------------------------- | -------- | ---------- | ------------------------------------------------------------ |
| `test_id`                 | Yes      | —          | Unique identifier (e.g. `T1-001`)                            |
| `query`                   | Yes      | —          | Natural language question                                    |
| `required_terms`          | No       | `""`       | Pipe-separated terms; at least one must appear in the answer |
| `min_sources`             | No       | `1`        | Minimum number of source chunks expected                     |
| `expected_query_category` | No       | `None`     | Validates the query classification category                  |
| `criticality`             | No       | `"medium"` | Severity label: `high`, `medium`, or `low`                   |
| `classify`                | No       | `False`    | Whether to enable query classification                       |
| `expand`                  | No       | `False`    | Whether to enable query expansion                            |
| `num_expansions`          | No       | `3`        | Number of query expansions when `expand=True`                |

## Run Linting

```bash
pylint src
```

## Troubleshooting

- `FileNotFoundError: GGUF model not found`

  - Verify `MODEL_PATH` or pass `model_path=` explicitly.

- OCR does not run for scanned pages

  - Ensure Tesseract binary is installed and discoverable.
  - Ensure `pytesseract` and `pillow` are installed.

- OCR image-folder build fails (`build_from_images`)

  - Verify the folder exists and contains supported files (`.png`, `.jpg`, `.jpeg`, `.tiff`, `.tif`, `.bmp`, `.gif`).
  - Ensure Tesseract runtime is available (same requirements as scanned PDF OCR).

- `RuntimeError: Pipeline not built`

  - Call `build(...)`, `build_from_multiple_pdfs(...)`, or `build_from_images(...)` before query methods.

- OCR phase is skipped in regression script

  - Ensure your docs directory contains folders named like `image_test_1`, `image_test_2`, etc.
  - Ensure OCR runtime is available; otherwise each image folder is skipped with an OCR runtime error.

- No PDF suite mapping found for a file

  - The regression script only runs suites for documents that have a matching `suite_test<N>` function in `scripts/run_regression.py`. Files without a suite are skipped with a `SKIP` message.

- Slow CPU performance
  - Use GPU offload (`N_GPU_LAYERS`) or a smaller quantized GGUF model.

# local-rag

A local, offline Retrieval-Augmented Generation (RAG) pipeline for querying PDF documents.
Runs entirely on-device — no external API calls, no data leaves the machine.

---

## Architecture

```
PDF Input
   │
   ▼
┌─────────────────────────────────────┐
│  1. LOAD  (load_pdf)                │
│  PyMuPDF → text per page            │
│  Scanned page? → Tesseract OCR      │
│  Output: List[Document] + metadata  │
└──────────────┬──────────────────────┘
               │
               ▼  (optional — classify_docs=True)
┌─────────────────────────────────────┐
│  2. CLASSIFY  (_annotate_pharma_doc_types) │
│  LLM classifies each page into one  │
│  of 8 pharma doc categories         │
│  Stored in pharma_doc_type metadata │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  3. CHUNK  (_chunk)                 │
│  SemanticSplitterNodeParser         │
│  Splits by embedding similarity,    │
│  not fixed token count              │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  4. INDEX  (_index)                 │
│  HuggingFace embeddings             │
│  FAISS in-memory vector store       │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────┐
│  5. RETRIEVE  (_build_retriever)                    │
│                                                     │
│  ┌──────────────────┐   ┌──────────────────┐        │
│  │  Vector Search   │   │  BM25 Search     │        │
│  │  (FAISS / dense) │   │  (keyword/sparse)│        │
│  └────────┬─────────┘   └────────┬─────────┘        │
│           └──────────┬───────────┘                  │
│                      ▼                              │
│           Reciprocal Rank Fusion                    │
│           (normalised re-ranking)                   │
│                                                     │
│  (optional — classify=True)                         │
│  Query classified → filter chunks by pharma type    │
└──────────────┬──────────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  6. ANSWER  (query / query_with_sources)│
│  Retrieved chunks → prompt          │
│  Local Mistral 7B GGUF (llama.cpp)  │
│  Output: answer + citations         │
└─────────────────────────────────────┘
```

---

## Stack

| Component      | Details                                                      |
| -------------- | ------------------------------------------------------------ |
| **LLM**        | Mistral 7B Instruct — local GGUF via `llama-cpp-python`      |
| **Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` (HuggingFace)       |
| **Chunking**   | Semantic — LlamaIndex `SemanticSplitterNodeParser`           |
| **Retrieval**  | Hybrid: FAISS vector + BM25, fused via reciprocal rerank     |
| **OCR**        | Tesseract — automatic fallback for scanned/image-based pages |
| **UI**         | Gradio                                                       |

---

## Requirements

- Python 3.10+
- A local Mistral GGUF model file (e.g. `mistral-7b-instruct-v0.2.Q4_K_M.gguf`)
- **GPU (recommended):** CUDA-capable GPU with CUDA 11.8+ and `nvidia-cuda-toolkit`
- **CPU-only:** Works without a GPU; expect significantly slower inference

---

## Installation

### 1. Clone and create a virtual environment

```bash
git clone <repo-url>
cd local-rag
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install pymupdf \
    llama-index llama-index-core \
    llama-index-embeddings-huggingface \
    llama-index-llms-llama-cpp \
    llama-index-retrievers-bm25 \
    llama-cpp-python \
    sentence-transformers huggingface-hub torch \
    pytesseract pillow \
    "gradio>=6.9.0" nest-asyncio
```

**GPU build of `llama-cpp-python` (CUDA):**

```bash
CMAKE_ARGS="-DLLAMA_CUDA=on" pip install llama-cpp-python --force-reinstall
```

**CPU-only build:**

```bash
pip install llama-cpp-python
```

### 3. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env` with your values:

```env
MODEL_PATH=C:\LLM Models\Mistral\mistral-7b-instruct-v0.2.Q4_K_M.gguf
N_GPU_LAYERS=-1
EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2
SIMILARITY_TOP_K=5
LOG_LEVEL=INFO
```

All variables have sensible defaults — only `MODEL_PATH` is required if your model is not at the default path.

### 4. Download a GGUF model

Download `mistral-7b-instruct-v0.2.Q4_K_M.gguf` from HuggingFace and save it locally.
The default path expected by the pipeline is:

```
C:\LLM Models\Mistral\mistral-7b-instruct-v0.2.Q4_K_M.gguf
```

You can override this at initialisation — see [Configuration](#configuration).

---

## Quick Start

### Single File

```python
from rag_pipeline import RAGPipeline

# MODEL_PATH (and other settings) are read from .env automatically.
# Pass arguments explicitly to override any env var.
rag = RAGPipeline()                      # uses .env / defaults
# rag = RAGPipeline(model_path="...")    # explicit override

rag.build("path/to/document.pdf")

answer = rag.query("What are the storage conditions?")
print(answer)
```

### Multiple Files

The pipeline now supports indexing multiple PDF files simultaneously into a unified searchable index:

```python
from rag_pipeline import RAGPipeline

rag = RAGPipeline()

# Index multiple documents at once
pdf_files = [
    "certificate_of_analysis.pdf",
    "material_specification.pdf",
    "bse_tse_declaration.pdf"
]

# Optional: Track progress
def show_progress(current, total, filename):
    print(f"Loading {current}/{total}: {filename}")

rag.build_from_multiple_pdfs(
    pdf_files,
    classify_docs=True,
    progress_callback=show_progress
)

# Query across all indexed documents
result = rag.query_with_sources("What is the BSE/TSE status?")
print(result["answer"])
for source in result["sources"]:
    print(f"Source: {source['file']}, page {source['page']}")
```

See [MULTI_FILE_FEATURE.md](MULTI_FILE_FEATURE.md) for detailed documentation on multi-file upload functionality.

---

## API Reference

### `RAGPipeline`

```python
class RAGPipeline:
    def __init__(
        self,
        model_path: str,
        embed_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        similarity_top_k: int = 5,
        num_queries: int = 1,
        n_gpu_layers: int = -1,
    ) -> None
```

#### `build(pdf_path, classify_docs)`

```python
rag.build(pdf_path: str, classify_docs: bool = False) -> None
```

Runs the full ingestion pipeline on a PDF: load → (classify) → chunk → embed → index → build retriever.
Call once per document. Calling again replaces the current index.

When `classify_docs=True`, each page is passed to the LLM and labelled with one of
eight pharmaceutical document categories (see [Classification](#classification)).
This label is stored in the `pharma_doc_type` chunk metadata field and enables
targeted retrieval when querying with `classify=True`. Adds one LLM call per page.

#### `query(question, expand, num_expansions, classify)`

```python
rag.query(
    question: str,
    expand: bool = False,
    num_expansions: int = 3,
    classify: bool = False,
) -> str
```

Returns a plain-text answer with inline citations. Set `expand=True` to enable
LLM-based query expansion before retrieval (improves recall, increases latency).
Set `classify=True` to classify the query into a pharma document category and
restrict retrieval to matching chunks (requires `classify_docs=True` at build time
for chunk filtering; classification still runs otherwise).

#### `query_with_sources(question, expand, num_expansions, classify)`

```python
rag.query_with_sources(
    question: str,
    expand: bool = False,
    num_expansions: int = 3,
    classify: bool = False,
) -> dict
```

Returns a structured result:

```python
{
    "answer": str,           # LLM answer with inline citations
    "sources": [             # Retrieved chunks
        {
            "text":           str,   # Full chunk text
            "file":           str,   # Source filename
            "page":           int,   # 1-based page number
            "score":          float, # Confidence % relative to top chunk (0–100)
            "doc_type":       str,   # "digital" or "scanned"
            "pharma_doc_type": str,  # Pharma category label ("unknown" if not classified)
        },
        ...
    ],
    "chunk_count":    int,        # Number of chunks retrieved
    "query_category": str | None, # Detected pharma category, or None if classify=False
}
```

#### `expand_query(query, num_expansions)`

```python
rag.expand_query(query: str, num_expansions: int = 3) -> List[str]
```

Uses the LLM to generate alternative phrasings of a query. Returns the original
query as the first element, followed by up to `num_expansions` alternatives.

---

## Classification

The pipeline can classify both documents and queries into one of eight pharmaceutical
document categories using the local LLM:

| Category                  | Description                             |
| ------------------------- | --------------------------------------- |
| `cover_letter`            | Accompanying cover letter               |
| `certificate_of_quality`  | CoA / CoQ document                      |
| `packaging_specification` | Packaging or labelling spec             |
| `bse_tse_declaration`     | BSE/TSE risk declaration                |
| `material_description`    | Raw material or ingredient description  |
| `supplier_qualification`  | Vendor/supplier audit or approval       |
| `chain_of_custody`        | Traceability or chain-of-custody record |
| `unknown`                 | Could not be classified                 |

### Document classification (at build time)

```python
rag.build("document.pdf", classify_docs=True)
```

Each page is sent to the LLM and labelled. The label is stored in the
`pharma_doc_type` metadata field on every chunk derived from that page.

### Query classification (at query time)

```python
answer = rag.query("What are the storage conditions?", classify=True)
result = rag.query_with_sources("Who is the supplier?", classify=True)
```

The query is classified into a pharma category. If the index was built with
`classify_docs=True`, retrieval is filtered to chunks whose `pharma_doc_type`
matches the detected category. The detected category is returned as
`result["query_category"]` in `query_with_sources`.

> **Note:** Query classification works even without `classify_docs=True` — the LLM
> still classifies the query, but no chunk filtering is applied because the metadata
> is not present.

---

## Configuration

All parameters are set at initialisation:

| Parameter          | Default            | Description                                                                       |
| ------------------ | ------------------ | --------------------------------------------------------------------------------- |
| `model_path`       | —                  | **Required.** Path to the local GGUF model file.                                  |
| `embed_model_name` | `all-MiniLM-L6-v2` | HuggingFace embedding model.                                                      |
| `similarity_top_k` | `5`                | Chunks returned per retriever before fusion.                                      |
| `num_queries`      | `1`                | Query variants for `QueryFusionRetriever`. Set `>1` to enable internal expansion. |
| `n_gpu_layers`     | `-1`               | GPU layers offloaded. `-1` = all (CUDA). `0` = CPU only.                          |

---

## Notebook UI

Open `rag.ipynb` in Jupyter and run all cells. A Gradio interface will launch at
`http://localhost:7860` with:

- **Upload panel** — drag-and-drop one or more PDFs (supports multiple file uploads), then click **Build Pipeline**
  - **Classify document pages** checkbox — when enabled, each page is classified
    by the LLM into a pharma document category before indexing (`classify_docs=True`)
  - Multiple files are processed sequentially and indexed into a unified searchable collection
  - Progress updates show which files have been loaded
- **Chat** — type questions and receive answers with source citations from all indexed documents
  - **Classify query** checkbox — when enabled, the query is classified and retrieval
    is restricted to matching document-type chunks (`classify=True`)
- **Sources panel** — per-chunk confidence scores, page references, source file names, and pharma
  document type labels; shows the detected query category when classification is active
- **Stats panel** — displays total files indexed, pages, chunks, and document type distribution

---

## Linting

Pylint is configured via [`.pylintrc`](.pylintrc) and [`pyproject.toml`](pyproject.toml).

```bash
pip install pylint
pylint rag_pipeline.py
```

---

## Troubleshooting

**`FileNotFoundError: GGUF model not found`**
The model path passed to `RAGPipeline` does not exist. Check the path and ensure
the GGUF file has been downloaded.

**`ImportError: No module named 'pytesseract'` / OCR not available**
OCR is optional. Install it to enable scanned-page support:

```bash
pip install pytesseract pillow
# Also install Tesseract binary: https://github.com/tesseract-ocr/tesseract
```

Without it, scanned pages are silently skipped.

**CUDA out of memory**
Reduce GPU offload: `RAGPipeline(..., n_gpu_layers=20)` to offload only 20 layers,
or set `n_gpu_layers=0` for CPU-only inference.

**Slow inference on CPU**
Use a more aggressively quantised model (e.g. `Q2_K` or `Q3_K_S`) or run on a
machine with a CUDA GPU.

**`RuntimeError: Pipeline not built`**
Call `rag.build("document.pdf")` before calling `query()` or `query_with_sources()`.

---

## Project Structure

```
RAG/
├── rag_pipeline.py       # Core RAGPipeline class
├── rag.ipynb             # Gradio UI notebook
├── rag_spec.md           # Feature specification
├── requirements.txt      # Runtime dependencies
├── requirements-dev.txt  # Dev dependencies (pylint, jupyter, isort)
├── .env.example          # Environment variable template
├── .env                  # Your local config (gitignored)
├── .pylintrc             # Pylint configuration
├── pyproject.toml        # Tool configuration (pylint, isort)
├── .gitignore
└── docs/                 # Sample documents
```

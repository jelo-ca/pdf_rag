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
               ▼
┌─────────────────────────────────────┐
│  2. CHUNK  (_chunk)                 │
│  SemanticSplitterNodeParser         │
│  Splits by embedding similarity,    │
│  not fixed token count              │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  3. INDEX  (_index)                 │
│  HuggingFace embeddings             │
│  FAISS in-memory vector store       │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────┐
│  4. RETRIEVE  (_build_retriever)                    │
│                                                     │
│  ┌──────────────────┐   ┌──────────────────┐       │
│  │  Vector Search   │   │  BM25 Search     │       │
│  │  (FAISS / dense) │   │  (keyword/sparse)│       │
│  └────────┬─────────┘   └────────┬─────────┘       │
│           └──────────┬───────────┘                 │
│                      ▼                              │
│           Reciprocal Rank Fusion                    │
│           (normalised re-ranking)                   │
└──────────────┬──────────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  5. ANSWER  (query / query_with_sources)│
│  Retrieved chunks → prompt          │
│  Local Mistral 7B GGUF (llama.cpp)  │
│  Output: answer + citations         │
└─────────────────────────────────────┘
```

---

## Stack

| Component | Details |
|---|---|
| **LLM** | Mistral 7B Instruct — local GGUF via `llama-cpp-python` |
| **Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` (HuggingFace) |
| **Chunking** | Semantic — LlamaIndex `SemanticSplitterNodeParser` |
| **Retrieval** | Hybrid: FAISS vector + BM25, fused via reciprocal rerank |
| **OCR** | Tesseract — automatic fallback for scanned/image-based pages |
| **UI** | Gradio |

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

#### `build(pdf_path)`

```python
rag.build(pdf_path: str) -> None
```

Runs the full ingestion pipeline on a PDF: load → chunk → embed → index → build retriever.
Call once per document. Calling again replaces the current index.

#### `query(question, expand, num_expansions)`

```python
rag.query(
    question: str,
    expand: bool = False,
    num_expansions: int = 3,
) -> str
```

Returns a plain-text answer with inline citations. Set `expand=True` to enable
LLM-based query expansion before retrieval (improves recall, increases latency).

#### `query_with_sources(question, expand, num_expansions)`

```python
rag.query_with_sources(
    question: str,
    expand: bool = False,
    num_expansions: int = 3,
) -> dict
```

Returns a structured result:

```python
{
    "answer": str,           # LLM answer with inline citations
    "sources": [             # Retrieved chunks
        {
            "text":     str,   # Full chunk text
            "file":     str,   # Source filename
            "page":     int,   # 1-based page number
            "score":    float, # Confidence % relative to top chunk (0–100)
            "doc_type": str,   # "digital" or "scanned"
        },
        ...
    ],
    "chunk_count": int,      # Number of chunks retrieved
}
```

#### `expand_query(query, num_expansions)`

```python
rag.expand_query(query: str, num_expansions: int = 3) -> List[str]
```

Uses the LLM to generate alternative phrasings of a query. Returns the original
query as the first element, followed by up to `num_expansions` alternatives.

---

## Configuration

All parameters are set at initialisation:

| Parameter | Default | Description |
|---|---|---|
| `model_path` | — | **Required.** Path to the local GGUF model file. |
| `embed_model_name` | `all-MiniLM-L6-v2` | HuggingFace embedding model. |
| `similarity_top_k` | `5` | Chunks returned per retriever before fusion. |
| `num_queries` | `1` | Query variants for `QueryFusionRetriever`. Set `>1` to enable internal expansion. |
| `n_gpu_layers` | `-1` | GPU layers offloaded. `-1` = all (CUDA). `0` = CPU only. |

---

## Notebook UI

Open `rag.ipynb` in Jupyter and run all cells. A Gradio interface will launch at
`http://localhost:7860` with:

- **Upload panel** — drag-and-drop a PDF, then click **Build Pipeline**
- **Chat** — type questions and receive answers with source citations
- **Sources panel** — per-chunk confidence scores and page references

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

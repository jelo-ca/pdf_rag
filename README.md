# local-rag

A local, offline RAG (Retrieval-Augmented Generation) pipeline for querying PDF documents via a Gradio chat UI.

## Stack

| Component | Details |
|---|---|
| **LLM** | Mistral 7B Instruct (local GGUF via `llama-cpp-python`) |
| **Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` |
| **Chunking** | Semantic (LlamaIndex `SemanticSplitterNodeParser`) |
| **Retrieval** | Hybrid — FAISS vector + BM25, fused via reciprocal rerank |
| **OCR** | Tesseract fallback for scanned/image-based pages |
| **UI** | Gradio |

## Requirements

- Python 3.10+
- A local Mistral GGUF model file (e.g. `mistral-7b-instruct-v0.2.Q4_K_M.gguf`)
- CUDA-capable GPU recommended (CPU-only mode supported)

Install dependencies:

```bash
pip install pymupdf llama-index llama-index-core llama-index-embeddings-huggingface \
    llama-index-llms-llama-cpp llama-index-retrievers-bm25 llama-cpp-python \
    sentence-transformers huggingface-hub torch pytesseract pillow "gradio>=6.9.0" nest-asyncio
```

## Usage

### Notebook (recommended)

Open `rag.ipynb`, set your model path in the **Initialize** cell, and run all cells. The Gradio UI will launch in your browser.

```python
MODEL_PATH = r"C:\LLM Models\Mistral\mistral-7b-instruct-v0.2.Q4_K_M.gguf"
rag = RAGPipeline(model_path=MODEL_PATH)
```

### Python

```python
from rag_pipeline import RAGPipeline

rag = RAGPipeline(model_path="/path/to/model.gguf")
rag.build("document.pdf")

# Simple query
answer = rag.query("What are the storage conditions?")

# Query with sources and confidence scores
result = rag.query_with_sources("What is the expiry date?")
print(result["answer"])
print(result["sources"])   # list of {file, page, score, text, doc_type}
print(result["chunk_count"])
```

### GPU vs CPU

```python
rag = RAGPipeline(model_path="...", n_gpu_layers=-1)  # GPU (default, all layers)
rag = RAGPipeline(model_path="...", n_gpu_layers=0)   # CPU only
```

## How It Works

1. **Load** — PyMuPDF extracts text page-by-page; scanned pages fall back to Tesseract OCR
2. **Chunk** — Semantic chunking splits text into meaningful segments
3. **Index** — Chunks are embedded and stored in a FAISS vector index
4. **Retrieve** — Queries hit both vector search and BM25; results are fused via reciprocal rerank
5. **Answer** — Retrieved context is passed to the local LLM with a citation-aware prompt

## Project Structure

```
RAG/
├── rag_pipeline.py   # Core RAGPipeline class
├── rag.ipynb         # Notebook with Gradio UI
├── rag_spec.md       # Feature spec
└── docs/             # Sample documents
```

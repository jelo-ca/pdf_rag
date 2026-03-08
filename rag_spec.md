# Pharmaceutical RAG — Feature Spec

---

## 1. Document Ingestion & Extraction
- Handle both digital PDFs and scanned documents
- Use OCR for scanned files where needed
- Ensure extracted text is clean and chunked properly

## 2. Metadata Tagging
- Store document type, page ranges, and source identifiers per chunk
- Use consistent metadata schema to support filtering and search

## 3. Embeddings & Indexing
- Use an open-source embedding model (e.g. `sentence-transformers`)
- Store and retrieve chunks via FAISS or LlamaIndex

## 4. Prompt Optimization
- Build prompts that are clear, concise, and grounded in retrieved context
- Instruct the model to cite sources in its answer

## 5. Model Choice
- Open-source only — no Gemini (HuggingFace or Ollama)
- Model must integrate cleanly without breaking retrieval logic

## 6. Full Retrieval Pipeline

```
Retrieval → Context Building → Prompt → Model → Answer + Sources
```

- Include confidence scores and chunk count in the response

## 7. User Interface (Gradio)
- Upload panel for documents
- Chat history display
- Answers shown with sources and confidence levels
- Clean, intuitive layout — product demo quality, not debug tool

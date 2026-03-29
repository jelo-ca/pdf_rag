"""
CI-friendly module stubs + pytest marker registration
======================================================
When heavy ML dependencies (llama-index, torch, llama-cpp-python, …) are not
installed – as is the case in lightweight CI environments – this conftest
pre-populates sys.modules with MagicMock stubs so that ``rag.pipeline``
can be imported without those packages present.

Markers
-------
unit        : pure-Python tests; no model files, OCR, or external packages needed.
integration : tests that require the real llama-index/sentence-transformers stack
              (no GPU required, but packages must be installed).
ocr_scan    : tests that require a Tesseract binary to be present on PATH.
benchmark   : timing and retrieval-quality tests that write CSV/PNG artifacts
              to the ``artifacts/`` directory.
"""

import sys
from unittest.mock import MagicMock

import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "unit: no external dependencies required")
    config.addinivalue_line("markers", "integration: requires llama-index stack (no GPU)")
    config.addinivalue_line("markers", "ocr_scan: requires Tesseract binary on PATH")
    config.addinivalue_line("markers", "benchmark: timing/quality tests; writes artifacts/")
    config.addinivalue_line("markers", "accuracy: end-to-end accuracy + system performance; requires real PDFs")
    config.addinivalue_line("markers", "truthfulqa: TruthfulQA benchmark; downloads data + requires RAG_MODEL_PATH")


def _available(module: str) -> bool:
    try:
        __import__(module)
        return True
    except ImportError:
        return False


if not _available("llama_index"):
    class _Document:
        def __init__(self, text: str = "", metadata: dict = None, **kwargs):
            self.text = text
            self.metadata = metadata or {}

    _STUBS = [
        "torch",
        "sentence_transformers",
        "llama_cpp",
        "llama_index",
        "llama_index.core",
        "llama_index.core.node_parser",
        "llama_index.core.postprocessor",
        "llama_index.core.prompts",
        "llama_index.core.query_engine",
        "llama_index.core.retrievers",
        "llama_index.core.vector_stores",
        "llama_index.core.vector_stores.types",
        "llama_index.embeddings",
        "llama_index.embeddings.huggingface",
        "llama_index.llms",
        "llama_index.llms.llama_cpp",
        "llama_index.retrievers",
        "llama_index.retrievers.bm25",
    ]
    for _mod in _STUBS:
        sys.modules.setdefault(_mod, MagicMock())

    sys.modules["llama_index.core"].Document = _Document

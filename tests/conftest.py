"""
CI-friendly module stubs
========================
When heavy ML dependencies (llama-index, torch, llama-cpp-python, …) are not
installed – as is the case in lightweight CI environments – this conftest
pre-populates sys.modules with MagicMock stubs so that ``rag.pipeline``
can be imported without those packages present.

The test fixtures in ``test_rag_pipeline.py`` patch the same symbols at the
``rag.pipeline`` namespace level (e.g. ``patch("rag.pipeline.LlamaCPP")``),
which is fully compatible with the stubs set here.

When the packages *are* installed (local dev), the real modules are used.
"""

import sys
from unittest.mock import MagicMock

import pytest


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers so pytest does not emit PytestUnknownMarkWarning."""
    config.addinivalue_line(
        "markers",
        "unit: harness-unit test using a deterministic fake pipeline (no real model required)",
    )
    config.addinivalue_line(
        "markers",
        "integration: true pipeline accuracy test requiring a real model and indexed documents",
    )

# ---------------------------------------------------------------------------
# Check whether the heavy stack is available; if not, inject stubs.
# ---------------------------------------------------------------------------

def _available(module: str) -> bool:
    try:
        __import__(module)
        return True
    except ImportError:
        return False


if not _available("llama_index"):
    # Minimal Document stub: stores text + metadata so tests can inspect them.
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

    # Inject the proper Document stub so metadata is preserved after construction.
    sys.modules["llama_index.core"].Document = _Document

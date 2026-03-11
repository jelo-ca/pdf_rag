"""
RAG Pipeline Package
====================
A high-performance pharmaceutical document question-answering system.

Main Components:
    - RAGPipeline: Core pipeline for PDF ingestion and question-answering
"""

from rag.pipeline import RAGPipeline
from rag.regression import RAGRegressionHarness, RegressionThresholds

__version__ = "0.1.0"
__all__ = ["RAGPipeline", "RAGRegressionHarness", "RegressionThresholds"]

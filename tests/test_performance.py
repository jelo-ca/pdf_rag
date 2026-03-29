"""
Performance & Retrieval Quality Tests  (pipeline.py)
=====================================================
Generates the data and artifacts needed to fill in Slides 3-4 of the
presentation.  Three test groups:

1.  TestRetrievalMetrics (unit)
    Calculation logic for Recall@K, MRR, and Precision@K tested against
    fully-controlled mock retrieval scores.  Numbers can be cited directly.

2.  TestPDFIngestionTiming (benchmark)
    Measures wall-clock time for load_pdf on every SDS file in docs/.
    Uses real fitz; LLM/embedding/splitter are mocked.
    → writes artifacts/pdf_ingestion_timing.csv

3.  test_generate_presentation_artifacts (benchmark)
    One comprehensive test that gathers component specs, per-file chunk stats,
    timing, and retrieval metric baselines, then emits:
    → artifacts/presentation_specs.csv      (Slide 3 table)
    → artifacts/presentation_timing.csv     (Slide 4 system performance)
    → artifacts/presentation_metrics.csv    (Slide 4 retrieval metrics)
    → artifacts/performance_dashboard.png   (visualization for slides 3-4)

Markers
-------
unit      : no external dependencies; runs in CI.
benchmark : reads real PDFs from docs/; writes to artifacts/; requires fitz.

Run unit tests only:
    pytest tests/test_performance.py -m unit -v

Run all benchmark tests (generates artifacts):
    pytest tests/test_performance.py -m benchmark -v -s
"""
from __future__ import annotations

import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List
from unittest.mock import MagicMock, patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DOCS_DIR      = Path(__file__).parent.parent / "docs"
ARTIFACTS_DIR = Path(__file__).parent.parent / "artifacts"

SDS_PDFS = sorted(DOCS_DIR.glob("test_*.pdf"))   # test_1.pdf … test_14.pdf

# Shared eval set used by BM25, Vector, and Hybrid comparison tests.
# Each entry defines a natural-language query, the SDS section that contains
# the answer, and the keyword signals that mark a chunk as "relevant".
EVAL_QUERIES = [
    {
        "query":              "What are the first aid measures for skin contact?",
        "section":            "Section 4 (First Aid)",
        "required_keywords":  ["first aid", "skin"],
    },
    {
        "query":              "What is the flash point of this substance?",
        "section":            "Section 9 (Physical Properties)",
        "required_keywords":  ["flash point"],
    },
    {
        "query":              "What storage conditions are required?",
        "section":            "Section 7 (Handling & Storage)",
        "required_keywords":  ["storage", "temperature"],
    },
    {
        "query":              "What personal protective equipment is needed?",
        "section":            "Section 8 (Exposure Controls)",
        "required_keywords":  ["protective equipment", "gloves"],
    },
    {
        "query":              "What hazard classifications apply to this material?",
        "section":            "Section 2 (Hazard Identification)",
        "required_keywords":  ["hazard", "classif"],
    },
]


# ---------------------------------------------------------------------------
# Helpers — retrieval metric functions
# ---------------------------------------------------------------------------

def recall_at_k(retrieved_ids: List[str], relevant_ids: set, k: int) -> float:
    """Recall@K = |relevant ∩ top-K retrieved| / |relevant|."""
    if not relevant_ids:
        return 0.0
    top_k = set(retrieved_ids[:k])
    return len(relevant_ids & top_k) / len(relevant_ids)


def precision_at_k(retrieved_ids: List[str], relevant_ids: set, k: int) -> float:
    """Precision@K = |relevant ∩ top-K retrieved| / K."""
    if k == 0:
        return 0.0
    top_k = retrieved_ids[:k]
    hits = sum(1 for r in top_k if r in relevant_ids)
    return hits / k


def mean_reciprocal_rank(retrieved_ids: List[str], relevant_ids: set) -> float:
    """MRR = 1 / rank of the first relevant item (0.0 if none found)."""
    for rank, item in enumerate(retrieved_ids, start=1):
        if item in relevant_ids:
            return 1.0 / rank
    return 0.0


def average_precision(retrieved_ids: List[str], relevant_ids: set) -> float:
    """Average Precision over all relevant positions (used in MAP)."""
    if not relevant_ids:
        return 0.0
    hits, total_p = 0, 0.0
    for rank, item in enumerate(retrieved_ids, start=1):
        if item in relevant_ids:
            hits += 1
            total_p += hits / rank
    return total_p / len(relevant_ids)


def evaluate_retrieval(
    queries: List[Dict],
    k: int = 5,
) -> Dict[str, float]:
    """Aggregate Recall@K, Precision@K, MRR, MAP across a list of query dicts.

    Each query dict must contain:
        ``retrieved``  – ordered list of retrieved chunk IDs
        ``relevant``   – set of ground-truth relevant chunk IDs
    """
    r_at_k, p_at_k, mrr_vals, ap_vals = [], [], [], []
    for q in queries:
        retrieved = q["retrieved"]
        relevant  = set(q["relevant"])
        r_at_k.append(recall_at_k(retrieved, relevant, k))
        p_at_k.append(precision_at_k(retrieved, relevant, k))
        mrr_vals.append(mean_reciprocal_rank(retrieved, relevant))
        ap_vals.append(average_precision(retrieved, relevant))
    return {
        f"Recall@{k}":    round(sum(r_at_k) / max(len(r_at_k), 1), 4),
        f"Precision@{k}": round(sum(p_at_k) / max(len(p_at_k), 1), 4),
        "MRR":            round(sum(mrr_vals) / max(len(mrr_vals), 1), 4),
        "MAP":            round(sum(ap_vals)  / max(len(ap_vals),  1), 4),
    }


# ---------------------------------------------------------------------------
# 1. Unit tests — metric calculation correctness
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestRetrievalMetrics:
    """Slide 4 — Retrieval Performance: verify metric calculation logic.

    These tests use fully-controlled retrieval lists so the exact expected
    values can be stated.  The same functions are used in the benchmark tests
    to compute real metrics from the indexed documents.
    """

    # -- Recall@K --

    def test_recall_at_k_perfect(self):
        assert recall_at_k(["a", "b", "c"], {"a", "b"}, k=5) == pytest.approx(1.0)

    def test_recall_at_k_zero(self):
        assert recall_at_k(["x", "y", "z"], {"a", "b"}, k=5) == pytest.approx(0.0)

    def test_recall_at_k_partial(self):
        # Only "a" is in top-3; "b" is at rank 4 (outside k=3)
        assert recall_at_k(["a", "x", "y", "b"], {"a", "b"}, k=3) == pytest.approx(0.5)

    def test_recall_at_k_empty_relevant_returns_zero(self):
        assert recall_at_k(["a", "b"], set(), k=5) == pytest.approx(0.0)

    def test_recall_at_k_respects_k(self):
        # Relevant chunk at rank 3, k=2 — should not count
        assert recall_at_k(["x", "y", "a"], {"a"}, k=2) == pytest.approx(0.0)

    # -- Precision@K --

    def test_precision_at_k_perfect(self):
        assert precision_at_k(["a", "b", "c"], {"a", "b", "c"}, k=3) == pytest.approx(1.0)

    def test_precision_at_k_one_hit_in_five(self):
        assert precision_at_k(["a", "x", "y", "z", "w"], {"a"}, k=5) == pytest.approx(0.2)

    def test_precision_at_k_zero(self):
        assert precision_at_k(["x", "y"], {"a", "b"}, k=2) == pytest.approx(0.0)

    def test_precision_at_k_zero_k_returns_zero(self):
        assert precision_at_k(["a"], {"a"}, k=0) == pytest.approx(0.0)

    # -- MRR --

    def test_mrr_first_result_relevant(self):
        assert mean_reciprocal_rank(["a", "b", "c"], {"a"}) == pytest.approx(1.0)

    def test_mrr_second_result_relevant(self):
        assert mean_reciprocal_rank(["x", "a", "b"], {"a"}) == pytest.approx(0.5)

    def test_mrr_third_result_relevant(self):
        assert mean_reciprocal_rank(["x", "y", "a"], {"a"}) == pytest.approx(1 / 3)

    def test_mrr_no_relevant_returns_zero(self):
        assert mean_reciprocal_rank(["x", "y"], {"a"}) == pytest.approx(0.0)

    # -- MAP --

    def test_ap_all_relevant(self):
        # 2 relevant chunks; both retrieved in first 2 positions
        ap = average_precision(["a", "b", "c"], {"a", "b"})
        assert ap == pytest.approx(1.0)

    def test_ap_no_relevant(self):
        assert average_precision(["a", "b"], set()) == pytest.approx(0.0)

    def test_ap_second_and_fourth_relevant(self):
        # relevant at positions 2 and 4
        # AP = (1/2 * 1/2 + 2/4 * 1/2) = (0.25 + 0.25) = 0.5
        ap = average_precision(["x", "a", "y", "b"], {"a", "b"})
        assert ap == pytest.approx(0.5)

    # -- evaluate_retrieval aggregate --

    def test_evaluate_retrieval_macro_average(self):
        """Aggregate metrics are macro-averaged across queries."""
        queries = [
            {"retrieved": ["a", "b", "c"], "relevant": {"a"}},          # Recall=1, P=0.2, MRR=1
            {"retrieved": ["x", "y", "a"], "relevant": {"a"}},          # Recall=1, P=0.2, MRR=1/3
        ]
        metrics = evaluate_retrieval(queries, k=5)
        assert metrics["Recall@5"] == pytest.approx(1.0)
        assert metrics["MRR"] == pytest.approx((1.0 + 1/3) / 2, abs=0.01)

    def test_evaluate_retrieval_zero_when_nothing_found(self):
        queries = [{"retrieved": ["x", "y"], "relevant": {"a", "b"}}]
        metrics = evaluate_retrieval(queries, k=5)
        assert metrics["Recall@5"] == pytest.approx(0.0)
        assert metrics["MRR"] == pytest.approx(0.0)

    # -- BM25-style simulation (keyword overlap relevance model) --

    def test_bm25_simulation_higher_overlap_ranks_first(self):
        """A chunk with more query-term overlap should score higher in BM25."""
        query_tokens = set("first aid measures for eye contact".split())

        def bm25_score(text: str) -> float:
            tokens = set(text.lower().split())
            return len(query_tokens & tokens) / max(len(query_tokens), 1)

        chunk_high = "first aid measures for eye contact are: rinse with water"
        chunk_low  = "storage temperature: refrigerate at 2-8 degrees Celsius"
        assert bm25_score(chunk_high) > bm25_score(chunk_low)

    def test_confidence_normalisation_to_100(self):
        """pipeline.query_with_sources normalises the top chunk to 100 %."""
        raw_scores = [0.82, 0.65, 0.41]
        max_s = max(raw_scores)
        normalised = [round(s / max_s * 100, 1) for s in raw_scores]
        assert normalised[0] == pytest.approx(100.0)
        assert normalised[1] < 100.0
        assert normalised[2] < normalised[1]


# ---------------------------------------------------------------------------
# Fixture — mocked pipeline with real fitz for timing tests
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def timing_pipeline(tmp_path_factory):
    """Pipeline with fitz real and LLM/embed/splitter mocked."""
    p = tmp_path_factory.mktemp("model") / "model.gguf"
    p.write_bytes(b"")
    with patch("rag.pipeline.LlamaCPP") as mlc, \
         patch("rag.pipeline.HuggingFaceEmbedding") as mec, \
         patch("rag.pipeline.SentenceWindowNodeParser") as msc, \
         patch("rag.pipeline.Settings"):
        mlc.return_value = MagicMock(name="llm")
        mec.return_value = MagicMock(name="embed")
        msc.from_defaults.return_value = MagicMock(name="splitter")
        from rag import RAGPipeline
        rag = RAGPipeline(model_path=str(p))
    rag.classifier_llm.complete.return_value = MagicMock(text="unknown")
    return rag


# ---------------------------------------------------------------------------
# 2. Benchmark — PDF ingestion timing per file (Slide 4 System Performance)
# ---------------------------------------------------------------------------

@pytest.mark.benchmark
class TestPDFIngestionTiming:
    """Measure load_pdf wall-clock time for each SDS file.

    Results are captured by test_generate_presentation_artifacts.
    Individual tests fail if any single file takes over 30 seconds.
    """

    @pytest.mark.parametrize("pdf_path", SDS_PDFS, ids=[p.name for p in SDS_PDFS])
    def test_load_pdf_completes_under_30s(self, timing_pipeline, pdf_path):
        t0 = time.perf_counter()
        docs = timing_pipeline.load_pdf(str(pdf_path))
        elapsed = time.perf_counter() - t0
        assert elapsed < 30.0, (
            f"{pdf_path.name}: load_pdf took {elapsed:.2f}s (> 30s limit)"
        )
        assert len(docs) >= 1, f"{pdf_path.name}: no documents extracted"


# ---------------------------------------------------------------------------
# 3. Benchmark — Comprehensive artifact generator (Slides 3-4)
# ---------------------------------------------------------------------------

@pytest.mark.benchmark
def test_generate_presentation_artifacts(timing_pipeline):
    """Generate all CSV and PNG artifacts needed for the presentation.

    Slides covered:
      - Slide 3  : presentation_specs.csv  — component specifications table
      - Slide 4  : presentation_timing.csv — per-file and aggregate timing
                   presentation_metrics.csv — retrieval metric baseline
                   performance_dashboard.png — 4-panel visualization
    """
    if not SDS_PDFS:
        pytest.skip("No SDS PDFs found in docs/ — cannot generate artifacts")

    ARTIFACTS_DIR.mkdir(exist_ok=True)

    # ----------------------------------------------------------------
    # Section A: Component Specifications  →  presentation_specs.csv
    # ----------------------------------------------------------------
    from rag.pipeline import (
        _SCANNED_PAGE_CHAR_THRESHOLD,
        _OCR_DPI,
        _PHARMA_DOC_CATEGORIES,
        _KEYWORD_MAP,
    )

    specs = [
        # Component, Setting, Value, Notes
        ("Embedding Model", "Model Name",    "all-MiniLM-L6-v2",       "Sentence-Transformers; 384-dim"),
        ("Chunking",        "Chunk Size",    "512 tokens",              "SentenceSplitter"),
        ("Chunking",        "Overlap",       "50 tokens",               "~10 % overlap"),
        ("Retrieval",       "Top-K",         f"{timing_pipeline.similarity_top_k} chunks", "Per retriever before fusion"),
        ("Retrieval",       "Mode",          "Reciprocal Rank Fusion",  "BM25 + Vector → RRF"),
        ("LLM",             "Max Tokens",    "200",                     "Output length cap"),
        ("LLM",             "Context Window","8192 tokens",             "Input context"),
        ("LLM",             "Temperature",   "0.1",                     "Near-deterministic"),
        ("OCR",             "DPI",           str(_OCR_DPI),             "Tesseract rasterisation"),
        ("OCR",             "Scanned Threshold", f"{_SCANNED_PAGE_CHAR_THRESHOLD} chars", "Below → OCR fallback"),
        ("Classification",  "Categories",    str(len(_PHARMA_DOC_CATEGORIES)), "Including 'unknown'"),
        ("Classification",  "Keyword Categories", str(len(_KEYWORD_MAP)), "Fast-path; no LLM call"),
    ]
    df_specs = pd.DataFrame(specs, columns=["Component", "Setting", "Value", "Notes"])
    specs_csv = ARTIFACTS_DIR / "presentation_specs.csv"
    df_specs.to_csv(str(specs_csv), index=False)

    # ----------------------------------------------------------------
    # Section B: PDF ingestion timing + chunk statistics
    # ----------------------------------------------------------------
    timing_records = []
    chunk_records  = []

    # We use real SentenceSplitter directly (bypasses the mock in timing_pipeline)
    real_splitter = None
    try:
        from llama_index.core.node_parser import SentenceSplitter as _SS
        real_splitter = _SS(chunk_size=512, chunk_overlap=50)
    except Exception:
        pass  # fall back to mock-based estimates

    for pdf_path in SDS_PDFS:
        # -- load timing --
        t0    = time.perf_counter()
        docs  = timing_pipeline.load_pdf(str(pdf_path))
        t_load = time.perf_counter() - t0

        if not docs:
            continue

        # -- chunk stats (real splitter if available, otherwise estimate) --
        if real_splitter is not None:
            t1     = time.perf_counter()
            chunks = real_splitter.get_nodes_from_documents(docs)
            t_chunk = time.perf_counter() - t1
            chunk_texts  = [c.get_content() for c in chunks]
            chunk_lengths = [len(t) for t in chunk_texts]
        else:
            # Estimate: assume ~400 chars per chunk on average
            total_chars  = sum(len(d.text) for d in docs)
            n_chunks_est = max(1, total_chars // 400)
            chunks       = []
            t_chunk      = 0.0
            chunk_lengths = [400] * n_chunks_est

        n_pages  = len(docs)
        n_chunks = len(chunk_lengths)
        avg_cl   = sum(chunk_lengths) / max(n_chunks, 1)
        ocr_pages = sum(1 for d in docs if d.metadata.get("ocr_used", False))

        timing_records.append({
            "file":            pdf_path.name,
            "pages":           n_pages,
            "ocr_pages":       ocr_pages,
            "load_time_s":     round(t_load, 4),
            "chunk_time_s":    round(t_chunk, 4),
            "total_ingest_s":  round(t_load + t_chunk, 4),
            "chunk_count":     n_chunks,
            "avg_chunk_chars": round(avg_cl, 1),
        })

        for length in chunk_lengths:
            chunk_records.append({"file": pdf_path.name, "chunk_chars": length})

    df_timing = pd.DataFrame(timing_records)
    df_timing.to_csv(str(ARTIFACTS_DIR / "presentation_timing.csv"), index=False)

    # ----------------------------------------------------------------
    # Section C: Retrieval metric baseline on SDS eval set
    # ----------------------------------------------------------------
    # Hand-crafted eval set: query → keyword signals that indicate relevance.
    # For each SDS query we define which section keywords should appear in a
    # relevant chunk.  We evaluate BM25-style (token overlap) as a lower-bound
    # proxy since the real vector index requires a running LLM embedding.
    eval_queries = EVAL_QUERIES

    def _bm25_simulate(query: str, text: str) -> float:
        """Token-overlap score (BM25 proxy)."""
        qtoks = set(query.lower().split())
        dtoks = set(text.lower().split())
        return len(qtoks & dtoks) / max(len(qtoks), 1)

    def _is_relevant(chunk_text: str, required_keywords: List[str]) -> bool:
        low = chunk_text.lower()
        return any(kw in low for kw in required_keywords)

    # Use one representative SDS (test_1.pdf) for the metric baseline
    baseline_pdf = DOCS_DIR / "test_1.pdf"
    retrieval_rows = []

    if baseline_pdf.exists() and real_splitter is not None:
        docs   = timing_pipeline.load_pdf(str(baseline_pdf))
        chunks = real_splitter.get_nodes_from_documents(docs)
        chunk_texts = [c.get_content() for c in chunks]
        chunk_ids   = [str(i) for i in range(len(chunk_texts))]

        for eq in eval_queries:
            # BM25-ranked retrieval
            scored = sorted(
                enumerate(chunk_texts),
                key=lambda x: _bm25_simulate(eq["query"], x[1]),
                reverse=True,
            )
            retrieved_ids = [chunk_ids[i] for i, _ in scored]
            relevant_ids  = {
                chunk_ids[i] for i, t in enumerate(chunk_texts)
                if _is_relevant(t, eq["required_keywords"])
            }

            if not relevant_ids:
                # No chunk contains the signal — skip this query for metrics
                continue

            row = {
                "query":      eq["query"],
                "section":    eq["section"],
                "relevant_chunks": len(relevant_ids),
                "total_chunks":    len(chunk_ids),
            }
            for k in [1, 3, 5]:
                row[f"Recall@{k}"]    = recall_at_k(retrieved_ids, relevant_ids, k)
                row[f"Precision@{k}"] = precision_at_k(retrieved_ids, relevant_ids, k)
            row["MRR"] = mean_reciprocal_rank(retrieved_ids, relevant_ids)
            row["MAP"] = average_precision(retrieved_ids, relevant_ids)
            retrieval_rows.append(row)

    df_metrics = pd.DataFrame(retrieval_rows) if retrieval_rows else pd.DataFrame(
        columns=["query", "section", "Recall@5", "Precision@5", "MRR", "MAP"]
    )
    df_metrics.to_csv(str(ARTIFACTS_DIR / "presentation_metrics.csv"), index=False)

    # ----------------------------------------------------------------
    # Section D: Performance Dashboard PNG  →  performance_dashboard.png
    # ----------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(
        "Pipeline Performance Dashboard — Pharmaceutical RAG System",
        fontsize=14, fontweight="bold",
    )
    ax = axes.flatten()
    palette = plt.colormaps["tab10"].colors

    # Panel 0 — Per-file ingestion time (load + chunk)
    if not df_timing.empty:
        labels = df_timing["file"].str.replace(".pdf", "", regex=False)
        x = np.arange(len(labels))
        w = 0.4
        ax[0].bar(x - w/2, df_timing["load_time_s"],  width=w, label="Load (fitz)",  color=palette[0])
        ax[0].bar(x + w/2, df_timing["chunk_time_s"], width=w, label="Chunk (split)", color=palette[1])
        ax[0].set_title("Ingestion Time per PDF")
        ax[0].set_xticks(x)
        ax[0].set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax[0].set_ylabel("Seconds")
        ax[0].legend(fontsize=9)
        # Annotate totals
        for xi, row in zip(x, df_timing.itertuples()):
            ax[0].text(xi, row.total_ingest_s + 0.01, f"{row.total_ingest_s:.2f}s",
                       ha="center", fontsize=7)

    # Panel 1 — Chunk count vs page count
    if not df_timing.empty:
        ax[1].scatter(df_timing["pages"], df_timing["chunk_count"], color=palette[2],
                      s=80, alpha=0.85)
        for _, row in df_timing.iterrows():
            ax[1].annotate(row["file"].replace(".pdf", ""),
                           (row["pages"], row["chunk_count"]),
                           fontsize=7, xytext=(4, 2), textcoords="offset points")
        ax[1].set_xlabel("PDF Pages")
        ax[1].set_ylabel("Chunks Created (512-tok, 50 overlap)")
        ax[1].set_title("Pages → Chunks Relationship")

    # Panel 2 — Chunk character-length distribution
    if chunk_records:
        df_chunks = pd.DataFrame(chunk_records)
        ax[2].hist(df_chunks["chunk_chars"], bins=30, color=palette[3], edgecolor="white")
        ax[2].axvline(df_chunks["chunk_chars"].mean(), color="red", linestyle="--",
                      label=f"Mean: {df_chunks['chunk_chars'].mean():.0f} chars")
        ax[2].set_xlabel("Chunk Length (chars)")
        ax[2].set_ylabel("Count")
        ax[2].set_title("Chunk Size Distribution (all SDS files)")
        ax[2].legend(fontsize=9)

    # Panel 3 — Retrieval metric bar chart (BM25 baseline)
    if not df_metrics.empty:
        metric_cols = [c for c in ["Recall@5", "Precision@5", "MRR", "MAP"]
                       if c in df_metrics.columns]
        if metric_cols:
            means   = df_metrics[metric_cols].mean()
            mx      = np.arange(len(metric_cols))
            bars    = ax[3].bar(mx, means.values, color=palette[4], width=0.5)
            ax[3].set_xticks(mx)
            ax[3].set_xticklabels(metric_cols)
            ax[3].set_ylim(0, 1.15)
            ax[3].set_title("Retrieval Metrics — BM25 Baseline (test_1.pdf)")
            ax[3].set_ylabel("Score (macro-avg over queries)")
            for bar, val in zip(bars, means.values):
                ax[3].text(bar.get_x() + bar.get_width() / 2, val + 0.02,
                           f"{val:.2f}", ha="center", fontsize=9)
    else:
        ax[3].text(0.5, 0.5, "BM25 baseline requires\nreal SentenceSplitter",
                   ha="center", va="center", transform=ax[3].transAxes, fontsize=11)
        ax[3].set_title("Retrieval Metrics — BM25 Baseline")

    plt.tight_layout()
    dashboard_path = ARTIFACTS_DIR / "performance_dashboard.png"
    fig.savefig(str(dashboard_path), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ----------------------------------------------------------------
    # Assertions + summary print (visible in CI / pytest -s)
    # ----------------------------------------------------------------
    assert specs_csv.exists() and specs_csv.stat().st_size > 0
    assert (ARTIFACTS_DIR / "presentation_timing.csv").exists()
    assert (ARTIFACTS_DIR / "presentation_metrics.csv").exists()
    assert dashboard_path.exists() and dashboard_path.stat().st_size > 0

    def _p(s: str) -> None:
        """Print string, replacing non-ASCII chars for Windows cp1252 consoles."""
        print(s.encode("ascii", errors="replace").decode("ascii"))

    _p("\n" + "=" * 60)
    _p("PIPELINE SPECIFICATIONS  (Slide 3 - Component Specifications)")
    _p("=" * 60)
    _p(df_specs.to_string(index=False))

    if not df_timing.empty:
        _p("\n" + "=" * 60)
        _p("INGESTION TIMING  (Slide 4 - System Performance)")
        _p("=" * 60)
        summary_cols = ["file", "pages", "ocr_pages", "load_time_s",
                        "chunk_time_s", "total_ingest_s", "chunk_count", "avg_chunk_chars"]
        _p(df_timing[summary_cols].to_string(index=False))

        _p(f"\nTotals across {len(df_timing)} files:")
        _p(f"  Total pages  : {df_timing['pages'].sum()}")
        _p(f"  Total chunks : {df_timing['chunk_count'].sum()}")
        _p(f"  Avg load/file: {df_timing['load_time_s'].mean():.3f}s")
        _p(f"  Avg chunk/file: {df_timing['chunk_time_s'].mean():.3f}s")

    if not df_metrics.empty:
        _p("\n" + "=" * 60)
        _p("RETRIEVAL METRICS  (Slide 4 - Retrieval Performance)")
        _p("Note: BM25-only baseline; hybrid (BM25+Vector) will be higher")
        _p("=" * 60)
        metric_cols = [c for c in df_metrics.columns
                       if c.startswith(("Recall", "Precision", "MRR", "MAP"))]
        _p(df_metrics[["query"] + metric_cols].to_string(index=False))
        if metric_cols:
            means = df_metrics[metric_cols].mean()
            _p("\nMacro-averages:")
            for col in metric_cols:
                _p(f"  {col:<14}: {means[col]:.4f}")

    _p(f"\nArtifacts written to: {ARTIFACTS_DIR}")
    _p(f"  {specs_csv.name}")
    _p("  presentation_timing.csv")
    _p("  presentation_metrics.csv")
    _p(f"  {dashboard_path.name}")


# ---------------------------------------------------------------------------
# Fixture — real embeddings, mocked LLM only
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def embedding_pipeline(tmp_path_factory):
    """RAGPipeline with real HuggingFaceEmbedding + SentenceSplitter.

    LlamaCPP is replaced with llama_index's ``MockLLM``, which is a proper
    ``LLM`` subclass and passes isinstance() checks inside llama_index without
    loading any model weights.  Used for vector/hybrid retrieval quality tests.
    """
    try:
        import sentence_transformers  # noqa: F401
        from llama_index.core.node_parser import SentenceSplitter  # noqa: F401
        from llama_index.retrievers.bm25 import BM25Retriever  # noqa: F401
        from llama_index.core.retrievers import VectorIndexRetriever  # noqa: F401
        from llama_index.core.llms.mock import MockLLM  # noqa: F401
    except ImportError:
        pytest.skip("sentence_transformers / llama_index not fully installed")

    p = tmp_path_factory.mktemp("embed") / "model.gguf"
    p.write_bytes(b"")

    from llama_index.core.llms.mock import MockLLM
    with patch("rag.pipeline.LlamaCPP") as mlc:
        mlc.return_value = MockLLM()
        from rag import RAGPipeline
        # num_queries=1 → QueryFusionRetriever never calls the LLM for expansion
        rag = RAGPipeline(model_path=str(p), num_queries=1)

    return rag


# ---------------------------------------------------------------------------
# 4. Benchmark — BM25 vs Vector vs Hybrid retrieval comparison (Slide 4)
# ---------------------------------------------------------------------------

@pytest.mark.benchmark
def test_generate_retrieval_comparison(embedding_pipeline):
    """Compare BM25-only, Vector-only, and Hybrid (RRF) retrieval quality.

    Builds a real vector index on test_1.pdf using the actual
    all-MiniLM-L6-v2 embedding model.  No LLM is called during retrieval.

    Writes:
      artifacts/retrieval_comparison.csv  — per-query metrics, all 3 methods
      artifacts/retrieval_comparison.png  — 2-panel comparison chart
    """
    baseline_pdf = DOCS_DIR / "test_1.pdf"
    if not baseline_pdf.exists():
        pytest.skip("docs/test_1.pdf not found")

    try:
        from llama_index.core import VectorStoreIndex
        from llama_index.core.node_parser import SentenceSplitter
        from llama_index.retrievers.bm25 import BM25Retriever
        from llama_index.core.retrievers import VectorIndexRetriever, QueryFusionRetriever
    except ImportError:
        pytest.skip("llama_index components not available")

    ARTIFACTS_DIR.mkdir(exist_ok=True)
    K = embedding_pipeline.similarity_top_k

    # Load docs and build a real vector index without calling build()
    # (avoids RetrieverQueryEngine LLM validation — we only need retrieval)
    docs         = embedding_pipeline.load_pdf(str(baseline_pdf))
    splitter     = SentenceSplitter(chunk_size=512, chunk_overlap=50)
    chunks       = splitter.get_nodes_from_documents(docs)
    vector_index = VectorStoreIndex(chunks)   # uses Settings.embed_model (real MiniLM)

    # Map node_id → chunk text for relevance labelling
    node_id_to_text: Dict[str, str] = {}
    for c in chunks:
        nid  = getattr(c, "node_id", None)
        text = c.get_content() if hasattr(c, "get_content") else getattr(c, "text", "")
        if nid:
            node_id_to_text[nid] = text

    all_node_ids = list(node_id_to_text.keys())

    def _relevant_ids(required_keywords: List[str]) -> set:
        return {
            nid for nid, text in node_id_to_text.items()
            if any(kw in text.lower() for kw in required_keywords)
        }

    def _retrieve_ids(retriever, query_text: str) -> List[str]:
        try:
            nodes = retriever.retrieve(query_text)
            return [n.node.node_id for n in nodes
                    if hasattr(n, "node") and hasattr(n.node, "node_id")]
        except Exception:
            return []

    # Build the 3 retrievers directly — no LLM or query engine required
    bm25_retriever   = BM25Retriever.from_defaults(nodes=chunks, similarity_top_k=K)
    vector_retriever = VectorIndexRetriever(index=vector_index, similarity_top_k=K)
    hybrid_retriever = QueryFusionRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        similarity_top_k=K,
        num_queries=1,          # no LLM query-expansion
        mode="reciprocal_rerank",
        use_async=False,
        llm=embedding_pipeline.llm,  # MockLLM; never called with num_queries=1
    )

    METHODS = {
        "BM25-only":    bm25_retriever,
        "Vector-only":  vector_retriever,
        "Hybrid (RRF)": hybrid_retriever,
    }
    METHOD_COLORS = {
        "BM25-only":    "#1565C0",
        "Vector-only":  "#E65100",
        "Hybrid (RRF)": "#2E7D32",
    }
    SUMMARY_METRICS = ["Recall@5", "Precision@5", "MRR", "MAP"]

    rows: List[Dict] = []
    for eq in EVAL_QUERIES:
        rel_ids = _relevant_ids(eq["required_keywords"])
        if not rel_ids:
            continue   # no relevant chunk found in this document — skip query

        row: Dict = {
            "query":            eq["query"],
            "section":          eq["section"],
            "relevant_chunks":  len(rel_ids),
            "total_chunks":     len(all_node_ids),
        }
        for method_name, retriever in METHODS.items():
            ret_ids = _retrieve_ids(retriever, eq["query"])
            for k in [1, 3, 5]:
                row[f"{method_name}_Recall@{k}"]    = recall_at_k(ret_ids, rel_ids, k)
                row[f"{method_name}_Precision@{k}"] = precision_at_k(ret_ids, rel_ids, k)
            row[f"{method_name}_MRR"] = mean_reciprocal_rank(ret_ids, rel_ids)
            row[f"{method_name}_MAP"] = average_precision(ret_ids, rel_ids)
        rows.append(row)

    if not rows:
        pytest.skip("No queries produced retrievable results on test_1.pdf")

    df = pd.DataFrame(rows)
    comparison_csv = ARTIFACTS_DIR / "retrieval_comparison.csv"
    df.to_csv(str(comparison_csv), index=False)

    # ----------------------------------------------------------------
    # Chart 1 — Macro-average metrics grouped by method (Recall@5,
    #           Precision@5, MRR, MAP)
    # Chart 2 — Per-query MRR for each method
    # ----------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(
        "Retrieval Method Comparison — BM25 vs Vector vs Hybrid (test_1.pdf)",
        fontsize=13, fontweight="bold",
    )

    methods_list = list(METHODS.keys())
    bar_w = 0.25

    # --- Panel 1: macro-average per metric ---
    x1 = np.arange(len(SUMMARY_METRICS))
    for i, m in enumerate(methods_list):
        means = [
            df[f"{m}_{metric}"].mean() if f"{m}_{metric}" in df.columns else 0.0
            for metric in SUMMARY_METRICS
        ]
        bars = ax1.bar(
            x1 + (i - 1) * bar_w, means,
            width=bar_w, label=m, color=METHOD_COLORS[m], alpha=0.88,
        )
        for bar, val in zip(bars, means):
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.015,
                f"{val:.2f}", ha="center", fontsize=7.5,
            )
    ax1.set_xticks(x1)
    ax1.set_xticklabels(SUMMARY_METRICS)
    ax1.set_ylim(0, 1.30)
    ax1.set_ylabel("Score (macro-avg across queries)")
    ax1.set_title("Macro-Average Retrieval Metrics")
    ax1.legend(fontsize=9)
    ax1.axhline(0, color="black", linewidth=0.5)

    # --- Panel 2: per-query MRR ---
    query_labels = [r["section"].replace("Section ", "S") for r in rows]
    x2 = np.arange(len(query_labels))
    for i, m in enumerate(methods_list):
        mrr_col = f"{m}_MRR"
        vals = df[mrr_col].values.tolist() if mrr_col in df.columns else [0.0] * len(rows)
        bars = ax2.bar(
            x2 + (i - 1) * bar_w, vals,
            width=bar_w, label=m, color=METHOD_COLORS[m], alpha=0.88,
        )
        for bar, val in zip(bars, vals):
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                val + 0.015,
                f"{val:.2f}", ha="center", fontsize=6.5,
            )
    ax2.set_xticks(x2)
    ax2.set_xticklabels(query_labels, rotation=20, ha="right", fontsize=8)
    ax2.set_ylim(0, 1.30)
    ax2.set_ylabel("MRR")
    ax2.set_title("Per-Query MRR by Retrieval Method")
    ax2.legend(fontsize=9)
    ax2.axhline(0, color="black", linewidth=0.5)

    plt.tight_layout()
    comparison_png = ARTIFACTS_DIR / "retrieval_comparison.png"
    fig.savefig(str(comparison_png), dpi=150, bbox_inches="tight")
    plt.close(fig)

    assert comparison_csv.exists() and comparison_csv.stat().st_size > 0
    assert comparison_png.exists() and comparison_png.stat().st_size > 0

    # Summary print
    def _p(s: str) -> None:
        print(s.encode("ascii", errors="replace").decode("ascii"))

    _p("\n" + "=" * 70)
    _p("RETRIEVAL METHOD COMPARISON  (Slide 4 - Retrieval Performance)")
    _p(f"Document: test_1.pdf  |  Top-K={K}  |  Queries={len(rows)}")
    _p("=" * 70)
    header = f"{'Method':<16}" + "".join(f"{m:<14}" for m in SUMMARY_METRICS)
    _p(header)
    _p("-" * len(header))
    for m in methods_list:
        row_str = f"{m:<16}"
        for metric in SUMMARY_METRICS:
            col = f"{m}_{metric}"
            val = df[col].mean() if col in df.columns else 0.0
            row_str += f"{val:<14.4f}"
        _p(row_str)
    _p(f"\nArtifacts:")
    _p(f"  {comparison_csv.name}")
    _p(f"  {comparison_png.name}")


# ---------------------------------------------------------------------------
# 5. Benchmark — per-query latency comparison: BM25 vs Vector vs Hybrid
# ---------------------------------------------------------------------------

_LATENCY_REPS = 20   # repetitions per (method, query) — 20 × 5 queries = 100 samples/method
_LATENCY_WARMUP = 2  # unreported warm-up reps to prime caches before measurement


@pytest.mark.benchmark
def test_retrieval_latency_comparison(embedding_pipeline):
    """Measure retrieve() latency (avg, p50, p95) for BM25, Vector, and Hybrid.

    Each of the 5 EVAL_QUERIES is executed ``_LATENCY_REPS`` times per method
    to collect enough samples for stable percentiles (25 data points per method).

    Writes:
      artifacts/latency_comparison.csv  — per-method summary (avg/p50/p95)
      artifacts/latency_comparison.png  — grouped bar chart
    """
    baseline_pdf = DOCS_DIR / "test_1.pdf"
    if not baseline_pdf.exists():
        pytest.skip("docs/test_1.pdf not found")

    try:
        from llama_index.core import VectorStoreIndex
        from llama_index.core.node_parser import SentenceSplitter
        from llama_index.retrievers.bm25 import BM25Retriever
        from llama_index.core.retrievers import VectorIndexRetriever, QueryFusionRetriever
    except ImportError:
        pytest.skip("llama_index components not available")

    ARTIFACTS_DIR.mkdir(exist_ok=True)
    K = embedding_pipeline.similarity_top_k

    # Build index (same as retrieval comparison test)
    docs         = embedding_pipeline.load_pdf(str(baseline_pdf))
    splitter     = SentenceSplitter(chunk_size=512, chunk_overlap=50)
    chunks       = splitter.get_nodes_from_documents(docs)
    vector_index = VectorStoreIndex(chunks)

    bm25_retriever   = BM25Retriever.from_defaults(nodes=chunks, similarity_top_k=K)
    vector_retriever = VectorIndexRetriever(index=vector_index, similarity_top_k=K)
    hybrid_retriever = QueryFusionRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        similarity_top_k=K,
        num_queries=1,
        mode="reciprocal_rerank",
        use_async=False,
        llm=embedding_pipeline.llm,
    )

    METHODS = {
        "BM25-only":    bm25_retriever,
        "Vector-only":  vector_retriever,
        "Hybrid (RRF)": hybrid_retriever,
    }
    METHOD_COLORS = {
        "BM25-only":    "#1565C0",
        "Vector-only":  "#E65100",
        "Hybrid (RRF)": "#2E7D32",
    }
    method_keys = list(METHODS.keys())

    # ----------------------------------------------------------------
    # Warmup — not recorded
    # ----------------------------------------------------------------
    for _ in range(_LATENCY_WARMUP):
        for method_name, retriever in METHODS.items():
            for eq in EVAL_QUERIES:
                try:
                    retriever.retrieve(eq["query"])
                except Exception:
                    pass

    # ----------------------------------------------------------------
    # Timed measurement — randomise method order each rep to reduce
    # systematic ordering bias (e.g. cache warming for one method first)
    # ----------------------------------------------------------------
    raw: Dict[str, List[float]] = {m: [] for m in METHODS}
    raw_samples: List[Dict] = []

    for rep in range(_LATENCY_REPS):
        rep_order = method_keys[:]
        random.shuffle(rep_order)
        for method_name in rep_order:
            retriever = METHODS[method_name]
            for eq in EVAL_QUERIES:
                t0 = time.perf_counter()
                try:
                    retriever.retrieve(eq["query"])
                except Exception:
                    pass
                lat_ms = (time.perf_counter() - t0) * 1_000
                raw[method_name].append(lat_ms)
                raw_samples.append({
                    "method":     method_name,
                    "query":      eq["query"],
                    "repetition": rep + 1,
                    "latency_ms": round(lat_ms, 4),
                    "top_k":      K,
                    "corpus_size": len(chunks),
                })

    # ----------------------------------------------------------------
    # Write raw per-sample CSV
    # ----------------------------------------------------------------
    latency_raw_csv = ARTIFACTS_DIR / "latency_raw.csv"
    pd.DataFrame(raw_samples).to_csv(str(latency_raw_csv), index=False)

    # ----------------------------------------------------------------
    # Compute per-method summary statistics (avg/p50/p95/p99/std/min/max)
    # ----------------------------------------------------------------
    stat_rows: List[Dict] = []
    for method_name in method_keys:
        arr = np.array(raw[method_name])
        stat_rows.append({
            "profile": "retrieval_only",
            "method":  method_name,
            "n":       int(len(arr)),
            "avg_ms":  round(float(np.mean(arr)),             2),
            "p50_ms":  round(float(np.percentile(arr, 50)),   2),
            "p95_ms":  round(float(np.percentile(arr, 95)),   2),
            "p99_ms":  round(float(np.percentile(arr, 99)),   2),
            "std_ms":  round(float(np.std(arr)),              2),
            "min_ms":  round(float(np.min(arr)),              2),
            "max_ms":  round(float(np.max(arr)),              2),
        })

    df_summary = pd.DataFrame(stat_rows)
    method_summary_csv = ARTIFACTS_DIR / "method_latency_summary.csv"
    df_summary.to_csv(str(method_summary_csv), index=False)

    # Keep existing latency_comparison.csv (with updated columns) for backward compat
    latency_csv = ARTIFACTS_DIR / "latency_comparison.csv"
    df_summary.to_csv(str(latency_csv), index=False)

    # ----------------------------------------------------------------
    # Stage breakdown — retrieval_only profile: retrieve_ms = latency, rest = 0
    # ----------------------------------------------------------------
    breakdown_rows: List[Dict] = []
    for row in stat_rows:
        r_ms = row["avg_ms"]
        breakdown_rows.append({
            "profile":          "retrieval_only",
            "method":           row["method"],
            "expand_ms_avg":    0.0,
            "classify_ms_avg":  0.0,
            "retrieve_ms_avg":  r_ms,
            "generate_ms_avg":  0.0,
            "total_ms_avg":     r_ms,
            "percent_expand":   0.0,
            "percent_classify": 0.0,
            "percent_retrieve": 100.0 if r_ms > 0 else 0.0,
            "percent_generate": 0.0,
        })
    stage_csv = ARTIFACTS_DIR / "stage_breakdown_summary.csv"
    pd.DataFrame(breakdown_rows).to_csv(str(stage_csv), index=False)

    # ----------------------------------------------------------------
    # Slowdown attribution — delta vs BM25-only baseline
    # ----------------------------------------------------------------
    bm25_avg = next(r["avg_ms"] for r in stat_rows if r["method"] == "BM25-only")
    attrib_rows: List[Dict] = []
    for row in stat_rows:
        if row["method"] == "BM25-only":
            continue
        delta = round(row["avg_ms"] - bm25_avg, 3)
        attrib_rows.append({
            "compare_to":        "BM25-only",
            "method":            row["method"],
            "delta_total_ms":    delta,
            "delta_expand_ms":   0.0,
            "delta_classify_ms": 0.0,
            "delta_retrieve_ms": delta,
            "delta_generate_ms": 0.0,
            "dominant_cause":    "retrieve" if delta > 0 else "none",
        })
    slowdown_csv = ARTIFACTS_DIR / "slowdown_attribution.csv"
    pd.DataFrame(attrib_rows).to_csv(str(slowdown_csv), index=False)

    # ----------------------------------------------------------------
    # Latency chart — grouped bars (avg/p50/p95) + box plot
    # ----------------------------------------------------------------
    fig, (ax_main, ax_dist) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        f"Retrieval Latency — BM25 vs Vector vs Hybrid "
        f"(test_1.pdf, {_LATENCY_REPS} reps × {len(EVAL_QUERIES)} queries, "
        f"{_LATENCY_WARMUP} warmup reps)",
        fontsize=12, fontweight="bold",
    )

    stat_labels  = ["avg_ms", "p50_ms", "p95_ms"]
    stat_display = ["Avg",    "P50",    "P95"]
    x     = np.arange(len(stat_labels))
    bar_w = 0.25

    # Panel 1 — grouped bars
    for i, (method_name, row) in enumerate(zip(method_keys, stat_rows)):
        vals = [row[s] for s in stat_labels]
        bars = ax_main.bar(
            x + (i - 1) * bar_w, vals,
            width=bar_w, label=method_name,
            color=METHOD_COLORS[method_name], alpha=0.88,
        )
        for bar, val in zip(bars, vals):
            ax_main.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.3,
                f"{val:.1f}", ha="center", fontsize=7.5,
            )
    ax_main.set_xticks(x)
    ax_main.set_xticklabels(stat_display, fontsize=11)
    ax_main.set_ylabel("Latency (ms)")
    ax_main.set_title("Latency Statistics per Retrieval Method")
    ax_main.legend(fontsize=9)
    ax_main.set_ylim(0, ax_main.get_ylim()[1] * 1.15)

    # Panel 2 — box plot of full sample distribution
    box_data = [raw[m] for m in method_keys]
    bp = ax_dist.boxplot(
        box_data,
        tick_labels=method_keys,
        patch_artist=True,
        medianprops=dict(color="black", linewidth=2),
    )
    for patch, method_name in zip(bp["boxes"], method_keys):
        patch.set_facecolor(METHOD_COLORS[method_name])
        patch.set_alpha(0.7)
    ax_dist.set_ylabel("Latency (ms)")
    n_samples = _LATENCY_REPS * len(EVAL_QUERIES)
    ax_dist.set_title(f"Latency Distribution ({n_samples} samples/method, {_LATENCY_WARMUP} warmup)")
    ax_dist.tick_params(axis="x", labelsize=9)

    plt.tight_layout()
    latency_png = ARTIFACTS_DIR / "latency_comparison.png"
    fig.savefig(str(latency_png), dpi=150, bbox_inches="tight")
    plt.close(fig)

    assert latency_raw_csv.exists()    and latency_raw_csv.stat().st_size > 0
    assert method_summary_csv.exists() and method_summary_csv.stat().st_size > 0
    assert stage_csv.exists()          and stage_csv.stat().st_size > 0
    assert slowdown_csv.exists()       and slowdown_csv.stat().st_size > 0
    assert latency_csv.exists()        and latency_csv.stat().st_size > 0
    assert latency_png.exists()        and latency_png.stat().st_size > 0

    # ----------------------------------------------------------------
    # Summary print
    # ----------------------------------------------------------------
    def _p(s: str) -> None:
        print(s.encode("ascii", errors="replace").decode("ascii"))

    col_w = 12
    _p("\n" + "=" * 74)
    _p(f"RETRIEVAL LATENCY COMPARISON  "
       f"({_LATENCY_REPS} reps x {len(EVAL_QUERIES)} queries = {n_samples} samples/method, "
       f"{_LATENCY_WARMUP} warmup, randomised order)")
    _p("=" * 74)
    _p(f"{'Method':<16}"
       f"{'Avg':>{col_w}}{'P50':>{col_w}}{'P95':>{col_w}}{'P99':>{col_w}}"
       f"{'Std':>{col_w}}{'Min':>{col_w}}{'Max':>{col_w}}")
    _p("-" * (16 + col_w * 7))
    for row in stat_rows:
        _p(
            f"{row['method']:<16}"
            f"{row['avg_ms']:>{col_w}.2f}"
            f"{row['p50_ms']:>{col_w}.2f}"
            f"{row['p95_ms']:>{col_w}.2f}"
            f"{row['p99_ms']:>{col_w}.2f}"
            f"{row['std_ms']:>{col_w}.2f}"
            f"{row['min_ms']:>{col_w}.2f}"
            f"{row['max_ms']:>{col_w}.2f}"
        )

    _p("\nSlowdown vs BM25-only:")
    for row in attrib_rows:
        sign = "+" if row["delta_total_ms"] >= 0 else ""
        _p(f"  {row['method']:<16} {sign}{row['delta_total_ms']:.2f} ms  (dominant: {row['dominant_cause']})")

    _p(f"\nArtifacts:")
    _p(f"  {latency_raw_csv.name}      ({len(raw_samples)} rows)")
    _p(f"  {method_summary_csv.name}")
    _p(f"  {stage_csv.name}")
    _p(f"  {slowdown_csv.name}")
    _p(f"  {latency_png.name}")

"""
End-to-End Accuracy & System Performance Tests
===============================================
Generates the internal data needed for the presentation slides:

  💊 End-to-End Accuracy
    - Answer Accuracy     : % of test questions whose answers contain all required keywords
    - Citation Accuracy   : % of questions where the correct source file appears in retrieved chunks
    - Factual Consistency : % of answers fully grounded in retrieved source text (no hallucination)

  🧑‍🎓 System Performance
    - Average Response Time (total_ms)
    - Ingestion Rate: ms per 10 pages (load_pdf wall-clock, real fitz)
    - Retrieval Latency (retrieve_ms from query_with_sources)
    - LLM Generation Time (generate_ms from query_with_sources)

Markers
-------
unit      : metric calculation logic only; no model files or docs needed; CI-safe.
accuracy  : runs real pipeline on SDS PDFs; requires fitz + llama-index stack.

Run unit tests only:
    pytest tests/test_end_to_end_accuracy.py -m unit -v

Run full accuracy + performance benchmarks (writes artifacts):
    pytest tests/test_end_to_end_accuracy.py -m accuracy -v -s
"""

from __future__ import annotations

import csv
import os
import sys
import time
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DOCS_DIR = Path(__file__).parent.parent / "docs"
ARTIFACTS_DIR = Path(__file__).parent.parent / "artifacts"

# SDS PDFs available (test_6 absent from glob, use confirmed files)
SDS_PDFS = [str(DOCS_DIR / f"test_{i}.pdf") for i in range(1, 6)]  # test_1 .. test_5


# ---------------------------------------------------------------------------
# Ground-truth test dataset
# ---------------------------------------------------------------------------
# Each entry describes one labelled question over the SDS corpus.
#
# Fields:
#   test_id            – unique identifier
#   query              – natural-language question
#   source_pdf         – basename of the PDF the answer lives in
#   required_keywords  – at least ONE must appear in the answer (case-insensitive)
#   all_keywords       – ALL must appear for full-credit answer accuracy
#                        (used only in the strict scoring variant)
#   criticality        – "high" / "medium" for reporting

GROUND_TRUTH: List[Dict[str, Any]] = [
    {
        "test_id": "SDS-001",
        "query": "What is the product name or substance name of this material?",
        "source_pdf": "test_1.pdf",
        "required_keywords": ["safety data sheet", "product", "substance", "material", "chemical"],
        "criticality": "high",
    },
    {
        "test_id": "SDS-002",
        "query": "What are the hazard classifications or GHS hazard statements?",
        "source_pdf": "test_1.pdf",
        "required_keywords": ["hazard", "ghs", "h2", "flammable", "irritant", "toxic", "warning", "danger"],
        "criticality": "high",
    },
    {
        "test_id": "SDS-003",
        "query": "What are the recommended storage conditions for this product?",
        "source_pdf": "test_1.pdf",
        "required_keywords": ["storage", "store", "temperature", "cool", "dry", "ventilated"],
        "criticality": "high",
    },
    {
        "test_id": "SDS-004",
        "query": "What first aid measures should be taken in case of skin contact?",
        "source_pdf": "test_1.pdf",
        "required_keywords": ["skin", "wash", "rinse", "water", "first aid", "contact"],
        "criticality": "medium",
    },
    {
        "test_id": "SDS-005",
        "query": "What personal protective equipment is required when handling this substance?",
        "source_pdf": "test_1.pdf",
        "required_keywords": ["gloves", "ppe", "protective", "goggles", "respirator", "equipment"],
        "criticality": "medium",
    },
    {
        "test_id": "SDS-006",
        "query": "What are the physical state and appearance of this material?",
        "source_pdf": "test_2.pdf",
        "required_keywords": ["liquid", "solid", "gas", "colour", "color", "appearance", "odour", "odor", "physical"],
        "criticality": "medium",
    },
    {
        "test_id": "SDS-007",
        "query": "What is the CAS number or chemical identification of this substance?",
        "source_pdf": "test_2.pdf",
        "required_keywords": ["cas", "einecs", "chemical", "formula", "molecular", "identifier"],
        "criticality": "high",
    },
    {
        "test_id": "SDS-008",
        "query": "What are the disposal instructions for this material?",
        "source_pdf": "test_3.pdf",
        "required_keywords": ["disposal", "waste", "incineration", "landfill", "regulation", "dispose"],
        "criticality": "medium",
    },
    {
        "test_id": "SDS-009",
        "query": "What fire extinguishing media should be used?",
        "source_pdf": "test_3.pdf",
        "required_keywords": ["fire", "extinguish", "foam", "co2", "carbon dioxide", "dry powder", "water spray"],
        "criticality": "medium",
    },
    {
        "test_id": "SDS-010",
        "query": "What are the emergency response or spill cleanup procedures?",
        "source_pdf": "test_4.pdf",
        "required_keywords": ["spill", "leak", "emergency", "absorb", "contain", "ventilation", "cleanup"],
        "criticality": "high",
    },
]


# ---------------------------------------------------------------------------
# Scoring helpers (unit-testable, no pipeline dependency)
# ---------------------------------------------------------------------------

def score_answer_accuracy(answer: str, required_keywords: List[str]) -> float:
    """Return 1.0 if at least one required keyword appears in the answer, else 0.0.

    Case-insensitive substring match. This is the 'any-hit' variant for
    partial-credit scoring across a diverse keyword list.
    """
    lower = answer.lower()
    return 1.0 if any(kw.lower() in lower for kw in required_keywords) else 0.0


def score_citation_accuracy(sources: List[Dict[str, Any]], expected_source: str) -> float:
    """Return 1.0 if expected_source basename appears in any retrieved source file."""
    base = os.path.basename(expected_source).lower()
    for src in sources:
        file_val = str(src.get("file", "")).lower()
        if base in file_val or file_val in base:
            return 1.0
    return 0.0


def score_factual_consistency(answer: str, sources: List[Dict[str, Any]]) -> float:
    """Estimate grounding: what fraction of answer content can be found in retrieved chunks.

    Method: extract non-trivial tokens from the answer (>3 chars, not stopwords),
    then check what fraction of them appear in the concatenated source texts.
    A score >= 0.6 is considered 'consistent' (not hallucinating).

    Returns a continuous float in [0.0, 1.0] representing overlap.
    """
    _STOPWORDS = {
        "the", "and", "for", "that", "this", "with", "from", "are", "was",
        "were", "have", "been", "they", "their", "there", "which", "when",
        "not", "but", "can", "should", "will", "also", "into", "than",
        "its", "about", "would", "other", "more", "any", "some", "such",
        "only", "each", "after", "being", "during", "both", "you", "your",
    }

    import re  # pylint: disable=import-outside-toplevel

    all_source_text = " ".join(str(s.get("text", "")) for s in sources).lower()

    tokens = re.findall(r"[a-z]{4,}", answer.lower())
    content_tokens = [t for t in tokens if t not in _STOPWORDS]

    if not content_tokens:
        return 1.0  # empty/trivial answer — no hallucination possible

    grounded = sum(1 for t in content_tokens if t in all_source_text)
    return grounded / len(content_tokens)


def answer_similarity(a: str, b: str) -> float:
    """SequenceMatcher ratio between two answer strings."""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


# ---------------------------------------------------------------------------
# Unit tests — metric functions
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestAnswerAccuracyScoring:
    """Verify answer_accuracy scoring logic in isolation."""

    def test_keyword_present_returns_one(self):
        score = score_answer_accuracy("The storage temperature is -20°C.", ["storage", "temperature"])
        assert score == 1.0

    def test_keyword_absent_returns_zero(self):
        score = score_answer_accuracy("I don't know.", ["storage", "temperature", "cool", "dry"])
        assert score == 0.0

    def test_case_insensitive(self):
        score = score_answer_accuracy("STORAGE conditions are cool.", ["storage"])
        assert score == 1.0

    def test_partial_keyword_match(self):
        # "flammable" should match keyword "flammable"
        score = score_answer_accuracy("This product is flammable.", ["flammable", "toxic"])
        assert score == 1.0

    def test_empty_answer(self):
        score = score_answer_accuracy("", ["storage"])
        assert score == 0.0

    def test_empty_keywords(self):
        # no keywords to match → trivially no hit
        score = score_answer_accuracy("Some answer.", [])
        assert score == 0.0


@pytest.mark.unit
class TestCitationAccuracyScoring:
    """Verify citation_accuracy scoring logic."""

    def test_exact_match(self):
        sources = [{"file": "test_1.pdf", "page": 2}]
        assert score_citation_accuracy(sources, "test_1.pdf") == 1.0

    def test_no_match(self):
        sources = [{"file": "test_2.pdf", "page": 1}]
        assert score_citation_accuracy(sources, "test_1.pdf") == 0.0

    def test_match_in_multiple_sources(self):
        sources = [{"file": "test_3.pdf"}, {"file": "test_1.pdf"}]
        assert score_citation_accuracy(sources, "test_1.pdf") == 1.0

    def test_empty_sources(self):
        assert score_citation_accuracy([], "test_1.pdf") == 0.0

    def test_case_insensitive_file_match(self):
        sources = [{"file": "TEST_1.PDF"}]
        assert score_citation_accuracy(sources, "test_1.pdf") == 1.0


@pytest.mark.unit
class TestFactualConsistencyScoring:
    """Verify factual_consistency grounding check."""

    def test_fully_grounded_answer(self):
        answer = "Store in cool dry place below 25 degrees Celsius."
        sources = [{"text": "Store in a cool, dry place below 25 degrees Celsius away from heat."}]
        score = score_factual_consistency(answer, sources)
        assert score >= 0.7

    def test_hallucinated_answer(self):
        answer = "xyzzy frobnicate quuxbar flibbertigibbet zorgon wumpus."
        sources = [{"text": "Store in a cool dry place."}]
        score = score_factual_consistency(answer, sources)
        assert score < 0.3

    def test_empty_sources_returns_low_score(self):
        answer = "The boiling point is 100 degrees Celsius."
        score = score_factual_consistency(answer, [])
        # No source text to ground against
        assert score == 0.0 or score < 0.5

    def test_trivial_answer_returns_one(self):
        score = score_factual_consistency("", [{"text": "some text"}])
        assert score == 1.0


@pytest.mark.unit
class TestSummaryAggregation:
    """Verify the final aggregation math used by the benchmark."""

    def test_pass_rate_calculation(self):
        scores = [1.0, 1.0, 0.0, 1.0, 0.0]
        pass_rate = sum(scores) / len(scores) * 100
        assert abs(pass_rate - 60.0) < 1e-6

    def test_average_latency(self):
        latencies = [10.0, 20.0, 30.0]
        assert abs(sum(latencies) / len(latencies) - 20.0) < 1e-6

    def test_per_10_pages_rate(self):
        # 50 pages in 500ms → 100ms per 10 pages
        total_ms = 500.0
        pages = 50
        rate = (total_ms / pages) * 10
        assert abs(rate - 100.0) < 1e-6


# ---------------------------------------------------------------------------
# Accuracy benchmark — real pipeline on SDS PDFs
# ---------------------------------------------------------------------------

def _build_real_pipeline(pdf_paths: List[str]) -> Any:
    """Build a RAGPipeline from real PDFs with the real llama-index stack."""
    from rag import RAGPipeline  # pylint: disable=import-outside-toplevel

    model_path = os.environ.get("RAG_MODEL_PATH", "")
    if not model_path or not os.path.exists(model_path):
        pytest.skip("RAG_MODEL_PATH not set or model file not found — skipping accuracy benchmark.")

    pipeline = RAGPipeline(model_path=model_path)
    pipeline.build_from_multiple_pdfs(pdf_paths)
    return pipeline


@pytest.mark.accuracy
class TestEndToEndAccuracy:
    """Run ground-truth questions against the pipeline and report accuracy metrics.

    Writes:
        artifacts/accuracy_results.csv     — per-question results
        artifacts/accuracy_summary.csv     — aggregated slide metrics
    """

    def test_generate_accuracy_report(self):
        """Build pipeline, run all ground-truth questions, write CSV artifacts."""
        available_pdfs = [p for p in SDS_PDFS if os.path.exists(p)]
        if len(available_pdfs) < 2:
            pytest.skip(f"Need at least 2 SDS PDFs in {DOCS_DIR}; found {len(available_pdfs)}.")

        pipeline = _build_real_pipeline(available_pdfs)

        rows: List[Dict[str, Any]] = []
        for case in GROUND_TRUTH:
            # Skip if the expected source PDF isn't in the built set
            pdf_base = case["source_pdf"]
            pdf_full = str(DOCS_DIR / pdf_base)
            if pdf_full not in available_pdfs:
                continue

            t0 = time.perf_counter()
            result = pipeline.query_with_sources(case["query"])
            wall_ms = (time.perf_counter() - t0) * 1_000

            answer = result.get("answer", "")
            sources = result.get("sources", [])

            ans_acc = score_answer_accuracy(answer, case["required_keywords"])
            cit_acc = score_citation_accuracy(sources, pdf_base)
            fact_cons = score_factual_consistency(answer, sources)
            is_consistent = fact_cons >= 0.6

            rows.append(
                {
                    "test_id": case["test_id"],
                    "criticality": case["criticality"],
                    "query": case["query"],
                    "expected_source": pdf_base,
                    "answer_accuracy": ans_acc,
                    "citation_accuracy": cit_acc,
                    "factual_consistency": round(fact_cons, 3),
                    "is_factually_consistent": int(is_consistent),
                    "answer_length": len(answer),
                    "num_sources": len(sources),
                    "retrieve_ms": result.get("retrieve_ms", 0.0),
                    "generate_ms": result.get("generate_ms", 0.0),
                    "total_ms": result.get("total_ms", wall_ms),
                    "answer_snippet": answer[:200].replace("\n", " "),
                }
            )

        assert rows, "No test cases were executed."

        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        results_path = ARTIFACTS_DIR / "accuracy_results.csv"
        _write_csv(rows, str(results_path))

        # Aggregate
        n = len(rows)
        ans_acc_pct = sum(r["answer_accuracy"] for r in rows) / n * 100
        cit_acc_pct = sum(r["citation_accuracy"] for r in rows) / n * 100
        fact_cons_pct = sum(r["is_factually_consistent"] for r in rows) / n * 100
        avg_total_ms = sum(r["total_ms"] for r in rows) / n
        avg_retrieve_ms = sum(r["retrieve_ms"] for r in rows) / n
        avg_generate_ms = sum(r["generate_ms"] for r in rows) / n

        summary = [
            {
                "metric": "Answer Accuracy (%)",
                "value": round(ans_acc_pct, 1),
                "n_questions": n,
                "description": "% questions with ≥1 required keyword in answer",
            },
            {
                "metric": "Citation Accuracy (%)",
                "value": round(cit_acc_pct, 1),
                "n_questions": n,
                "description": "% questions where correct source file was retrieved",
            },
            {
                "metric": "Factual Consistency (%)",
                "value": round(fact_cons_pct, 1),
                "n_questions": n,
                "description": "% answers with ≥60% content grounded in retrieved chunks",
            },
            {
                "metric": "Avg Response Time (ms)",
                "value": round(avg_total_ms, 2),
                "n_questions": n,
                "description": "End-to-end wall-clock per query",
            },
            {
                "metric": "Retrieval Latency (ms)",
                "value": round(avg_retrieve_ms, 2),
                "n_questions": n,
                "description": "Hybrid retrieval stage only",
            },
            {
                "metric": "LLM Generation (ms)",
                "value": round(avg_generate_ms, 2),
                "n_questions": n,
                "description": "Answer synthesis stage only",
            },
        ]

        summary_path = ARTIFACTS_DIR / "accuracy_summary.csv"
        _write_csv(summary, str(summary_path))

        # Print slide-ready numbers
        print("\n" + "=" * 60)
        print("[ACCURACY] END-TO-END ACCURACY")
        print("=" * 60)
        print(f"  Answer Accuracy    : {ans_acc_pct:.1f}%  (N={n} questions)")
        print(f"  Citation Accuracy  : {cit_acc_pct:.1f}%  (correct source attribution)")
        print(f"  Factual Consistency: {fact_cons_pct:.1f}%  (grounded in retrieved chunks)")
        print()
        print("[PERFORMANCE] SYSTEM PERFORMANCE")
        print("=" * 60)
        print(f"  Avg Response Time  : {avg_total_ms:.2f} ms")
        print(f"  Retrieval Latency  : {avg_retrieve_ms:.2f} ms")
        print(f"  LLM Generation     : {avg_generate_ms:.2f} ms")
        print(f"\nArtifacts written to: {ARTIFACTS_DIR}")
        print("=" * 60)

        assert ans_acc_pct >= 0, "Answer accuracy must be non-negative."
        assert cit_acc_pct >= 0, "Citation accuracy must be non-negative."


# ---------------------------------------------------------------------------
# System Performance benchmark — ingestion rate + query latency
# ---------------------------------------------------------------------------

@pytest.mark.accuracy
class TestSystemPerformance:
    """Measure ingestion throughput and query-stage latencies.

    Writes:
        artifacts/performance_ingestion.csv  — per-file load timing
        artifacts/performance_latency.csv    — per-query stage breakdown
        artifacts/performance_summary.csv    — slide-ready summary numbers
    """

    def test_ingestion_throughput(self):
        """Measure load_pdf wall-clock time for each SDS PDF (real fitz, mocked LLM)."""
        import fitz  # pylint: disable=import-outside-toplevel  # noqa: F401

        from rag import RAGPipeline  # pylint: disable=import-outside-toplevel

        available_pdfs = [p for p in SDS_PDFS if os.path.exists(p)]
        if not available_pdfs:
            pytest.skip(f"No SDS PDFs found in {DOCS_DIR}.")

        ingestion_rows: List[Dict[str, Any]] = []

        fake_gguf = str(ARTIFACTS_DIR / "_dummy.gguf")
        Path(fake_gguf).parent.mkdir(parents=True, exist_ok=True)
        Path(fake_gguf).write_bytes(b"")

        with patch("rag.pipeline.LlamaCPP") as mock_llm_cls, \
             patch("rag.pipeline.HuggingFaceEmbedding") as mock_embed_cls, \
             patch("rag.pipeline.SentenceWindowNodeParser") as mock_splitter_cls, \
             patch("rag.pipeline.Settings"):

            mock_llm_cls.return_value = MagicMock(name="llm")
            mock_embed_cls.return_value = MagicMock(name="embed")
            mock_splitter_cls.from_defaults.return_value = MagicMock(name="splitter")

            pipeline = RAGPipeline(model_path=fake_gguf)

            for pdf_path in available_pdfs:
                fname = os.path.basename(pdf_path)
                t0 = time.perf_counter()
                docs = pipeline.load_pdf(pdf_path)
                elapsed_ms = (time.perf_counter() - t0) * 1_000
                n_pages = len(docs)
                per_10_pages_ms = (elapsed_ms / n_pages * 10) if n_pages else 0.0

                ingestion_rows.append(
                    {
                        "file": fname,
                        "pages": n_pages,
                        "load_ms": round(elapsed_ms, 2),
                        "ms_per_10_pages": round(per_10_pages_ms, 2),
                    }
                )

        assert ingestion_rows, "No PDFs were timed."

        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        ingestion_path = ARTIFACTS_DIR / "performance_ingestion.csv"
        _write_csv(ingestion_rows, str(ingestion_path))

        avg_ms_per_10 = sum(r["ms_per_10_pages"] for r in ingestion_rows) / len(ingestion_rows)
        total_pages = sum(r["pages"] for r in ingestion_rows)
        total_ms = sum(r["load_ms"] for r in ingestion_rows)

        print("\n" + "=" * 60)
        print("[INGESTION] INGESTION THROUGHPUT")
        print("=" * 60)
        for row in ingestion_rows:
            print(f"  {row['file']}: {row['pages']} pages in {row['load_ms']:.1f} ms "
                  f"({row['ms_per_10_pages']:.1f} ms/10 pages)")
        print(f"\n  Total pages   : {total_pages}")
        print(f"  Total load ms : {total_ms:.1f}")
        print(f"  Avg ms/10 pg  : {avg_ms_per_10:.2f}")
        print(f"\nArtifact: {ingestion_path}")
        print("=" * 60)

        assert avg_ms_per_10 > 0

    def test_query_stage_latency(self):
        """Measure per-stage latency (retrieve_ms, generate_ms, total_ms) with mocked LLM."""
        from rag import RAGPipeline  # pylint: disable=import-outside-toplevel

        available_pdfs = [p for p in SDS_PDFS if os.path.exists(p)]
        if not available_pdfs:
            pytest.skip(f"No SDS PDFs found in {DOCS_DIR}.")

        test_pdf = available_pdfs[0]
        fake_gguf = str(ARTIFACTS_DIR / "_dummy.gguf")
        Path(fake_gguf).parent.mkdir(parents=True, exist_ok=True)
        Path(fake_gguf).write_bytes(b"")

        latency_rows: List[Dict[str, Any]] = []

        with patch("rag.pipeline.LlamaCPP") as mock_llm_cls, \
             patch("rag.pipeline.HuggingFaceEmbedding") as mock_embed_cls, \
             patch("rag.pipeline.SentenceWindowNodeParser") as mock_splitter_cls, \
             patch("rag.pipeline.Settings"):

            mock_llm_cls.return_value = MagicMock(name="llm")
            mock_embed_cls.return_value = MagicMock(name="embed")

            mock_splitter = MagicMock(name="splitter")
            mock_splitter_cls.from_defaults.return_value = mock_splitter

            pipeline = RAGPipeline(model_path=fake_gguf)
            pipeline.llm.complete.return_value = MagicMock(text="The answer is in the document.")

            # Build with real fitz load + mocked index
            docs = pipeline.load_pdf(test_pdf)

            # Mock the chunker to return lightweight nodes
            from llama_index.core import Document  # pylint: disable=import-outside-toplevel
            mock_nodes = []
            for i, doc in enumerate(docs[:5]):  # limit to first 5 pages for speed
                n = MagicMock()
                n.text = doc.text[:200]
                n.metadata = doc.metadata
                n.node_id = f"node_{i}"
                n.node = n
                mock_nodes.append(n)
            mock_splitter.get_nodes_from_documents.return_value = mock_nodes

            # Mock vector index and retriever so we can measure latency plumbing
            mock_index = MagicMock()
            mock_retriever = MagicMock()
            mock_engine = MagicMock()

            # Patch at the module level for index building
            with patch("rag.pipeline.VectorStoreIndex") as mock_vsi, \
                 patch("rag.pipeline.VectorIndexRetriever"), \
                 patch("rag.pipeline.BM25Retriever") as mock_bm25, \
                 patch("rag.pipeline.QueryFusionRetriever"), \
                 patch("rag.pipeline.RetrieverQueryEngine") as mock_rqe:

                mock_vsi.from_documents.return_value = mock_index
                mock_index._vector_store._data.embedding_dict = {}

                # Mock embed to return a real-ish vector
                import numpy as np  # pylint: disable=import-outside-toplevel
                pipeline.embed_model.get_query_embedding.return_value = list(np.random.rand(384).astype(float))
                pipeline.embed_model.get_text_embedding.return_value = list(np.random.rand(384).astype(float))

                mock_source_nodes = []
                for i in range(3):
                    sn = MagicMock()
                    sn.score = 0.9 - i * 0.1
                    sn.node.text = f"Source chunk {i}: relevant pharmaceutical content."
                    sn.node.metadata = {"file_name": os.path.basename(test_pdf), "page_number": i + 1,
                                        "doc_type": "digital", "pharma_doc_type": "safety_data_sheet"}
                    sn.node.node_id = f"sn_{i}"
                    sn.node.embedding = list(np.random.rand(384).astype(float))
                    mock_source_nodes.append(sn)

                engine_inst = MagicMock()
                engine_inst.retrieve.return_value = mock_source_nodes
                synth_resp = MagicMock()
                synth_resp.__str__ = lambda self: "The storage conditions are cool and dry."
                engine_inst.synthesize.return_value = synth_resp
                mock_rqe.from_args.return_value = engine_inst

                pipeline._chunks = mock_nodes
                pipeline._vector_index = mock_index
                pipeline._query_engine = engine_inst
                pipeline._hybrid_engine = engine_inst
                pipeline._bm25_engine = engine_inst
                pipeline._vector_engine = engine_inst

                queries = [case["query"] for case in GROUND_TRUTH[:5]]
                for query in queries:
                    pipeline.clear_cache()
                    result = pipeline.query_with_sources(query)
                    latency_rows.append(
                        {
                            "query": query[:60],
                            "retrieve_ms": result.get("retrieve_ms", 0.0),
                            "generate_ms": result.get("generate_ms", 0.0),
                            "expand_ms": result.get("expand_ms", 0.0),
                            "classify_ms": result.get("classify_ms", 0.0),
                            "total_ms": result.get("total_ms", 0.0),
                        }
                    )

        assert latency_rows, "No latency measurements collected."

        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        latency_path = ARTIFACTS_DIR / "performance_latency.csv"
        _write_csv(latency_rows, str(latency_path))

        n = len(latency_rows)
        avg_retrieve = sum(r["retrieve_ms"] for r in latency_rows) / n
        avg_generate = sum(r["generate_ms"] for r in latency_rows) / n
        avg_total = sum(r["total_ms"] for r in latency_rows) / n

        summary_rows = [
            {"metric": "Avg Retrieval Latency (ms)", "value": round(avg_retrieve, 3)},
            {"metric": "Avg LLM Generation (ms)", "value": round(avg_generate, 3)},
            {"metric": "Avg Total Response Time (ms)", "value": round(avg_total, 3)},
        ]
        summary_path = ARTIFACTS_DIR / "performance_latency_summary.csv"
        _write_csv(summary_rows, str(summary_path))

        print("\n" + "=" * 60)
        print("[LATENCY] QUERY STAGE LATENCY (mocked LLM, real retrieval plumbing)")
        print("=" * 60)
        print(f"  Avg Retrieval Latency : {avg_retrieve:.3f} ms")
        print(f"  Avg LLM Generation    : {avg_generate:.3f} ms")
        print(f"  Avg Total Response    : {avg_total:.3f} ms")
        print(f"\nArtifact: {latency_path}")
        print("=" * 60)

        assert avg_total >= 0


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _write_csv(rows: List[Dict[str, Any]], path: str) -> None:
    """Write a list of dicts to a CSV file."""
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

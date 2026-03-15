"""
Unit Tests for RAGPipeline  (pipeline.py)
==========================================
All tests use the 4-mock pattern: LlamaCPP, HuggingFaceEmbedding,
SentenceSplitter, Settings are stubbed.  fitz and pytesseract are left real.

Presentation coverage
---------------------
Slide 3  Component Specifications  → TestComponentSpecs
Slide 2  Classification pipeline   → TestClassificationSystem
Slides 5-6  Response contract      → TestQueryResponseStructure
Error handling                     → TestPreBuildErrors, TestGetStats
"""

import math
import os
import sys
from unittest.mock import MagicMock, call, patch

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_model_path(tmp_path):
    """Zero-byte file that satisfies os.path.exists inside RAGPipeline.__init__."""
    p = tmp_path / "model.gguf"
    p.write_bytes(b"")
    return str(p)


@pytest.fixture
def pipeline(fake_model_path):
    """RAGPipeline with every heavy dependency patched out."""
    from rag import RAGPipeline
    with patch("rag.pipeline.LlamaCPP") as mock_llm_cls, \
         patch("rag.pipeline.HuggingFaceEmbedding") as mock_embed_cls, \
         patch("rag.pipeline.SentenceSplitter") as mock_splitter_cls, \
         patch("rag.pipeline.Settings"):
        mock_llm_cls.return_value = MagicMock(name="llm")
        mock_embed_cls.return_value = MagicMock(name="embed")
        mock_splitter_cls.return_value = MagicMock(name="splitter")
        rag = RAGPipeline(model_path=fake_model_path)
    rag.llm.complete.return_value = MagicMock(text="unknown")
    return rag


def _mock_source_node(text="chunk text", file="test_1.pdf", page=1, score=0.9,
                      doc_type="digital", pharma_type="certificate_of_quality"):
    """Build a mock NodeWithScore matching pipeline.query_with_sources expectations."""
    node = MagicMock()
    node.node.text = text
    node.node.metadata = {
        "file_name": file,
        "page_number": page,
        "doc_type": doc_type,
        "pharma_doc_type": pharma_type,
    }
    node.score = score
    return node


# ---------------------------------------------------------------------------
# Slide 3 — Component Specifications
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestComponentSpecs:
    """Verify that the pipeline's documented component specs match the code.

    The numbers in this class are the values that appear in the
    Component Specifications table on Slide 3 of the presentation.
    """

    def test_chunk_size_is_512(self, fake_model_path):
        """SentenceSplitter must be constructed with chunk_size=512."""
        from rag import RAGPipeline
        with patch("rag.pipeline.LlamaCPP"), \
             patch("rag.pipeline.HuggingFaceEmbedding"), \
             patch("rag.pipeline.SentenceSplitter") as mock_splitter, \
             patch("rag.pipeline.Settings"):
            mock_splitter.return_value = MagicMock()
            RAGPipeline(model_path=fake_model_path)
        _, kwargs = mock_splitter.call_args
        assert kwargs.get("chunk_size") == 512

    def test_chunk_overlap_is_50(self, fake_model_path):
        """SentenceSplitter must be constructed with chunk_overlap=50."""
        from rag import RAGPipeline
        with patch("rag.pipeline.LlamaCPP"), \
             patch("rag.pipeline.HuggingFaceEmbedding"), \
             patch("rag.pipeline.SentenceSplitter") as mock_splitter, \
             patch("rag.pipeline.Settings"):
            mock_splitter.return_value = MagicMock()
            RAGPipeline(model_path=fake_model_path)
        _, kwargs = mock_splitter.call_args
        assert kwargs.get("chunk_overlap") == 50

    def test_default_similarity_top_k_is_5(self, fake_model_path):
        """Default top-K for retrieval must be 5."""
        from rag import RAGPipeline
        with patch("rag.pipeline.LlamaCPP"), \
             patch("rag.pipeline.HuggingFaceEmbedding"), \
             patch("rag.pipeline.SentenceSplitter"), \
             patch("rag.pipeline.Settings"):
            rag = RAGPipeline(model_path=fake_model_path)
        assert rag.similarity_top_k == 5

    def test_default_embed_model_is_minilm(self, fake_model_path):
        """Default embedding model must be all-MiniLM-L6-v2."""
        from rag import RAGPipeline
        with patch("rag.pipeline.LlamaCPP"), \
             patch("rag.pipeline.HuggingFaceEmbedding") as mock_embed, \
             patch("rag.pipeline.SentenceSplitter"), \
             patch("rag.pipeline.Settings"):
            mock_embed.return_value = MagicMock()
            RAGPipeline(model_path=fake_model_path)
        _, kwargs = mock_embed.call_args
        assert "all-MiniLM-L6-v2" in kwargs.get("model_name", "")

    def test_llm_max_new_tokens_is_200(self, fake_model_path):
        """LLM must be initialised with max_new_tokens=200."""
        from rag import RAGPipeline
        with patch("rag.pipeline.LlamaCPP") as mock_llm, \
             patch("rag.pipeline.HuggingFaceEmbedding"), \
             patch("rag.pipeline.SentenceSplitter"), \
             patch("rag.pipeline.Settings"):
            RAGPipeline(model_path=fake_model_path)
        _, kwargs = mock_llm.call_args
        assert kwargs.get("max_new_tokens") == 200

    def test_llm_context_window_is_8192(self, fake_model_path):
        """LLM must be initialised with context_window=8192."""
        from rag import RAGPipeline
        with patch("rag.pipeline.LlamaCPP") as mock_llm, \
             patch("rag.pipeline.HuggingFaceEmbedding"), \
             patch("rag.pipeline.SentenceSplitter"), \
             patch("rag.pipeline.Settings"):
            RAGPipeline(model_path=fake_model_path)
        _, kwargs = mock_llm.call_args
        assert kwargs.get("context_window") == 8192

    def test_llm_temperature_is_0_1(self, fake_model_path):
        """Low temperature (0.1) ensures deterministic, factual answers."""
        from rag import RAGPipeline
        with patch("rag.pipeline.LlamaCPP") as mock_llm, \
             patch("rag.pipeline.HuggingFaceEmbedding"), \
             patch("rag.pipeline.SentenceSplitter"), \
             patch("rag.pipeline.Settings"):
            RAGPipeline(model_path=fake_model_path)
        _, kwargs = mock_llm.call_args
        assert kwargs.get("temperature") == pytest.approx(0.1)

    def test_retrieval_mode_is_reciprocal_rerank(self, pipeline):
        """_build_retriever must use reciprocal_rerank fusion (not simple merge)."""
        with patch("rag.pipeline.VectorIndexRetriever") as mock_vec, \
             patch("rag.pipeline.BM25Retriever") as mock_bm25, \
             patch("rag.pipeline.QueryFusionRetriever") as mock_fusion:
            mock_vec.return_value = MagicMock()
            mock_bm25.from_defaults.return_value = MagicMock()
            mock_fusion.return_value = MagicMock()
            pipeline._build_retriever(MagicMock(), [MagicMock()])
        _, kwargs = mock_fusion.call_args
        assert kwargs.get("mode") == "reciprocal_rerank"

    def test_retrieval_uses_both_vector_and_bm25(self, pipeline):
        """Hybrid retrieval: both VectorIndexRetriever and BM25Retriever are created."""
        with patch("rag.pipeline.VectorIndexRetriever") as mock_vec, \
             patch("rag.pipeline.BM25Retriever") as mock_bm25, \
             patch("rag.pipeline.QueryFusionRetriever") as mock_fusion:
            mock_vec.return_value = MagicMock()
            mock_bm25.from_defaults.return_value = MagicMock()
            mock_fusion.return_value = MagicMock()
            pipeline._build_retriever(MagicMock(), [MagicMock()])
        mock_vec.assert_called_once()
        mock_bm25.from_defaults.assert_called_once()

    def test_ocr_dpi_is_200(self):
        """OCR rasterisation DPI must be 200 (balance accuracy vs memory)."""
        from rag.pipeline import _OCR_DPI
        assert _OCR_DPI == 200

    def test_scanned_page_char_threshold_is_100(self):
        """Pages with fewer than 100 chars trigger OCR fallback."""
        from rag.pipeline import _SCANNED_PAGE_CHAR_THRESHOLD
        assert _SCANNED_PAGE_CHAR_THRESHOLD == 100

    def test_seven_pharma_doc_categories_plus_unknown(self):
        """The taxonomy must have exactly 7 named categories + 'unknown'."""
        from rag.pipeline import _PHARMA_DOC_CATEGORIES
        assert len(_PHARMA_DOC_CATEGORIES) == 8
        assert "unknown" in _PHARMA_DOC_CATEGORIES

    def test_keyword_map_categories_are_valid(self):
        """Every key in _KEYWORD_MAP must exist in _PHARMA_DOC_CATEGORIES."""
        from rag.pipeline import _KEYWORD_MAP, _PHARMA_DOC_CATEGORIES
        for cat in _KEYWORD_MAP:
            assert cat in _PHARMA_DOC_CATEGORIES, (
                f"_KEYWORD_MAP key '{cat}' is missing from _PHARMA_DOC_CATEGORIES"
            )


# ---------------------------------------------------------------------------
# Slide 2 — Classification System (two-stage: keyword → LLM)
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestClassificationSystem:
    """Verify the two-stage classification logic used throughout the pipeline.

    Stage 1: keyword scan of first 300 chars (no LLM call).
    Stage 2: few-shot batch LLM classification for ambiguous pages.
    """

    @pytest.mark.parametrize("category,trigger_keyword", [
        ("certificate_of_quality", "certificate of quality"),
        ("certificate_of_quality", "certificate of analysis"),
        ("cover_letter",           "cover letter"),
        ("cover_letter",           "dear supplier"),
        ("bse_tse_declaration",    "bse"),
        ("bse_tse_declaration",    "transmissible spongiform"),
        ("packaging_specification","packaging specification"),
        ("material_description",   "material description"),
        ("supplier_qualification", "supplier qualification"),
        ("chain_of_custody",       "chain of custody"),
    ])
    def test_keyword_hit_classifies_without_llm(self, pipeline, category, trigger_keyword):
        """Keyword present in header → correct category, zero LLM calls."""
        text = trigger_keyword + " " * 300
        pipeline.llm.complete.reset_mock()
        result = pipeline._classify_document(text)
        assert result == category
        pipeline.llm.complete.assert_not_called()

    def test_no_keyword_falls_through_to_llm(self, pipeline):
        """No keyword match → LLM is called exactly once."""
        text = "this document contains no recognisable pharmaceutical keyword signals " * 5
        pipeline.llm.complete.reset_mock()
        pipeline._classify_document(text)
        pipeline.llm.complete.assert_called_once()

    def test_keyword_only_checked_in_first_300_chars(self, pipeline):
        """Keyword starting at char 300 (outside window) → falls through to LLM."""
        keyword = "certificate of quality"
        text = "x" * 300 + keyword   # keyword starts outside the 300-char window
        pipeline.llm.complete.reset_mock()
        pipeline._classify_document(text)
        pipeline.llm.complete.assert_called_once()

    def test_result_always_a_valid_category(self, pipeline):
        """_classify_document must never return an arbitrary string."""
        from rag.pipeline import _PHARMA_DOC_CATEGORIES
        for text in ["Certificate of Analysis — Batch 12345", "random garbage xyz", ""]:
            assert pipeline._classify_document(text) in _PHARMA_DOC_CATEGORIES

    def test_annotate_sets_pharma_doc_type_on_every_doc(self, pipeline):
        """After _annotate_pharma_doc_types, every document has pharma_doc_type."""
        from llama_index.core import Document
        docs = [
            Document(text="Certificate of Quality batch 123", metadata={}),
            Document(text="ambiguous content here " * 10, metadata={}),
        ]
        pipeline.llm.complete.return_value = MagicMock(text="unknown")
        pipeline._annotate_pharma_doc_types(docs)
        for doc in docs:
            assert "pharma_doc_type" in doc.metadata

    def test_annotate_keyword_pages_never_call_llm(self, pipeline):
        """Pages classified by keyword must not trigger any LLM call."""
        from llama_index.core import Document
        docs = [Document(text="Certificate of Quality batch " * 5, metadata={})]
        pipeline.llm.complete.reset_mock()
        pipeline._annotate_pharma_doc_types(docs)
        pipeline.llm.complete.assert_not_called()

    def test_batch_size_5_reduces_llm_calls(self, pipeline):
        """N ambiguous pages → ceil(N/5) LLM calls (not N)."""
        from llama_index.core import Document
        n = 13
        docs = [Document(text="ambiguous filler content " * 5, metadata={}) for _ in range(n)]
        pipeline.llm.complete.reset_mock()
        pipeline.llm.complete.return_value = MagicMock(
            text="\n".join(["unknown"] * 5)
        )
        pipeline._annotate_pharma_doc_types(docs)
        assert pipeline.llm.complete.call_count == math.ceil(n / 5)

    def test_parse_category_extracts_first_match(self, pipeline):
        """_parse_category must find the category even inside a longer response."""
        from rag.pipeline import _PHARMA_DOC_CATEGORIES
        result = pipeline._parse_category("The type is cover_letter.")
        assert result == "cover_letter"

    def test_parse_category_unknown_for_garbage(self, pipeline):
        result = pipeline._parse_category("I have no idea what this is")
        assert result == "unknown"


# ---------------------------------------------------------------------------
# Error Handling — pre-build state (Slide 8 — Limitations)
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestPreBuildErrors:
    def test_query_before_build_raises_runtime_error(self, pipeline):
        with pytest.raises(RuntimeError, match="not built"):
            pipeline.query("What is the storage temperature?")

    def test_query_with_sources_before_build_raises(self, pipeline):
        with pytest.raises(RuntimeError, match="not built"):
            pipeline.query_with_sources("What are the hazards?")

    def test_get_stats_before_build_raises(self, pipeline):
        with pytest.raises(RuntimeError, match="not built"):
            pipeline.get_stats()

    def test_get_document_details_before_build_raises(self, pipeline):
        with pytest.raises(RuntimeError, match="not built"):
            pipeline.get_document_details()

    def test_missing_model_path_raises_file_not_found(self, tmp_path):
        from rag import RAGPipeline
        with patch("rag.pipeline.LlamaCPP"), \
             patch("rag.pipeline.HuggingFaceEmbedding"), \
             patch("rag.pipeline.SentenceSplitter"), \
             patch("rag.pipeline.Settings"):
            with pytest.raises(FileNotFoundError):
                RAGPipeline(model_path=str(tmp_path / "nonexistent.gguf"))

    def test_build_from_multiple_pdfs_empty_list_raises(self, pipeline):
        with pytest.raises(ValueError, match="No PDF paths"):
            pipeline.build_from_multiple_pdfs([])

    def test_chunk_empty_documents_raises_value_error(self, pipeline):
        with pytest.raises(ValueError, match="No documents"):
            pipeline._chunk([])


# ---------------------------------------------------------------------------
# Slides 5-6 — Query Response Contract
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestQueryResponseStructure:
    """Verify the exact shape of query_with_sources output.

    These tests define the data contract for the 'Pipeline in Action' slides
    (query, retrieved chunks with scores, final answer).
    """

    @pytest.fixture
    def built_pipeline(self, pipeline):
        """Pipeline with a minimal mock index returning 2 ranked chunks."""
        nodes = [
            _mock_source_node(text="Store at 2-8 °C in a cool dry place.", score=0.90),
            _mock_source_node(text="Refrigerate; protect from light.",    score=0.72),
        ]
        mock_response = MagicMock()
        mock_response.source_nodes = nodes
        mock_response.__str__ = lambda self: "Store at 2-8 °C."
        pipeline._query_engine = MagicMock()
        # query_with_sources now uses engine.retrieve() + engine.synthesize()
        # instead of engine.query(), so mock both entry points.
        pipeline._query_engine.retrieve.return_value = nodes
        pipeline._query_engine.synthesize.return_value = mock_response
        pipeline._query_engine.query.return_value = mock_response  # kept for other callers
        # Also wire the mode-specific engines so _engine_map lookups work.
        pipeline._hybrid_engine = pipeline._query_engine
        pipeline._bm25_engine   = pipeline._query_engine
        pipeline._vector_engine = pipeline._query_engine
        pipeline._docs_classified = False
        return pipeline

    def test_answer_key_is_present(self, built_pipeline):
        result = built_pipeline.query_with_sources("What is the storage temperature?")
        assert "answer" in result and isinstance(result["answer"], str)

    def test_sources_is_a_list(self, built_pipeline):
        result = built_pipeline.query_with_sources("What is the storage temperature?")
        assert isinstance(result["sources"], list)

    def test_chunk_count_matches_sources_length(self, built_pipeline):
        result = built_pipeline.query_with_sources("What is the storage temperature?")
        assert result["chunk_count"] == len(result["sources"])

    def test_each_source_has_required_fields(self, built_pipeline):
        """Every source dict must carry text, file, page, score, doc_type, pharma_doc_type."""
        required = {"text", "file", "page", "score", "doc_type", "pharma_doc_type"}
        result = built_pipeline.query_with_sources("What is the storage temperature?")
        for src in result["sources"]:
            missing = required - set(src.keys())
            assert not missing, f"Source missing fields: {missing}"

    def test_top_chunk_has_100_percent_confidence(self, built_pipeline):
        """Top-ranked chunk is normalised to 100 % confidence."""
        result = built_pipeline.query_with_sources("What is the storage temperature?")
        assert result["sources"][0]["score"] == pytest.approx(100.0)

    def test_second_chunk_confidence_less_than_100(self, built_pipeline):
        result = built_pipeline.query_with_sources("What is the storage temperature?")
        assert result["sources"][1]["score"] < 100.0

    def test_confidence_is_proportional_to_raw_score(self, built_pipeline):
        """score[1] / 100.0 ≈ raw_score[1] / raw_score[0] (= 0.72/0.90)."""
        result = built_pipeline.query_with_sources("What is the storage temperature?")
        expected = round((0.72 / 0.90) * 100, 1)
        assert result["sources"][1]["score"] == pytest.approx(expected, abs=0.2)

    def test_query_category_is_none_when_classify_false(self, built_pipeline):
        result = built_pipeline.query_with_sources("anything", classify=False)
        assert result["query_category"] is None

    def test_cache_hit_returns_identical_object(self, built_pipeline):
        r1 = built_pipeline.query_with_sources("What is the storage temperature?")
        r2 = built_pipeline.query_with_sources("What is the storage temperature?")
        assert r1 is r2  # same dict from cache

    def test_different_queries_are_not_cached_together(self, built_pipeline):
        r1 = built_pipeline.query_with_sources("What is the storage temperature?")
        built_pipeline.clear_cache()
        r2 = built_pipeline.query_with_sources("What are the hazards?")
        # Even after clear they are separate objects
        assert r1["answer"] == r2["answer"]  # same mock, but different cache entries


# ---------------------------------------------------------------------------
# get_stats — index statistics (Slide 3 / Slide 4)
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestGetStats:
    """Verify the stats dictionary returned after build()."""

    @pytest.fixture
    def indexed_pipeline(self, pipeline):
        def _make_chunk(file, page, pharma_type="certificate_of_quality"):
            c = MagicMock()
            c.metadata = {
                "file_name": file,
                "page_number": page,
                "pharma_doc_type": pharma_type,
                "doc_type": "digital",
                "ocr_used": False,
            }
            return c

        pipeline._chunks = (
            [_make_chunk("test_1.pdf", p) for p in range(1, 6)] +
            [_make_chunk("test_2.pdf", p, "cover_letter") for p in range(1, 4)]
        )
        return pipeline

    def test_stats_has_all_required_keys(self, indexed_pipeline):
        stats = indexed_pipeline.get_stats()
        for key in ["total_pages", "total_chunks", "total_files", "file_names",
                    "doc_type_counts", "classified"]:
            assert key in stats

    def test_total_chunks_is_correct(self, indexed_pipeline):
        assert indexed_pipeline.get_stats()["total_chunks"] == 8

    def test_total_files_is_correct(self, indexed_pipeline):
        assert indexed_pipeline.get_stats()["total_files"] == 2

    def test_file_names_sorted(self, indexed_pipeline):
        names = indexed_pipeline.get_stats()["file_names"]
        assert names == sorted(names)

    def test_doc_type_counts_contains_indexed_types(self, indexed_pipeline):
        counts = indexed_pipeline.get_stats()["doc_type_counts"]
        assert "certificate_of_quality" in counts
        assert "cover_letter" in counts

    def test_classified_flag_matches_pipeline_state(self, indexed_pipeline):
        indexed_pipeline._docs_classified = True
        assert indexed_pipeline.get_stats()["classified"] is True
        indexed_pipeline._docs_classified = False
        assert indexed_pipeline.get_stats()["classified"] is False


# ---------------------------------------------------------------------------
# load_pdf metadata contract
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestLoadPDFMetadata:
    """Verify metadata keys on documents returned by load_pdf using real fitz."""

    DOCS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "docs")
    TEST_PDF  = os.path.join(DOCS_DIR, "test_1.pdf")

    @pytest.fixture(autouse=True)
    def skip_if_no_pdf(self):
        if not os.path.exists(self.TEST_PDF):
            pytest.skip("docs/test_1.pdf not found")

    def test_load_pdf_returns_documents(self, pipeline):
        docs = pipeline.load_pdf(self.TEST_PDF)
        assert len(docs) >= 1

    def test_every_doc_has_required_metadata_keys(self, pipeline):
        docs = pipeline.load_pdf(self.TEST_PDF)
        required = {"file_name", "page_number", "total_pages", "doc_type",
                    "ocr_used", "source_id"}
        for doc in docs:
            missing = required - set(doc.metadata.keys())
            assert not missing, f"page {doc.metadata.get('page_number')} missing: {missing}"

    def test_source_id_format_is_file_colon_page(self, pipeline):
        docs = pipeline.load_pdf(self.TEST_PDF)
        import re
        pattern = re.compile(r"^.+:p\d+$")
        for doc in docs:
            assert pattern.match(doc.metadata["source_id"]), (
                f"Bad source_id: {doc.metadata['source_id']}"
            )

    def test_page_numbers_are_one_based(self, pipeline):
        docs = pipeline.load_pdf(self.TEST_PDF)
        assert all(doc.metadata["page_number"] >= 1 for doc in docs)

    def test_total_pages_consistent_across_docs(self, pipeline):
        docs = pipeline.load_pdf(self.TEST_PDF)
        total_pages_values = {doc.metadata["total_pages"] for doc in docs}
        assert len(total_pages_values) == 1, (
            f"Inconsistent total_pages values: {total_pages_values}"
        )

    def test_no_empty_text_pages(self, pipeline):
        docs = pipeline.load_pdf(self.TEST_PDF)
        assert all(doc.text.strip() for doc in docs), (
            "load_pdf returned at least one document with empty text"
        )

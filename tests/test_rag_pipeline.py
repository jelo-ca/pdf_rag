"""
Tests for RAGPipeline
=====================
All heavy dependencies (LlamaCPP, HuggingFaceEmbedding, fitz, etc.) are mocked
so the suite runs without GPU hardware or large model files.

Run with:
    pytest tests/test_rag_pipeline.py -v
"""

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

# Ensure the project root is importable regardless of where pytest is invoked.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# Shared helpers / fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_model_path(tmp_path):
    """A real (zero-byte) file that satisfies os.path.exists."""
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

    return rag


def _make_source_node(
    text="chunk text",
    file="doc.pdf",
    page=1,
    score=0.8,
    doc_type="digital",
    pharma_doc_type="unknown",
):
    """Factory for mocked NodeWithScore objects returned by the query engine."""
    node_inner = MagicMock()
    node_inner.metadata = {
        "file_name": file,
        "page_number": page,
        "doc_type": doc_type,
        "pharma_doc_type": pharma_doc_type,
    }
    node_inner.text = text

    source_node = MagicMock()
    source_node.score = score
    source_node.node = node_inner
    return source_node


def _make_chunk(pharma_doc_type="certificate_of_quality"):
    """Factory for mocked BaseNode chunks stored in pipeline._chunks."""
    chunk = MagicMock()
    chunk.metadata = {"pharma_doc_type": pharma_doc_type}
    return chunk


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------

class TestInit:
    def test_raises_file_not_found_for_missing_model(self):
        from rag import RAGPipeline

        with patch("rag.pipeline.LlamaCPP"), \
             patch("rag.pipeline.HuggingFaceEmbedding"), \
             patch("rag.pipeline.SentenceSplitter"), \
             patch("rag.pipeline.Settings"):
            with pytest.raises(FileNotFoundError, match="GGUF model not found"):
                RAGPipeline(model_path="/nonexistent/path/model.gguf")

    def test_uses_model_path_env_var(self, tmp_path, monkeypatch):
        from rag import RAGPipeline

        fake_model = tmp_path / "env_model.gguf"
        fake_model.write_bytes(b"")
        monkeypatch.setenv("MODEL_PATH", str(fake_model))

        with patch("rag.pipeline.LlamaCPP") as mock_llm_cls, \
             patch("rag.pipeline.HuggingFaceEmbedding"), \
             patch("rag.pipeline.SentenceSplitter"), \
             patch("rag.pipeline.Settings"):
            mock_llm_cls.return_value = MagicMock()
            RAGPipeline()  # no model_path arg

        assert mock_llm_cls.call_args.kwargs["model_path"] == str(fake_model)

    def test_default_similarity_top_k_is_5(self, fake_model_path, monkeypatch):
        from rag import RAGPipeline

        monkeypatch.delenv("SIMILARITY_TOP_K", raising=False)

        with patch("rag.pipeline.LlamaCPP"), \
             patch("rag.pipeline.HuggingFaceEmbedding"), \
             patch("rag.pipeline.SentenceSplitter"), \
             patch("rag.pipeline.Settings"):
            rag = RAGPipeline(model_path=fake_model_path)

        assert rag.similarity_top_k == 5

    def test_custom_similarity_top_k(self, fake_model_path):
        from rag import RAGPipeline

        with patch("rag.pipeline.LlamaCPP"), \
             patch("rag.pipeline.HuggingFaceEmbedding"), \
             patch("rag.pipeline.SentenceSplitter"), \
             patch("rag.pipeline.Settings"):
            rag = RAGPipeline(model_path=fake_model_path, similarity_top_k=10)

        assert rag.similarity_top_k == 10

    def test_similarity_top_k_from_env(self, fake_model_path, monkeypatch):
        from rag import RAGPipeline

        monkeypatch.setenv("SIMILARITY_TOP_K", "7")
        with patch("rag.pipeline.LlamaCPP"), \
             patch("rag.pipeline.HuggingFaceEmbedding"), \
             patch("rag.pipeline.SentenceSplitter"), \
             patch("rag.pipeline.Settings"):
            rag = RAGPipeline(model_path=fake_model_path)

        assert rag.similarity_top_k == 7

    def test_similarity_top_k_arg_overrides_env(self, fake_model_path, monkeypatch):
        from rag import RAGPipeline

        monkeypatch.setenv("SIMILARITY_TOP_K", "99")
        with patch("rag.pipeline.LlamaCPP"), \
             patch("rag.pipeline.HuggingFaceEmbedding"), \
             patch("rag.pipeline.SentenceSplitter"), \
             patch("rag.pipeline.Settings"):
            rag = RAGPipeline(model_path=fake_model_path, similarity_top_k=3)

        assert rag.similarity_top_k == 3

    def test_initial_state_is_empty(self, pipeline):
        assert pipeline._vector_index is None
        assert pipeline._query_engine is None
        assert pipeline._pdf_path is None
        assert pipeline._chunks == []
        assert pipeline._docs_classified is False

    def test_default_num_queries_is_1(self, pipeline):
        assert pipeline.num_queries == 1

    def test_custom_num_queries(self, fake_model_path):
        from rag import RAGPipeline

        with patch("rag.pipeline.LlamaCPP"), \
             patch("rag.pipeline.HuggingFaceEmbedding"), \
             patch("rag.pipeline.SentenceSplitter"), \
             patch("rag.pipeline.Settings"):
            rag = RAGPipeline(model_path=fake_model_path, num_queries=4)

        assert rag.num_queries == 4


# ---------------------------------------------------------------------------
# load_pdf
# ---------------------------------------------------------------------------

class TestLoadPdf:

    def _mock_doc(self, pages, total=None):
        """Return a context-manager-compatible fitz.Document mock."""
        mock_doc = MagicMock()
        mock_doc.__iter__ = MagicMock(return_value=iter(pages))
        mock_doc.__len__ = MagicMock(return_value=total if total is not None else len(pages))
        mock_doc.__enter__ = MagicMock(return_value=mock_doc)
        mock_doc.__exit__ = MagicMock(return_value=False)
        return mock_doc

    def _page(self, text):
        p = MagicMock()
        p.get_text.return_value = text
        return p

    def test_digital_page_metadata(self, pipeline, tmp_path):
        page = self._page("A" * 200)
        mock_doc = self._mock_doc([page])

        with patch("rag.pipeline.fitz.open", return_value=mock_doc):
            docs = pipeline.load_pdf(str(tmp_path / "doc.pdf"))

        assert len(docs) == 1
        meta = docs[0].metadata
        assert meta["page_number"] == 1
        assert meta["total_pages"] == 1
        assert meta["ocr_used"] is False
        assert meta["doc_type"] == "digital"
        assert meta["file_name"] == "doc.pdf"
        assert meta["source_id"] == "doc.pdf:p1"

    def test_skips_fully_empty_pages(self, pipeline, tmp_path):
        """Pages that yield no text after all extraction attempts are dropped."""
        page = self._page("   \n  \t  ")
        mock_doc = self._mock_doc([page])

        with patch("rag.pipeline.fitz.open", return_value=mock_doc), \
             patch("rag.pipeline._OCR_AVAILABLE", False):
            docs = pipeline.load_pdf(str(tmp_path / "empty.pdf"))

        assert docs == []

    def test_ocr_fallback_triggered_for_sparse_page(self, pipeline, tmp_path):
        """Pages with < 100 chars trigger OCR when available."""
        sparse_page = self._page("short")  # < 100 chars
        mock_doc = self._mock_doc([sparse_page])
        ocr_text = "OCR extracted text " * 20

        with patch("rag.pipeline.fitz.open", return_value=mock_doc), \
             patch("rag.pipeline._OCR_AVAILABLE", True), \
             patch.object(pipeline, "_ocr_page", return_value=ocr_text) as mock_ocr:
            docs = pipeline.load_pdf(str(tmp_path / "scan.pdf"))

        mock_ocr.assert_called_once_with(sparse_page)
        assert len(docs) == 1
        assert docs[0].metadata["ocr_used"] is True
        assert docs[0].metadata["doc_type"] == "scanned"

    def test_no_ocr_when_unavailable(self, pipeline, tmp_path):
        """When OCR is unavailable, sparse page is kept as-is (not blank)."""
        sparse_page = self._page("hi")  # < 100 chars, but non-empty
        mock_doc = self._mock_doc([sparse_page])

        with patch("rag.pipeline.fitz.open", return_value=mock_doc), \
             patch("rag.pipeline._OCR_AVAILABLE", False), \
             patch.object(pipeline, "_ocr_page") as mock_ocr:
            docs = pipeline.load_pdf(str(tmp_path / "scan.pdf"))

        mock_ocr.assert_not_called()
        assert len(docs) == 1
        assert docs[0].metadata["ocr_used"] is False

    def test_no_ocr_when_tesseract_binary_missing(self, pipeline, tmp_path):
        """If pytesseract imports but no tesseract binary exists, OCR is skipped safely."""
        sparse_page = self._page("hi")
        mock_doc = self._mock_doc([sparse_page])

        with patch("rag.pipeline.fitz.open", return_value=mock_doc), \
             patch("rag.pipeline._OCR_AVAILABLE", True), \
             patch("rag.pipeline._is_ocr_runtime_available", return_value=False), \
             patch.object(pipeline, "_ocr_page") as mock_ocr:
            docs = pipeline.load_pdf(str(tmp_path / "scan.pdf"))

        mock_ocr.assert_not_called()
        assert len(docs) == 1
        assert docs[0].metadata["ocr_used"] is False

    def test_ocr_page_yields_empty_string_is_skipped(self, pipeline, tmp_path):
        """If OCR returns blank text the page is discarded."""
        sparse_page = self._page("hi")
        mock_doc = self._mock_doc([sparse_page])

        with patch("rag.pipeline.fitz.open", return_value=mock_doc), \
             patch("rag.pipeline._OCR_AVAILABLE", True), \
             patch.object(pipeline, "_ocr_page", return_value="   "):
            docs = pipeline.load_pdf(str(tmp_path / "blank_scan.pdf"))

        assert docs == []

    def test_multi_page_numbering_and_empty_skip(self, pipeline, tmp_path):
        pages = [self._page("A" * 200), self._page("B" * 200), self._page("   ")]
        mock_doc = self._mock_doc(pages, total=3)

        with patch("rag.pipeline.fitz.open", return_value=mock_doc), \
             patch("rag.pipeline._OCR_AVAILABLE", False):
            docs = pipeline.load_pdf(str(tmp_path / "multi.pdf"))

        assert len(docs) == 2
        assert docs[0].metadata["page_number"] == 1
        assert docs[1].metadata["page_number"] == 2
        assert docs[0].metadata["total_pages"] == 3

    def test_source_id_format(self, pipeline, tmp_path):
        page = self._page("A" * 200)
        mock_doc = self._mock_doc([page])

        with patch("rag.pipeline.fitz.open", return_value=mock_doc):
            docs = pipeline.load_pdf(str(tmp_path / "report.pdf"))

        assert docs[0].metadata["source_id"] == "report.pdf:p1"

    def test_all_pages_empty_returns_empty_list(self, pipeline, tmp_path):
        pages = [self._page(""), self._page("  "), self._page("\n")]
        mock_doc = self._mock_doc(pages)

        with patch("rag.pipeline.fitz.open", return_value=mock_doc), \
             patch("rag.pipeline._OCR_AVAILABLE", False):
            docs = pipeline.load_pdf(str(tmp_path / "allblank.pdf"))

        assert docs == []


# ---------------------------------------------------------------------------
# _chunk
# ---------------------------------------------------------------------------

class TestChunk:

    def test_raises_value_error_on_empty_input(self, pipeline):
        with pytest.raises(ValueError, match="No documents provided"):
            pipeline._chunk([])

    def test_delegates_to_splitter(self, pipeline):
        from llama_index.core import Document

        docs = [Document(text="Pharmaceutical content.")]
        mock_chunks = [MagicMock(), MagicMock()]
        pipeline._splitter.get_nodes_from_documents.return_value = mock_chunks

        result = pipeline._chunk(docs)

        pipeline._splitter.get_nodes_from_documents.assert_called_once_with(docs)
        assert result is mock_chunks

    def test_single_document_returns_chunks(self, pipeline):
        from llama_index.core import Document

        single_chunk = [MagicMock()]
        pipeline._splitter.get_nodes_from_documents.return_value = single_chunk

        result = pipeline._chunk([Document(text="Single page content.")])

        assert len(result) == 1


# ---------------------------------------------------------------------------
# query / query_with_sources before build → RuntimeError
# ---------------------------------------------------------------------------

class TestQueryBeforeBuild:

    def test_query_raises_runtime_error(self, pipeline):
        with pytest.raises(RuntimeError, match="Pipeline not built"):
            pipeline.query("What is the dosage?")

    def test_query_with_sources_raises_runtime_error(self, pipeline):
        with pytest.raises(RuntimeError, match="Pipeline not built"):
            pipeline.query_with_sources("What are storage conditions?")


# ---------------------------------------------------------------------------
# query
# ---------------------------------------------------------------------------

class TestQuery:

    def _attach_engine(self, pipeline, answer="Test answer."):
        mock_engine = MagicMock()
        mock_response = MagicMock()
        mock_response.__str__ = MagicMock(return_value=answer)
        mock_engine.query.return_value = mock_response
        pipeline._query_engine = mock_engine
        return mock_engine

    def test_returns_string_answer(self, pipeline):
        engine = self._attach_engine(pipeline)
        result = pipeline.query("What is the active ingredient?")
        assert result == "Test answer."
        engine.query.assert_called_once_with("What is the active ingredient?")

    def test_expand_uses_second_variant(self, pipeline):
        engine = self._attach_engine(pipeline)
        expansions = ["original", "expanded variant 1", "expanded variant 2"]

        with patch.object(pipeline, "expand_query", return_value=expansions) as mock_exp:
            pipeline.query("original", expand=True)

        mock_exp.assert_called_once_with("original", num_expansions=3)
        engine.query.assert_called_once_with("expanded variant 1")

    def test_expand_with_single_variant_falls_back_to_original(self, pipeline):
        engine = self._attach_engine(pipeline)

        with patch.object(pipeline, "expand_query", return_value=["only query"]):
            pipeline.query("only query", expand=True)

        engine.query.assert_called_once_with("only query")

    def test_no_expand_by_default(self, pipeline):
        self._attach_engine(pipeline)

        with patch.object(pipeline, "expand_query") as mock_exp:
            pipeline.query("test question")

        mock_exp.assert_not_called()

    def test_custom_num_expansions_forwarded(self, pipeline):
        self._attach_engine(pipeline)
        expansions = ["q", "v1", "v2", "v3", "v4", "v5"]

        with patch.object(pipeline, "expand_query", return_value=expansions) as mock_exp:
            pipeline.query("q", expand=True, num_expansions=5)

        mock_exp.assert_called_once_with("q", num_expansions=5)


# ---------------------------------------------------------------------------
# query_with_sources
# ---------------------------------------------------------------------------

class TestQueryWithSources:

    def _attach_engine(self, pipeline, source_nodes, answer="Answer."):
        mock_engine = MagicMock()
        mock_response = MagicMock()
        mock_response.__str__ = MagicMock(return_value=answer)
        mock_response.source_nodes = source_nodes
        mock_engine.query.return_value = mock_response
        pipeline._query_engine = mock_engine
        return mock_engine

    def test_result_has_expected_keys(self, pipeline):
        self._attach_engine(pipeline, [_make_source_node(score=0.9)])
        result = pipeline.query_with_sources("test?")
        assert set(result.keys()) == {"answer", "sources", "chunk_count", "query_category"}

    def test_chunk_count_matches_source_list_length(self, pipeline):
        nodes = [_make_source_node(), _make_source_node()]
        self._attach_engine(pipeline, nodes)
        result = pipeline.query_with_sources("test?")
        assert result["chunk_count"] == 2
        assert len(result["sources"]) == 2

    def test_confidence_normalised_to_top_score(self, pipeline):
        nodes = [_make_source_node(score=1.0), _make_source_node(score=0.5)]
        self._attach_engine(pipeline, nodes)
        result = pipeline.query_with_sources("test?")
        assert result["sources"][0]["score"] == 100.0
        assert result["sources"][1]["score"] == 50.0

    def test_rank_based_fallback_when_all_scores_zero(self, pipeline):
        nodes = [
            _make_source_node(score=0.0),
            _make_source_node(score=0.0),
            _make_source_node(score=0.0),
        ]
        self._attach_engine(pipeline, nodes)
        result = pipeline.query_with_sources("test?")
        scores = [s["score"] for s in result["sources"]]
        assert scores[0] == 100.0
        assert scores[1] == 50.0
        assert round(scores[2], 1) == round(100.0 / 3, 1)

    def test_rank_based_fallback_when_scores_are_none(self, pipeline):
        nodes = [_make_source_node(score=None), _make_source_node(score=None)]
        self._attach_engine(pipeline, nodes)
        result = pipeline.query_with_sources("test?")
        assert result["sources"][0]["score"] == 100.0
        assert result["sources"][1]["score"] == 50.0

    def test_source_metadata_fields_propagated(self, pipeline):
        node = _make_source_node(
            text="relevant chunk",
            file="pfizer.pdf",
            page=3,
            score=1.0,
            doc_type="scanned",
            pharma_doc_type="certificate_of_quality",
        )
        self._attach_engine(pipeline, [node])
        src = pipeline.query_with_sources("test?")["sources"][0]
        assert src["text"] == "relevant chunk"
        assert src["file"] == "pfizer.pdf"
        assert src["page"] == 3
        assert src["doc_type"] == "scanned"
        assert src["pharma_doc_type"] == "certificate_of_quality"

    def test_missing_metadata_uses_defaults(self, pipeline):
        node = MagicMock()
        node.score = 1.0
        node.node.metadata = {}
        node.node.text = "text"
        self._attach_engine(pipeline, [node])
        src = pipeline.query_with_sources("test?")["sources"][0]
        assert src["file"] == "unknown"
        assert src["page"] == "?"
        assert src["doc_type"] == "digital"
        assert src["pharma_doc_type"] == "unknown"

    def test_empty_source_nodes_returns_zero_chunks(self, pipeline):
        self._attach_engine(pipeline, [])
        result = pipeline.query_with_sources("test?")
        assert result["chunk_count"] == 0
        assert result["sources"] == []

    def test_answer_string_is_passed_through(self, pipeline):
        self._attach_engine(pipeline, [], answer="Precise pharmaceutical answer.")
        result = pipeline.query_with_sources("anything?")
        assert result["answer"] == "Precise pharmaceutical answer."

    def test_expand_query_used_when_requested(self, pipeline):
        engine = self._attach_engine(pipeline, [])
        expansions = ["q", "alt q"]

        with patch.object(pipeline, "expand_query", return_value=expansions):
            pipeline.query_with_sources("q", expand=True)

        engine.query.assert_called_once_with("alt q")

    def test_mixed_none_and_nonzero_scores(self, pipeline):
        """Nodes where some scores are None fall back to rank-based scoring."""
        nodes = [_make_source_node(score=0.8), _make_source_node(score=None)]
        self._attach_engine(pipeline, nodes)
        result = pipeline.query_with_sources("test?")
        # max_score = 0.8, so first node: 100%, second has score=None → rank fallback
        assert result["sources"][0]["score"] == 100.0
        # second node: score is None → rank fallback: 100 / (1+1) = 50.0
        assert result["sources"][1]["score"] == 50.0


# ---------------------------------------------------------------------------
# expand_query
# ---------------------------------------------------------------------------

class TestExpandQuery:

    def test_prepends_original_when_not_in_response(self, pipeline):
        pipeline.llm.complete.return_value = MagicMock(text="variant A\nvariant B")
        result = pipeline.expand_query("original query")
        assert result[0] == "original query"
        assert "variant A" in result

    def test_does_not_duplicate_original_when_already_present(self, pipeline):
        pipeline.llm.complete.return_value = MagicMock(text="original query\nvariant B")
        result = pipeline.expand_query("original query")
        assert result.count("original query") == 1

    def test_empty_llm_response_returns_only_original(self, pipeline):
        pipeline.llm.complete.return_value = MagicMock(text="\n\n   \n")
        result = pipeline.expand_query("test query")
        assert result == ["test query"]

    def test_blank_lines_are_stripped(self, pipeline):
        pipeline.llm.complete.return_value = MagicMock(text="v1\n\nv2\n  \nv3")
        result = pipeline.expand_query("q")
        # blank / whitespace lines must not appear
        assert all(s.strip() for s in result)

    def test_default_num_expansions_included_in_prompt(self, pipeline):
        pipeline.llm.complete.return_value = MagicMock(text="v1\nv2\nv3")
        pipeline.expand_query("q")
        prompt_text = pipeline.llm.complete.call_args[0][0]
        assert "3" in prompt_text

    def test_custom_num_expansions_included_in_prompt(self, pipeline):
        pipeline.llm.complete.return_value = MagicMock(text="v1\nv2")
        pipeline.expand_query("q", num_expansions=5)
        prompt_text = pipeline.llm.complete.call_args[0][0]
        assert "5" in prompt_text


# ---------------------------------------------------------------------------
# build (integration-style, with all sub-methods patched)
# ---------------------------------------------------------------------------

class TestBuild:

    def _run_build(self, pipeline, fake_pdf):
        mock_docs = [MagicMock()]
        mock_chunks = [MagicMock()]
        mock_index = MagicMock()
        mock_retriever = MagicMock()
        mock_engine = MagicMock()

        with patch.object(pipeline, "load_pdf", return_value=mock_docs) as p_load, \
             patch.object(pipeline, "_chunk", return_value=mock_chunks) as p_chunk, \
             patch.object(pipeline, "_index", return_value=mock_index) as p_index, \
             patch.object(pipeline, "_build_retriever", return_value=mock_retriever) as p_ret, \
             patch("rag.pipeline.RetrieverQueryEngine") as mock_qe_cls:
            mock_qe_cls.from_args.return_value = mock_engine
            pipeline.build(fake_pdf)

        return dict(
            docs=mock_docs, chunks=mock_chunks, index=mock_index,
            retriever=mock_retriever, engine=mock_engine,
            p_load=p_load, p_chunk=p_chunk, p_index=p_index,
            p_ret=p_ret,
        )

    def test_sets_pdf_path(self, pipeline, tmp_path):
        fake_pdf = str(tmp_path / "doc.pdf")
        self._run_build(pipeline, fake_pdf)
        assert pipeline._pdf_path == fake_pdf

    def test_sets_chunks_and_index_and_engine(self, pipeline, tmp_path):
        fake_pdf = str(tmp_path / "doc.pdf")
        ctx = self._run_build(pipeline, fake_pdf)
        assert pipeline._chunks is ctx["chunks"]
        assert pipeline._vector_index is ctx["index"]
        assert pipeline._query_engine is ctx["engine"]

    def test_pipeline_stages_called_in_order(self, pipeline, tmp_path):
        fake_pdf = str(tmp_path / "doc.pdf")
        call_order = []

        with patch.object(pipeline, "load_pdf", side_effect=lambda p: call_order.append("load") or [MagicMock()]), \
             patch.object(pipeline, "_chunk", side_effect=lambda d: call_order.append("chunk") or [MagicMock()]), \
             patch.object(pipeline, "_index", side_effect=lambda c: call_order.append("index") or MagicMock()), \
             patch.object(pipeline, "_build_retriever", side_effect=lambda i, c: call_order.append("retrieve") or MagicMock()), \
             patch("rag.pipeline.RetrieverQueryEngine"):
            pipeline.build(fake_pdf)

        assert call_order == ["load", "chunk", "index", "retrieve"]

    def test_second_build_replaces_old_engine(self, pipeline, tmp_path):
        fake_pdf = str(tmp_path / "doc.pdf")
        old_engine = MagicMock(name="old_engine")
        pipeline._query_engine = old_engine

        ctx = self._run_build(pipeline, fake_pdf)

        assert pipeline._query_engine is ctx["engine"]
        assert pipeline._query_engine is not old_engine

    def test_load_pdf_called_with_correct_path(self, pipeline, tmp_path):
        fake_pdf = str(tmp_path / "specific.pdf")
        ctx = self._run_build(pipeline, fake_pdf)
        ctx["p_load"].assert_called_once_with(fake_pdf)

    def test_chunk_receives_load_pdf_output(self, pipeline, tmp_path):
        fake_pdf = str(tmp_path / "doc.pdf")
        ctx = self._run_build(pipeline, fake_pdf)
        ctx["p_chunk"].assert_called_once_with(ctx["docs"])

    def test_index_receives_chunk_output(self, pipeline, tmp_path):
        fake_pdf = str(tmp_path / "doc.pdf")
        ctx = self._run_build(pipeline, fake_pdf)
        ctx["p_index"].assert_called_once_with(ctx["chunks"])

    def test_build_retriever_receives_index_and_chunks(self, pipeline, tmp_path):
        fake_pdf = str(tmp_path / "doc.pdf")
        ctx = self._run_build(pipeline, fake_pdf)
        ctx["p_ret"].assert_called_once_with(ctx["index"], ctx["chunks"])


# ---------------------------------------------------------------------------
# build – classify_docs flag
# ---------------------------------------------------------------------------

class TestBuildWithClassification:

    def test_docs_classified_false_by_default(self, pipeline, tmp_path):
        fake_pdf = str(tmp_path / "doc.pdf")
        with patch.object(pipeline, "load_pdf", return_value=[MagicMock()]), \
             patch.object(pipeline, "_chunk", return_value=[MagicMock()]), \
             patch.object(pipeline, "_index", return_value=MagicMock()), \
             patch.object(pipeline, "_build_retriever", return_value=MagicMock()), \
             patch("rag.pipeline.RetrieverQueryEngine"):
            pipeline.build(fake_pdf)
        assert pipeline._docs_classified is False

    def test_docs_classified_true_when_flag_set(self, pipeline, tmp_path):
        fake_pdf = str(tmp_path / "doc.pdf")
        with patch.object(pipeline, "load_pdf", return_value=[MagicMock()]), \
             patch.object(pipeline, "_annotate_pharma_doc_types", return_value=[MagicMock()]) as mock_ann, \
             patch.object(pipeline, "_chunk", return_value=[MagicMock()]), \
             patch.object(pipeline, "_index", return_value=MagicMock()), \
             patch.object(pipeline, "_build_retriever", return_value=MagicMock()), \
             patch("rag.pipeline.RetrieverQueryEngine"):
            pipeline.build(fake_pdf, classify_docs=True)
        assert pipeline._docs_classified is True
        mock_ann.assert_called_once()

    def test_annotate_not_called_when_classify_docs_false(self, pipeline, tmp_path):
        fake_pdf = str(tmp_path / "doc.pdf")
        with patch.object(pipeline, "load_pdf", return_value=[MagicMock()]), \
             patch.object(pipeline, "_annotate_pharma_doc_types") as mock_ann, \
             patch.object(pipeline, "_chunk", return_value=[MagicMock()]), \
             patch.object(pipeline, "_index", return_value=MagicMock()), \
             patch.object(pipeline, "_build_retriever", return_value=MagicMock()), \
             patch("rag.pipeline.RetrieverQueryEngine"):
            pipeline.build(fake_pdf, classify_docs=False)
        mock_ann.assert_not_called()

    def test_annotate_receives_load_pdf_output(self, pipeline, tmp_path):
        fake_pdf = str(tmp_path / "doc.pdf")
        mock_docs = [MagicMock(), MagicMock()]
        annotated = [MagicMock(), MagicMock()]
        with patch.object(pipeline, "load_pdf", return_value=mock_docs), \
             patch.object(pipeline, "_annotate_pharma_doc_types", return_value=annotated) as mock_ann, \
             patch.object(pipeline, "_chunk", return_value=[MagicMock()]) as mock_chunk, \
             patch.object(pipeline, "_index", return_value=MagicMock()), \
             patch.object(pipeline, "_build_retriever", return_value=MagicMock()), \
             patch("rag.pipeline.RetrieverQueryEngine"):
            pipeline.build(fake_pdf, classify_docs=True)
        mock_ann.assert_called_once_with(mock_docs)
        mock_chunk.assert_called_once_with(annotated)

    def test_stage_order_with_classification(self, pipeline, tmp_path):
        fake_pdf = str(tmp_path / "doc.pdf")
        call_order = []
        with patch.object(pipeline, "load_pdf", side_effect=lambda p: call_order.append("load") or [MagicMock()]), \
             patch.object(pipeline, "_annotate_pharma_doc_types", side_effect=lambda d: call_order.append("annotate") or d), \
             patch.object(pipeline, "_chunk", side_effect=lambda d: call_order.append("chunk") or [MagicMock()]), \
             patch.object(pipeline, "_index", side_effect=lambda c: call_order.append("index") or MagicMock()), \
             patch.object(pipeline, "_build_retriever", side_effect=lambda i, c: call_order.append("retrieve") or MagicMock()), \
             patch("rag.pipeline.RetrieverQueryEngine"):
            pipeline.build(fake_pdf, classify_docs=True)
        assert call_order == ["load", "annotate", "chunk", "index", "retrieve"]


# ---------------------------------------------------------------------------
# _classify_query
# ---------------------------------------------------------------------------

class TestClassifyQuery:

    def test_returns_valid_category(self, pipeline):
        pipeline.llm.complete.return_value = MagicMock(text="certificate_of_quality")
        assert pipeline._classify_query("What is the purity?") == "certificate_of_quality"

    def test_returns_unknown_for_unrecognized_response(self, pipeline):
        pipeline.llm.complete.return_value = MagicMock(text="garbage_value")
        assert pipeline._classify_query("test") == "unknown"

    def test_returns_unknown_for_empty_response(self, pipeline):
        pipeline.llm.complete.return_value = MagicMock(text="")
        assert pipeline._classify_query("test") == "unknown"

    def test_returns_unknown_for_whitespace_response(self, pipeline):
        pipeline.llm.complete.return_value = MagicMock(text="   \n  ")
        assert pipeline._classify_query("test") == "unknown"

    def test_normalizes_to_lowercase(self, pipeline):
        pipeline.llm.complete.return_value = MagicMock(text="COVER_LETTER")
        assert pipeline._classify_query("test") == "cover_letter"

    def test_takes_last_word_of_multiword_response(self, pipeline):
        """LLM sometimes adds preamble; only the last word should be used."""
        pipeline.llm.complete.return_value = MagicMock(text="The answer is: packaging_specification")
        assert pipeline._classify_query("test") == "packaging_specification"

    def test_prompt_contains_all_categories(self, pipeline):
        from rag.pipeline import _PHARMA_DOC_CATEGORIES
        pipeline.llm.complete.return_value = MagicMock(text="unknown")
        pipeline._classify_query("How is the material sourced?")
        prompt = pipeline.llm.complete.call_args[0][0]
        for cat in _PHARMA_DOC_CATEGORIES:
            assert cat in prompt

    def test_prompt_contains_query(self, pipeline):
        pipeline.llm.complete.return_value = MagicMock(text="unknown")
        pipeline._classify_query("What is the batch number?")
        prompt = pipeline.llm.complete.call_args[0][0]
        assert "What is the batch number?" in prompt

    def test_all_valid_categories_accepted(self, pipeline):
        from rag.pipeline import _PHARMA_DOC_CATEGORIES
        for cat in _PHARMA_DOC_CATEGORIES:
            pipeline.llm.complete.return_value = MagicMock(text=cat)
            assert pipeline._classify_query("q") == cat


# ---------------------------------------------------------------------------
# _classify_document
# ---------------------------------------------------------------------------

class TestClassifyDocument:

    def test_returns_valid_category(self, pipeline):
        pipeline.llm.complete.return_value = MagicMock(text="bse_tse_declaration")
        assert pipeline._classify_document("BSE/TSE declaration text") == "bse_tse_declaration"

    def test_returns_unknown_for_unrecognized_response(self, pipeline):
        pipeline.llm.complete.return_value = MagicMock(text="nonsense")
        assert pipeline._classify_document("some text") == "unknown"

    def test_returns_unknown_for_empty_response(self, pipeline):
        pipeline.llm.complete.return_value = MagicMock(text="")
        assert pipeline._classify_document("some text") == "unknown"

    def test_prompt_includes_first_600_chars_only(self, pipeline):
        pipeline.llm.complete.return_value = MagicMock(text="unknown")
        long_text = "X" * 2000
        pipeline._classify_document(long_text)
        prompt = pipeline.llm.complete.call_args[0][0]
        assert "X" * 600 in prompt
        assert "X" * 601 not in prompt

    def test_prompt_contains_all_categories(self, pipeline):
        from rag.pipeline import _PHARMA_DOC_CATEGORIES
        pipeline.llm.complete.return_value = MagicMock(text="unknown")
        pipeline._classify_document("page text")
        prompt = pipeline.llm.complete.call_args[0][0]
        for cat in _PHARMA_DOC_CATEGORIES:
            assert cat in prompt


# ---------------------------------------------------------------------------
# _annotate_pharma_doc_types
# ---------------------------------------------------------------------------

class TestAnnotatePharmaDocTypes:

    def _make_doc(self, text="sample text", page=1):
        from llama_index.core import Document
        return Document(text=text, metadata={"page_number": page, "file_name": "doc.pdf"})

    def test_sends_unmatched_pages_to_batch(self, pipeline):
        # Texts with no keyword matches should all go to _classify_documents_batch
        docs = [self._make_doc(text=f"page {i}") for i in range(3)]
        with patch.object(pipeline, "_classify_documents_batch", return_value=["cover_letter"] * 3) as mock_batch:
            pipeline._annotate_pharma_doc_types(docs)
        mock_batch.assert_called_once()
        assert mock_batch.call_args[0][0] == ["page 0", "page 1", "page 2"]

    def test_keyword_matched_pages_skip_batch(self, pipeline):
        # A page whose header contains a keyword should not go to the LLM
        docs = [self._make_doc(text="Certificate of Quality for batch 123")]
        with patch.object(pipeline, "_classify_documents_batch") as mock_batch:
            result = pipeline._annotate_pharma_doc_types(docs)
        mock_batch.assert_not_called()
        assert result[0].metadata["pharma_doc_type"] == "certificate_of_quality"

    def test_sets_pharma_doc_type_on_each_document(self, pipeline):
        docs = [self._make_doc(text=f"page {i}") for i in range(2)]
        with patch.object(pipeline, "_classify_documents_batch", return_value=["material_description", "material_description"]):
            result = pipeline._annotate_pharma_doc_types(docs)
        for doc in result:
            assert doc.metadata["pharma_doc_type"] == "material_description"

    def test_returns_same_list_object(self, pipeline):
        docs = [self._make_doc()]
        with patch.object(pipeline, "_classify_documents_batch", return_value=["unknown"]):
            result = pipeline._annotate_pharma_doc_types(docs)
        assert result is docs

    def test_preserves_existing_metadata(self, pipeline):
        docs = [self._make_doc(page=7)]
        with patch.object(pipeline, "_classify_documents_batch", return_value=["cover_letter"]):
            result = pipeline._annotate_pharma_doc_types(docs)
        assert result[0].metadata["page_number"] == 7
        assert result[0].metadata["file_name"] == "doc.pdf"

    def test_per_page_category_can_differ(self, pipeline):
        docs = [self._make_doc(text="p1"), self._make_doc(text="p2")]
        with patch.object(pipeline, "_classify_documents_batch", return_value=["cover_letter", "certificate_of_quality"]):
            result = pipeline._annotate_pharma_doc_types(docs)
        assert result[0].metadata["pharma_doc_type"] == "cover_letter"
        assert result[1].metadata["pharma_doc_type"] == "certificate_of_quality"

    def test_empty_list_returns_empty(self, pipeline):
        with patch.object(pipeline, "_classify_documents_batch") as mock_batch:
            result = pipeline._annotate_pharma_doc_types([])
        mock_batch.assert_not_called()
        assert result == []


# ---------------------------------------------------------------------------
# _build_filtered_engine
# ---------------------------------------------------------------------------

class TestBuildFilteredEngine:

    def test_raises_when_index_is_none(self, pipeline):
        pipeline._vector_index = None
        with pytest.raises(RuntimeError, match="Pipeline not built"):
            pipeline._build_filtered_engine("certificate_of_quality")

    def test_vector_retriever_receives_metadata_filter(self, pipeline):
        pipeline._vector_index = MagicMock(name="index")
        pipeline._chunks = [_make_chunk("certificate_of_quality")]

        with patch("rag.pipeline.VectorIndexRetriever") as mock_vr, \
             patch("rag.pipeline.BM25Retriever") as mock_bm25, \
             patch("rag.pipeline.QueryFusionRetriever") as mock_qfr, \
             patch("rag.pipeline.RetrieverQueryEngine") as mock_qe, \
             patch("rag.pipeline.MetadataFilters") as mock_filters_cls, \
             patch("rag.pipeline.MetadataFilter") as mock_filter_cls:
            pipeline._build_filtered_engine("certificate_of_quality")

        mock_filter_cls.assert_called_once_with(key="pharma_doc_type", value="certificate_of_quality")
        mock_filters_cls.assert_called_once()
        mock_vr.assert_called_once()

    def test_bm25_receives_only_matching_chunks(self, pipeline):
        pipeline._vector_index = MagicMock(name="index")
        pipeline._chunks = [
            _make_chunk("certificate_of_quality"),
            _make_chunk("cover_letter"),
            _make_chunk("certificate_of_quality"),
        ]

        with patch("rag.pipeline.VectorIndexRetriever"), \
             patch("rag.pipeline.BM25Retriever") as mock_bm25, \
             patch("rag.pipeline.QueryFusionRetriever"), \
             patch("rag.pipeline.RetrieverQueryEngine"), \
             patch("rag.pipeline.MetadataFilters"), \
             patch("rag.pipeline.MetadataFilter"):
            pipeline._build_filtered_engine("certificate_of_quality")

        called_nodes = mock_bm25.from_defaults.call_args.kwargs.get(
            "nodes", mock_bm25.from_defaults.call_args[1].get("nodes",
            mock_bm25.from_defaults.call_args[0][0] if mock_bm25.from_defaults.call_args[0] else None)
        )
        assert len(called_nodes) == 2
        for n in called_nodes:
            assert n.metadata["pharma_doc_type"] == "certificate_of_quality"

    def test_only_vector_retriever_when_no_matching_chunks(self, pipeline):
        pipeline._vector_index = MagicMock(name="index")
        pipeline._chunks = [_make_chunk("cover_letter")]  # no CoQ chunks

        with patch("rag.pipeline.VectorIndexRetriever") as mock_vr, \
             patch("rag.pipeline.BM25Retriever") as mock_bm25, \
             patch("rag.pipeline.QueryFusionRetriever") as mock_qfr, \
             patch("rag.pipeline.RetrieverQueryEngine"), \
             patch("rag.pipeline.MetadataFilters"), \
             patch("rag.pipeline.MetadataFilter"):
            pipeline._build_filtered_engine("certificate_of_quality")

        mock_bm25.from_defaults.assert_not_called()
        retrievers_arg = mock_qfr.call_args.kwargs.get(
            "retrievers", mock_qfr.call_args[1].get("retrievers",
            mock_qfr.call_args[0][0] if mock_qfr.call_args[0] else None)
        )
        assert len(retrievers_arg) == 1

    def test_returns_engine_from_query_engine_factory(self, pipeline):
        pipeline._vector_index = MagicMock(name="index")
        pipeline._chunks = [_make_chunk("cover_letter")]
        fake_engine = MagicMock(name="filtered_engine")

        with patch("rag.pipeline.VectorIndexRetriever"), \
             patch("rag.pipeline.BM25Retriever"), \
             patch("rag.pipeline.QueryFusionRetriever"), \
             patch("rag.pipeline.RetrieverQueryEngine") as mock_qe, \
             patch("rag.pipeline.MetadataFilters"), \
             patch("rag.pipeline.MetadataFilter"):
            mock_qe.from_args.return_value = fake_engine
            result = pipeline._build_filtered_engine("cover_letter")

        assert result is fake_engine


# ---------------------------------------------------------------------------
# query – classify flag
# ---------------------------------------------------------------------------

class TestQueryClassify:

    def _attach_engine(self, pipeline, answer="Answer."):
        mock_engine = MagicMock()
        mock_response = MagicMock()
        mock_response.__str__ = MagicMock(return_value=answer)
        mock_engine.query.return_value = mock_response
        pipeline._query_engine = mock_engine
        return mock_engine

    def test_classify_false_uses_default_engine(self, pipeline):
        engine = self._attach_engine(pipeline)
        with patch.object(pipeline, "_classify_query") as mock_cls, \
             patch.object(pipeline, "_build_filtered_engine") as mock_fe:
            pipeline.query("What is the purity?", classify=False)
        mock_cls.assert_not_called()
        mock_fe.assert_not_called()
        engine.query.assert_called_once()

    def test_classify_true_without_docs_classified_uses_default_engine(self, pipeline):
        """When classify=True but docs were not classified at build time,
        query() skips both classification and filtering (no category to filter on)
        and falls back to the default engine."""
        engine = self._attach_engine(pipeline)
        pipeline._docs_classified = False
        with patch.object(pipeline, "_classify_query") as mock_cls, \
             patch.object(pipeline, "_build_filtered_engine") as mock_fe:
            pipeline.query("test", classify=True)
        mock_cls.assert_not_called()
        mock_fe.assert_not_called()
        engine.query.assert_called_once()

    def test_classify_true_with_docs_classified_uses_filtered_engine(self, pipeline):
        self._attach_engine(pipeline)
        pipeline._docs_classified = True
        pipeline._vector_index = MagicMock()
        filtered_engine = MagicMock()
        filtered_response = MagicMock()
        filtered_response.__str__ = MagicMock(return_value="filtered answer")
        filtered_engine.query.return_value = filtered_response

        with patch.object(pipeline, "_classify_query", return_value="certificate_of_quality"), \
             patch.object(pipeline, "_build_filtered_engine", return_value=filtered_engine) as mock_fe:
            result = pipeline.query("What is the purity?", classify=True)

        mock_fe.assert_called_once_with("certificate_of_quality")
        assert result == "filtered answer"

    def test_classify_unknown_category_uses_default_engine(self, pipeline):
        engine = self._attach_engine(pipeline)
        pipeline._docs_classified = True
        pipeline._vector_index = MagicMock()
        with patch.object(pipeline, "_classify_query", return_value="unknown"), \
             patch.object(pipeline, "_build_filtered_engine") as mock_fe:
            pipeline.query("test", classify=True)
        mock_fe.assert_not_called()
        engine.query.assert_called_once()


# ---------------------------------------------------------------------------
# query_with_sources – classify flag
# ---------------------------------------------------------------------------

class TestQueryWithSourcesClassify:

    def _attach_engine(self, pipeline, source_nodes=None, answer="Answer."):
        mock_engine = MagicMock()
        mock_response = MagicMock()
        mock_response.__str__ = MagicMock(return_value=answer)
        mock_response.source_nodes = source_nodes or []
        mock_engine.query.return_value = mock_response
        pipeline._query_engine = mock_engine
        return mock_engine

    def test_query_category_none_when_classify_false(self, pipeline):
        self._attach_engine(pipeline, [_make_source_node()])
        result = pipeline.query_with_sources("test?", classify=False)
        assert result["query_category"] is None

    def test_query_category_returned_when_classify_true(self, pipeline):
        self._attach_engine(pipeline)
        pipeline._docs_classified = False
        with patch.object(pipeline, "_classify_query", return_value="packaging_specification"):
            result = pipeline.query_with_sources("test?", classify=True)
        assert result["query_category"] == "packaging_specification"

    def test_filtered_engine_used_when_docs_classified(self, pipeline):
        self._attach_engine(pipeline)
        pipeline._docs_classified = True
        pipeline._vector_index = MagicMock()
        filtered_engine = MagicMock()
        filtered_response = MagicMock()
        filtered_response.__str__ = MagicMock(return_value="filtered")
        filtered_response.source_nodes = []
        filtered_engine.query.return_value = filtered_response

        with patch.object(pipeline, "_classify_query", return_value="cover_letter"), \
             patch.object(pipeline, "_build_filtered_engine", return_value=filtered_engine) as mock_fe:
            result = pipeline.query_with_sources("test?", classify=True)

        mock_fe.assert_called_once_with("cover_letter")
        assert result["query_category"] == "cover_letter"

    def test_default_engine_used_when_docs_not_classified(self, pipeline):
        engine = self._attach_engine(pipeline)
        pipeline._docs_classified = False
        with patch.object(pipeline, "_classify_query", return_value="cover_letter"), \
             patch.object(pipeline, "_build_filtered_engine") as mock_fe:
            pipeline.query_with_sources("test?", classify=True)
        mock_fe.assert_not_called()
        engine.query.assert_called_once()

    def test_unknown_category_uses_default_engine(self, pipeline):
        engine = self._attach_engine(pipeline)
        pipeline._docs_classified = True
        pipeline._vector_index = MagicMock()
        with patch.object(pipeline, "_classify_query", return_value="unknown"), \
             patch.object(pipeline, "_build_filtered_engine") as mock_fe:
            pipeline.query_with_sources("test?", classify=True)
        mock_fe.assert_not_called()
        engine.query.assert_called_once()

    def test_sources_include_pharma_doc_type(self, pipeline):
        node = _make_source_node(score=1.0, pharma_doc_type="chain_of_custody")
        self._attach_engine(pipeline, [node])
        result = pipeline.query_with_sources("test?")
        assert result["sources"][0]["pharma_doc_type"] == "chain_of_custody"

    def test_pharma_doc_type_defaults_to_unknown_when_missing(self, pipeline):
        node = MagicMock()
        node.score = 1.0
        node.node.metadata = {}
        node.node.text = "text"
        self._attach_engine(pipeline, [node])
        result = pipeline.query_with_sources("test?")
        assert result["sources"][0]["pharma_doc_type"] == "unknown"


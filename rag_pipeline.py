"""
RAG Pipeline
============
Reusable RAG pipeline for document question-answering over PDFs.

Architecture:
    PDF → Load (PyMuPDF + OCR fallback) → Semantic Chunk → Embed & Index (FAISS)
    → Hybrid Retrieve (Vector + BM25, reciprocal rerank) → Prompt → Local LLM → Answer

Embedding Model : sentence-transformers/all-MiniLM-L6-v2
Chunking        : Semantic chunking (LlamaIndex SemanticSplitterNodeParser)
Retrieval       : Hybrid – vector (FAISS) + BM25, fused via reciprocal rerank
LLM             : Mistral GGUF (local, via llama-cpp-python)
"""

import logging
import os
from typing import Any, Dict, List, Optional

import fitz  # PyMuPDF
from dotenv import load_dotenv
from llama_index.core import Document, Settings, VectorStoreIndex
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core.prompts import PromptTemplate
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import QueryFusionRetriever, VectorIndexRetriever
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.retrievers.bm25 import BM25Retriever

try:
    import pytesseract
    from PIL import Image as PILImage
    _OCR_AVAILABLE = True
except ImportError:
    _OCR_AVAILABLE = False

load_dotenv()

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper())
logger = logging.getLogger(__name__)

_PHARMA_QA_PROMPT = PromptTemplate(
    "You are a pharmaceutical document assistant. "
    "Answer the question based ONLY on the context provided below. "
    "After your answer, cite the specific source document name and page number(s) you referenced.\n\n"
    "Context:\n{context_str}\n\n"
    "Question: {query_str}\n\n"
    "Answer (with citations):"
)

# Scanned page detection threshold: pages with fewer characters than this
# are assumed to be image-based and will be re-processed with Tesseract OCR.
_SCANNED_PAGE_CHAR_THRESHOLD = 100

# DPI used when rasterising a PDF page for OCR. 300 dpi gives a good
# balance between OCR accuracy and memory usage.
_OCR_DPI = 300


class RAGPipeline:
    """End-to-end RAG pipeline that ingests a PDF and answers natural-language questions.

    The pipeline runs entirely locally — no external API calls are made.
    Call :meth:`build` once per document to index it, then use :meth:`query`
    or :meth:`query_with_sources` to retrieve answers.

    Example:
        >>> rag = RAGPipeline(model_path=r"C:\\LLM Models\\mistral.gguf")
        >>> rag.build("report.pdf")
        >>> answer = rag.query("What are the storage conditions?")
        >>> print(answer)
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        embed_model_name: Optional[str] = None,
        similarity_top_k: Optional[int] = None,
        num_queries: int = 1,
        n_gpu_layers: Optional[int] = None,
    ) -> None:
        """Initialise the RAG pipeline by loading the LLM and embedding model.

        Args:
            model_path: Absolute path to a local Mistral GGUF model file.
                Falls back to the ``MODEL_PATH`` environment variable if ``None``.
            embed_model_name: HuggingFace sentence-transformer model identifier
                used to produce dense embeddings for retrieval.
                Falls back to ``EMBED_MODEL`` env var, then ``all-MiniLM-L6-v2``.
            similarity_top_k: Number of top-ranked chunks returned by each
                individual retriever (vector and BM25) before fusion.
                Falls back to ``SIMILARITY_TOP_K`` env var, then ``5``.
            num_queries: Number of LLM-generated query variants used by
                :class:`QueryFusionRetriever`. Set to ``1`` to disable
                internal query expansion (recommended for speed).
            n_gpu_layers: Number of model layers offloaded to GPU.
                ``-1`` offloads all layers (requires CUDA). ``0`` runs on CPU only.
                Falls back to ``N_GPU_LAYERS`` env var, then ``-1``.

        Raises:
            FileNotFoundError: If the resolved ``model_path`` does not exist.
            ValueError: If no model path is provided and ``MODEL_PATH`` env var is unset.
        """
        _model_path: str = (
            model_path
            or os.getenv("MODEL_PATH")
            or r"C:\LLM Models\Mistral\mistral-7b-instruct-v0.2.Q4_K_M.gguf"
        )
        _embed_model: str = (
            embed_model_name
            or os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        )
        _top_k: int = (
            similarity_top_k
            if similarity_top_k is not None
            else int(os.getenv("SIMILARITY_TOP_K", "5"))
        )
        _gpu_layers: int = (
            n_gpu_layers
            if n_gpu_layers is not None
            else int(os.getenv("N_GPU_LAYERS", "-1"))
        )

        if not os.path.exists(_model_path):
            raise FileNotFoundError(f"GGUF model not found: {_model_path}")

        self.similarity_top_k: int = _top_k
        self.num_queries: int = num_queries

        self.llm: LlamaCPP = LlamaCPP(
            model_path=_model_path,
            temperature=0.1,
            max_new_tokens=1024,
            context_window=32768,
            model_kwargs={"n_gpu_layers": _gpu_layers},
            verbose=False,
        )

        self.embed_model: HuggingFaceEmbedding = HuggingFaceEmbedding(
            model_name=_embed_model
        )

        Settings.embed_model = self.embed_model
        Settings.llm = self.llm

        self._splitter: SemanticSplitterNodeParser = SemanticSplitterNodeParser(
            embed_model=self.embed_model
        )

        self._chunks: List[Any] = []
        self._vector_index: Optional[VectorStoreIndex] = None
        self._query_engine: Optional[RetrieverQueryEngine] = None
        self._pdf_path: Optional[str] = None

    # ------------------------------------------------------------------
    # PDF Loading
    # ------------------------------------------------------------------

    def _ocr_page(self, page: Any) -> str:
        """Rasterise a PDF page and extract text via Tesseract OCR.

        The page is rendered at :data:`_OCR_DPI` dpi to balance accuracy
        against memory usage.

        Args:
            page: A :class:`fitz.Page` object from an open PDF document.

        Returns:
            Raw text extracted from the page image by Tesseract.
        """
        pix = page.get_pixmap(dpi=_OCR_DPI)
        img = PILImage.frombytes("RGB", [pix.width, pix.height], pix.samples)
        return pytesseract.image_to_string(img)

    def load_pdf(self, pdf_path: str) -> List[Document]:
        """Extract text from every page of a PDF, with OCR fallback for scanned pages.

        Pages with fewer than :data:`_SCANNED_PAGE_CHAR_THRESHOLD` characters of
        extracted text are treated as scanned/image-based and re-processed with
        Tesseract OCR (when available). Pages that yield no text after both
        methods are skipped entirely.

        Args:
            pdf_path: Absolute or relative path to the PDF file.

        Returns:
            A list of :class:`~llama_index.core.Document` objects, one per
            non-empty page, each carrying the following metadata keys:

            - ``file_name`` – base name of the source file.
            - ``page_number`` – 1-based page index.
            - ``total_pages`` – total page count in the document.
            - ``doc_type`` – ``"scanned"`` if OCR was used, else ``"digital"``.
            - ``ocr_used`` – boolean flag.
            - ``source_id`` – unique identifier in the form ``"<file>:p<page>"``.
        """
        documents: List[Document] = []
        file_name = os.path.basename(pdf_path)

        with fitz.open(pdf_path) as doc:
            for i, page in enumerate(doc):
                text = page.get_text()
                ocr_used = False

                if len(text.strip()) < _SCANNED_PAGE_CHAR_THRESHOLD and _OCR_AVAILABLE:
                    text = self._ocr_page(page)
                    ocr_used = True

                if not text.strip():
                    continue

                documents.append(
                    Document(
                        text=text,
                        metadata={
                            "file_name": file_name,
                            "page_number": i + 1,
                            "total_pages": len(doc),
                            "doc_type": "scanned" if ocr_used else "digital",
                            "ocr_used": ocr_used,
                            "source_id": f"{file_name}:p{i + 1}",
                        },
                    )
                )

        logger.info("Loaded '%s': %d pages with content.", file_name, len(documents))
        return documents

    # ------------------------------------------------------------------
    # Chunking
    # ------------------------------------------------------------------

    def _chunk(self, documents: List[Document]) -> List[Any]:
        """Split documents into semantically coherent chunks.

        Uses :class:`~llama_index.core.node_parser.SemanticSplitterNodeParser`
        which groups sentences by embedding similarity rather than by a fixed
        character/token count.

        Args:
            documents: List of :class:`~llama_index.core.Document` objects
                produced by :meth:`load_pdf`.

        Returns:
            A flat list of :class:`~llama_index.core.schema.BaseNode` chunks
            ready for indexing.

        Raises:
            ValueError: If ``documents`` is empty.
        """
        if not documents:
            raise ValueError("No documents provided for chunking.")
        logger.info("Performing semantic chunking...")
        chunks = self._splitter.get_nodes_from_documents(documents)
        logger.info("Total chunks created: %d", len(chunks))
        return chunks

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def _index(self, chunks: List[Any]) -> VectorStoreIndex:
        """Build an in-memory FAISS vector index from a list of chunk nodes.

        Args:
            chunks: List of :class:`~llama_index.core.schema.BaseNode` objects
                produced by :meth:`_chunk`.

        Returns:
            A :class:`~llama_index.core.VectorStoreIndex` backed by an
            in-memory FAISS store.
        """
        vector_index = VectorStoreIndex.from_documents(chunks)
        logger.info("Indexed %d chunks.", len(chunks))
        return vector_index

    # ------------------------------------------------------------------
    # Retriever
    # ------------------------------------------------------------------

    def _build_retriever(
        self,
        vector_index: VectorStoreIndex,
        chunks: List[Any],
    ) -> QueryFusionRetriever:
        """Construct a hybrid retriever combining vector search and BM25.

        Results from both retrievers are merged using Reciprocal Rank Fusion
        (RRF), which re-ranks results by combining each item's reciprocal
        rank from each retriever without relying on raw score magnitudes.

        Args:
            vector_index: A :class:`~llama_index.core.VectorStoreIndex` built
                from the indexed chunks.
            chunks: The same chunk nodes used to build the index, required
                to initialise the :class:`~llama_index.retrievers.bm25.BM25Retriever`.

        Returns:
            A :class:`~llama_index.core.retrievers.QueryFusionRetriever`
            configured to return the top ``similarity_top_k`` fused results.
        """
        vector_retriever = VectorIndexRetriever(
            index=vector_index, similarity_top_k=self.similarity_top_k
        )
        bm25_retriever = BM25Retriever.from_defaults(
            nodes=chunks, similarity_top_k=self.similarity_top_k
        )
        hybrid_retriever = QueryFusionRetriever(
            retrievers=[vector_retriever, bm25_retriever],
            similarity_top_k=self.similarity_top_k,
            num_queries=self.num_queries,
            mode="reciprocal_rerank",
            use_async=False,
            llm=self.llm,
        )
        logger.info("Hybrid retriever ready.")
        return hybrid_retriever

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(self, pdf_path: str) -> None:
        """Index a PDF document and prepare the query engine.

        This method runs the full ingestion pipeline:
        load → chunk → embed & index → build hybrid retriever → attach query engine.

        Call this once per document. Calling it again replaces the current index.

        Args:
            pdf_path: Path to the PDF file to ingest.
        """
        self._pdf_path = pdf_path
        documents = self.load_pdf(pdf_path)
        self._chunks = self._chunk(documents)
        self._vector_index = self._index(self._chunks)
        hybrid_retriever = self._build_retriever(self._vector_index, self._chunks)

        self._query_engine = RetrieverQueryEngine.from_args(
            retriever=hybrid_retriever,
            llm=self.llm,
            text_qa_template=_PHARMA_QA_PROMPT,
        )
        logger.info("RAG pipeline ready.")

    def expand_query(self, query: str, num_expansions: int = 3) -> List[str]:
        """Generate alternative phrasings of a query using the LLM.

        Query expansion can improve recall by rephrasing the question with
        different but semantically related terminology before retrieval.

        Args:
            query: The original natural-language query.
            num_expansions: Number of alternative phrasings to generate.

        Returns:
            A list with the original query as the first element, followed
            by up to ``num_expansions`` LLM-generated alternatives.
        """
        prompt = (
            f'I need to search a document with this query: "{query}"\n\n'
            f"Please generate {num_expansions} alternative versions that:\n"
            "1. Use different but related terminology\n"
            "2. Include relevant pharmaceutical/quality terms that might appear in a certificate or SDF\n"
            "3. Cover similar concepts but phrased differently\n\n"
            "Format your response as a list of alternative queries only, with no additional text."
        )
        response = self.llm.complete(prompt)
        expansions = [line.strip() for line in response.text.split("\n") if line.strip()]
        if query not in expansions:
            expansions = [query] + expansions
        return expansions

    def query(
        self,
        question: str,
        expand: bool = False,
        num_expansions: int = 3,
    ) -> str:
        """Query the pipeline and return a plain-text answer.

        Args:
            question: Natural-language question to answer.
            expand: If ``True``, use :meth:`expand_query` to generate
                alternative phrasings before retrieval. Increases latency.
            num_expansions: Number of query expansions to generate when
                ``expand=True``.

        Returns:
            The LLM's answer as a plain string, including inline citations
            to the source document and page number(s).

        Raises:
            RuntimeError: If :meth:`build` has not been called yet.
        """
        if self._query_engine is None:
            raise RuntimeError("Pipeline not built. Call build(pdf_path) first.")

        search_query = question
        if expand:
            queries = self.expand_query(question, num_expansions=num_expansions)
            search_query = queries[1] if len(queries) > 1 else question

        response = self._query_engine.query(search_query)
        return str(response)

    def query_with_sources(
        self,
        question: str,
        expand: bool = False,
        num_expansions: int = 3,
    ) -> Dict[str, Any]:
        """Query the pipeline and return a structured result with sources and confidence.

        Confidence scores are derived from the RRF retrieval scores normalised
        to a 0–100 % scale relative to the top-ranked chunk. When all raw scores
        are zero or unavailable (common with :class:`QueryFusionRetriever`),
        a rank-based fallback is used: ``100 / rank`` (1st = 100 %, 2nd = 50 %, …).

        Args:
            question: Natural-language question to answer.
            expand: If ``True``, expand the query before retrieval. See
                :meth:`expand_query` for details.
            num_expansions: Number of query expansions when ``expand=True``.

        Returns:
            A dictionary with the following keys:

            - ``answer`` (*str*) – LLM-generated answer with inline citations.
            - ``sources`` (*list[dict]*) – Retrieved chunks, each containing:

              - ``text`` – Raw chunk text (truncated in the UI, full here).
              - ``file`` – Source filename.
              - ``page`` – 1-based page number.
              - ``score`` – Confidence percentage (0–100), relative to top chunk.
              - ``doc_type`` – ``"digital"`` or ``"scanned"``.

            - ``chunk_count`` (*int*) – Number of chunks retrieved.

        Raises:
            RuntimeError: If :meth:`build` has not been called yet.
        """
        if self._query_engine is None:
            raise RuntimeError("Pipeline not built. Call build(pdf_path) first.")

        search_query = question
        if expand:
            queries = self.expand_query(question, num_expansions=num_expansions)
            search_query = queries[1] if len(queries) > 1 else question

        response = self._query_engine.query(search_query)

        raw_scores = [n.score for n in response.source_nodes if n.score is not None]
        max_score = max(raw_scores) if raw_scores else 0.0

        sources: List[Dict[str, Any]] = []
        for rank, node in enumerate(response.source_nodes):
            meta = node.node.metadata
            if max_score > 0 and node.score is not None:
                confidence = round((node.score / max_score) * 100, 1)
            else:
                confidence = round(100.0 / (rank + 1), 1)
            sources.append({
                "text": node.node.text,
                "file": meta.get("file_name", "unknown"),
                "page": meta.get("page_number", "?"),
                "score": confidence,
                "doc_type": meta.get("doc_type", "digital"),
            })

        return {
            "answer": str(response),
            "sources": sources,
            "chunk_count": len(sources),
        }

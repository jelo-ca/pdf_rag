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

# pylint: disable=too-many-lines

import logging
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

import fitz  # PyMuPDF
import torch
from dotenv import load_dotenv
from llama_index.core import Document, Settings, StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core.prompts import PromptTemplate
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import QueryFusionRetriever, VectorIndexRetriever
from llama_index.core.vector_stores.types import MetadataFilter, MetadataFilters
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.retrievers.bm25 import BM25Retriever

try:
    import pytesseract
    from PIL import Image as PILImage

    _OCR_AVAILABLE = True
except ImportError:
    _OCR_AVAILABLE = False


def _resolve_tesseract_cmd() -> Optional[str]:
    """Resolve a runnable Tesseract executable path.

    Returns:
        Absolute path to the executable when one can be found, otherwise ``None``.
    """
    resolved_cmd: Optional[str] = None
    if _OCR_AVAILABLE:
        # Respect explicit override if user already provided one.
        configured_cmd = str(getattr(pytesseract.pytesseract, "tesseract_cmd", "") or "").strip()
        if configured_cmd:
            if os.path.isabs(configured_cmd) and os.path.exists(configured_cmd):
                resolved_cmd = configured_cmd
            else:
                found_cmd = shutil.which(configured_cmd)
                if found_cmd:
                    resolved_cmd = found_cmd

        # Standard PATH lookup.
        if not resolved_cmd:
            found_on_path = shutil.which("tesseract")
            if found_on_path:
                resolved_cmd = found_on_path

        if not resolved_cmd and os.name == "nt":
            # Common Windows install locations.
            candidates = [
                os.path.join(os.environ.get("ProgramFiles", r"C:\Program Files"), "Tesseract-OCR", "tesseract.exe"),
                os.path.join(
                    os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"),
                    "Tesseract-OCR",
                    "tesseract.exe",
                ),
                os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs", "Tesseract-OCR", "tesseract.exe"),
            ]

            for candidate in candidates:
                if candidate and os.path.exists(candidate):
                    resolved_cmd = candidate
                    break

    return resolved_cmd


def _is_ocr_runtime_available() -> bool:
    """Return True only when pytesseract and a Tesseract binary are usable."""
    if not _OCR_AVAILABLE:
        return False

    resolved_cmd = _resolve_tesseract_cmd()
    if not resolved_cmd:
        return False

    pytesseract.pytesseract.tesseract_cmd = resolved_cmd

    # Populate TESSDATA_PREFIX for common Windows installs when unset.
    tessdata_dir = os.path.join(os.path.dirname(resolved_cmd), "tessdata")
    if os.name == "nt" and os.path.isdir(tessdata_dir) and not os.environ.get("TESSDATA_PREFIX"):
        os.environ["TESSDATA_PREFIX"] = tessdata_dir

    return True

load_dotenv()

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper())
logger = logging.getLogger(__name__)

_PHARMA_QA_PROMPT = PromptTemplate(
    "You are a pharmaceutical document assistant. "
    "Answer the question based ONLY on the context provided below. "
    "Keep the answer short and direct (1-3 sentences unless a list is explicitly requested). "
    "Do not explain your reasoning, retrieval process, or broader context. "
    "Do not include citations in the answer text.\n\n"
    "Context:\n{context_str}\n\n"
    "Question: {query_str}\n\n"
    "Answer:"
)

# Scanned page detection threshold: pages with fewer characters than this
# are assumed to be image-based and will be re-processed with Tesseract OCR.
_SCANNED_PAGE_CHAR_THRESHOLD = 100

# Recognised pharmaceutical document categories used by query and document
# classification.  Values are snake_case strings stored in the
# ``pharma_doc_type`` metadata field on every chunk.
_PHARMA_DOC_CATEGORIES: List[str] = [
    "cover_letter",
    "certificate_of_quality",
    "packaging_specification",
    "bse_tse_declaration",
    "material_description",
    "supplier_qualification",
    "chain_of_custody",
    "unknown",
]

# Keyword signals for fast pre-classification before any LLM call.
# Keys must be valid entries in _PHARMA_DOC_CATEGORIES.
_KEYWORD_MAP: Dict[str, List[str]] = {
    "cover_letter": [
        "cover letter",
        "dear sir",
        "dear madam",
        "dear supplier",
        "please find enclosed",
        "we herewith",
        "herewith enclosed",
    ],
    "certificate_of_quality": [
        "certificate of quality",
        "certificate of analysis",
        "cert. of quality",
        "coa ",
        "c.o.a",
    ],
    "packaging_specification": [
        "packaging specification",
        "packaging spec",
        "pack spec",
        "label specification",
        "labelling specification",
    ],
    "bse_tse_declaration": [
        "bse",
        "tse",
        "transmissible spongiform",
        "bovine spongiform",
        "spongiform encephalopathy",
    ],
    "material_description": [
        "material description",
        "material data sheet",
        "product description",
        "substance description",
        "raw material description",
    ],
    "supplier_qualification": [
        "supplier qualification",
        "vendor qualification",
        "approved supplier",
        "audit report",
        "supplier audit",
    ],
    "chain_of_custody": [
        "chain of custody",
        "chain-of-custody",
        "custody transfer",
    ],
}

# One labelled example per category shown to the LLM when keyword matching
# fails.  Short excerpts are enough — the goal is to anchor the format.
_FEW_SHOT_EXAMPLES: str = (
    "cover_letter\n"
    'Example: "Dear Supplier, Please find enclosed the updated documentation '
    'for batch 2024-001."\n\n'
    "certificate_of_quality\n"
    'Example: "Certificate of Quality — Batch No: 12345 — Product: '
    'Excipient X — Conforms to specification."\n\n'
    "packaging_specification\n"
    'Example: "Packaging Specification Rev. 3 — Primary container: '
    'HDPE bottle 250 mL — Closure torque: 15–20 Nm."\n\n'
    "bse_tse_declaration\n"
    'Example: "BSE/TSE Declaration — We confirm that no materials of bovine or ovine origin are used."\n\n'
    "material_description\n"
    'Example: "Material Description — Chemical name: Microcrystalline '
    'Cellulose — CAS: 9004-34-6 — Function: Filler."\n\n'
    "supplier_qualification\n"
    'Example: "Supplier Qualification Report — Audit date: 2023-05 — Site: Plant A — Status: Approved."\n\n'
    "chain_of_custody\n"
    'Example: "Chain of Custody — Transferred from Manufacturer X to Distributor Y on 2024-03-01."\n\n'
)

# DPI used when rasterising a PDF page for OCR. 200 dpi gives a good
# balance between OCR accuracy, memory usage, and speed.
_OCR_DPI = 200

# Max number of page-text characters included in LLM doc-classification prompt.
_DOC_CLASSIFY_PROMPT_CHARS = 600


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
        persist_dir: Optional[str] = None,
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
                Falls back to ``SIMILARITY_TOP_K`` env var, then ``3``.
            num_queries: Number of LLM-generated query variants used by
                :class:`QueryFusionRetriever`. Set to ``1`` to disable
                internal query expansion (recommended for speed).
            n_gpu_layers: Number of model layers offloaded to GPU.
                ``-1`` offloads all layers (requires CUDA). ``0`` runs on CPU only.
                Falls back to ``N_GPU_LAYERS`` env var, then ``-1``.
            persist_dir: Directory path for persisting the vector index.
                If provided, the index will be saved after building and loaded
                on subsequent runs, eliminating rebuild time.

        Raises:
            FileNotFoundError: If the resolved ``model_path`` does not exist.
            ValueError: If no model path is provided and ``MODEL_PATH`` env var is unset.
        """
        _model_path: str = (
            model_path or os.getenv("MODEL_PATH") or r"C:\LLM Models\Mistral\mistral-7b-instruct-v0.2.Q4_K_M.gguf"
        )
        _embed_model: str = embed_model_name or os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        _top_k: int = similarity_top_k if similarity_top_k is not None else int(os.getenv("SIMILARITY_TOP_K", "5"))
        _gpu_layers: int = n_gpu_layers if n_gpu_layers is not None else int(os.getenv("N_GPU_LAYERS", "-1"))

        if not os.path.exists(_model_path):
            raise FileNotFoundError(f"GGUF model not found: {_model_path}")

        self.similarity_top_k: int = _top_k
        self.num_queries: int = num_queries

        self.llm: LlamaCPP = LlamaCPP(
            model_path=_model_path,
            temperature=0.1,
            max_new_tokens=512,
            context_window=4096,
            model_kwargs={"n_gpu_layers": _gpu_layers},
            verbose=False,
        )

        self.embed_model: HuggingFaceEmbedding = HuggingFaceEmbedding(
            model_name=_embed_model,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

        Settings.embed_model = self.embed_model
        Settings.llm = self.llm

        self._splitter: SemanticSplitterNodeParser = SemanticSplitterNodeParser(
            embed_model=self.embed_model,
        )

        self.persist_dir: Optional[str] = persist_dir

        self._chunks: List[Any] = []
        self._vector_index: Optional[VectorStoreIndex] = None
        self._query_engine: Optional[RetrieverQueryEngine] = None
        self._pdf_path: Optional[str] = None
        self._docs_classified: bool = False

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
        Tesseract OCR (when available) in parallel. Pages that yield no text after
        both methods are skipped entirely.

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
        ocr_enabled = _is_ocr_runtime_available()

        if _OCR_AVAILABLE and not ocr_enabled:
            logger.warning(
                "pytesseract is installed but no Tesseract binary was found. "
                "Install Tesseract OCR or add it to PATH to enable scanned-page OCR."
            )

        with fitz.open(pdf_path) as doc:
            # First pass: extract text and identify pages needing OCR
            page_data = []
            for i, page in enumerate(doc):
                text = page.get_text()
                page_data.append((i, page, text))

            # Parallel OCR processing for scanned pages
            if ocr_enabled:
                pages_needing_ocr = [
                    (i, page) for i, page, text in page_data if len(text.strip()) < _SCANNED_PAGE_CHAR_THRESHOLD
                ]

                if pages_needing_ocr:
                    with ThreadPoolExecutor(max_workers=4) as executor:
                        ocr_results = list(executor.map(lambda p: (p[0], self._ocr_page(p[1])), pages_needing_ocr))
                    ocr_dict = dict(ocr_results)
                else:
                    ocr_dict = {}
            else:
                ocr_dict = {}

            # Build documents
            for i, page, text in page_data:
                ocr_used = False
                if i in ocr_dict:
                    text = ocr_dict[i]
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
        """Split documents into fixed-size chunks with overlap.

        Uses :class:`~llama_index.core.node_parser.SemanticSplitterNodeParser`
        which groups sentences into chunks based on embedding similarity,
        so chunk boundaries align with semantic topic shifts.

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
        """Build a vector index from a list of chunk nodes with optional persistence.

        If persist_dir was provided during initialization and an index exists,
        it will be loaded. Otherwise, a new index is built.

        Args:
            chunks: List of :class:`~llama_index.core.schema.BaseNode` objects
                produced by :meth:`_chunk`.

        Returns:
            A :class:`~llama_index.core.VectorStoreIndex` backed by an
            in-memory FAISS store, optionally persisted to disk.
        """
        vector_index = VectorStoreIndex.from_documents(chunks)
        logger.info("Indexed %d chunks.", len(chunks))

        if self.persist_dir:
            vector_index.storage_context.persist(persist_dir=self.persist_dir)
            logger.info("Index persisted to %s", self.persist_dir)

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
        vector_retriever = VectorIndexRetriever(index=vector_index, similarity_top_k=self.similarity_top_k)
        bm25_retriever = BM25Retriever.from_defaults(nodes=chunks, similarity_top_k=self.similarity_top_k)
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
    # Document & Query Classification
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_category(response_text: str) -> str:
        """Extract the first valid pharma category from an LLM response string.

        Scans every whitespace-separated token (after stripping punctuation) for
        a match against :data:`_PHARMA_DOC_CATEGORIES`.  This is intentionally
        more permissive than a last-word check so that responses like
        ``"The type is cover_letter."`` or ``"certificate_of_quality\\n\\nThis…"``
        are still parsed correctly.

        Args:
            response_text: Raw text returned by the LLM.

        Returns:
            The first matching category string, or ``"unknown"`` if none found.
        """
        normalised = response_text.strip().lower()
        # Fast path: check for each category as a substring in declaration order
        for cat in _PHARMA_DOC_CATEGORIES:
            if cat != "unknown" and cat in normalised:
                return cat
        return "unknown"

    def _classify_query(self, query: str) -> str:
        """Classify a natural-language query into a pharma document category.

        Uses the LLM to determine which type of pharmaceutical document is most
        likely to contain the answer, enabling targeted retrieval.

        Args:
            query: The user's natural-language question.

        Returns:
            A snake_case category string from :data:`_PHARMA_DOC_CATEGORIES`.
            Falls back to ``"unknown"`` if the LLM response is ambiguous.
        """
        categories_str = ", ".join(_PHARMA_DOC_CATEGORIES)
        prompt = (
            "You are an expert pharmaceutical document classifier.\n"
            "Given a user query, identify which document type most likely contains the answer.\n"
            f"Choose exactly one from: {categories_str}.\n"
            "Respond with only the category name in snake_case. No extra text.\n\n"
            f"Query: {query}\n"
            "Category:"
        )
        response = self.llm.complete(prompt)
        result = self._parse_category(response.text)
        logger.info("Query classified as: %s", result)
        return result

    def _classify_document(self, text: str) -> str:
        """Classify a page of document text into a pharma document category.

        Uses a two-stage approach for efficiency:

        1. **Keyword scan** — checks the first 300 characters of the page
           against :data:`_KEYWORD_MAP`.  Returns immediately on a match
           (no LLM call).
        2. **Few-shot LLM** — if no keyword matched, asks the LLM with one
           labelled example per category to anchor the output format.

        Args:
            text: Raw text extracted from a single PDF page.

        Returns:
            A snake_case category string from :data:`_PHARMA_DOC_CATEGORIES`.
        """
        # Stage 1: keyword scan on the page header (fast, no LLM)
        header = text[:300].lower()
        for cat, keywords in _KEYWORD_MAP.items():
            if any(kw in header for kw in keywords):
                return cat

        # Stage 2: few-shot LLM classification for ambiguous pages
        snippet = text[:_DOC_CLASSIFY_PROMPT_CHARS]
        categories_str = ", ".join(_PHARMA_DOC_CATEGORIES)
        prompt = (
            "You are an expert pharmaceutical document classifier.\n"
            "Given a page of text, identify its document type.\n"
            f"Choose exactly one from: {categories_str}.\n"
            "Respond with only the category name in snake_case. No extra text.\n\n"
            "Examples:\n"
            f"{_FEW_SHOT_EXAMPLES}"
            f"Document text:\n{snippet}\n\n"
            "Category:"
        )
        response = self.llm.complete(prompt)
        return self._parse_category(response.text)

    def _annotate_pharma_doc_types(self, documents: List[Document]) -> List[Document]:
        """Add a ``pharma_doc_type`` metadata field to each document via LLM classification.

        Classifies each page individually so the LLM returns exactly one label
        per call, avoiding the alignment errors that occur when parsing a
        multi-label batch response.

        Args:
            documents: Pages loaded by :meth:`load_pdf`.

        Returns:
            The same list with ``pharma_doc_type`` set on every document's metadata.
        """
        logger.info("Classifying %d pages into pharma doc types...", len(documents))
        for doc in documents:
            doc_type = self._classify_document(doc.text)
            doc.metadata["pharma_doc_type"] = doc_type
            logger.debug("Page %d → %s", doc.metadata.get("page_number", "?"), doc_type)
        logger.info("Document classification complete.")
        return documents

    def _build_filtered_engine(self, pharma_doc_type: str) -> RetrieverQueryEngine:
        """Build a query engine whose retrieval is scoped to a single pharma doc type.

        Both the vector retriever (via :class:`MetadataFilters`) and the BM25
        retriever (via pre-filtered node list) are restricted to chunks whose
        ``pharma_doc_type`` metadata matches *pharma_doc_type*.

        Args:
            pharma_doc_type: A category string from :data:`_PHARMA_DOC_CATEGORIES`.

        Returns:
            A :class:`~llama_index.core.query_engine.RetrieverQueryEngine` scoped
            to the requested document category.

        Raises:
            RuntimeError: If :meth:`build` has not been called yet.
        """
        if self._vector_index is None:
            raise RuntimeError("Pipeline not built. Call build(pdf_path) first.")

        filters = MetadataFilters(filters=[MetadataFilter(key="pharma_doc_type", value=pharma_doc_type)])
        vector_retriever = VectorIndexRetriever(
            index=self._vector_index,
            similarity_top_k=self.similarity_top_k,
            filters=filters,
        )

        filtered_chunks = [c for c in self._chunks if c.metadata.get("pharma_doc_type") == pharma_doc_type]
        if filtered_chunks:
            bm25_retriever = BM25Retriever.from_defaults(
                nodes=filtered_chunks,
                similarity_top_k=min(self.similarity_top_k, len(filtered_chunks)),
            )
            retrievers: List[Any] = [vector_retriever, bm25_retriever]
        else:
            retrievers = [vector_retriever]

        hybrid_retriever = QueryFusionRetriever(
            retrievers=retrievers,
            similarity_top_k=self.similarity_top_k,
            num_queries=self.num_queries,
            mode="reciprocal_rerank",
            use_async=False,
            llm=self.llm,
        )
        return RetrieverQueryEngine.from_args(
            retriever=hybrid_retriever,
            llm=self.llm,
            text_qa_template=_PHARMA_QA_PROMPT,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(self, pdf_path: str, classify_docs: bool = False) -> None:
        """Index a PDF document and prepare the query engine.

        This method runs the full ingestion pipeline:
        load → (classify) → chunk → embed & index → build hybrid retriever → attach query engine.

        If persist_dir was provided during initialization, the method will attempt
        to load a previously saved index, skipping the indexing step.

        Call this once per document. Calling it again replaces the current index.

        Args:
            pdf_path: Path to the PDF file to ingest.
            classify_docs: When ``True``, each page is classified by the LLM and
                its ``pharma_doc_type`` is stored in chunk metadata.  This enables
                targeted retrieval via the ``classify`` parameter on :meth:`query`
                and :meth:`query_with_sources`.  Uses batched classification to
                reduce LLM calls.
        """
        self._pdf_path = pdf_path

        # Try to load existing index if persistence is enabled
        if self.persist_dir and os.path.exists(self.persist_dir):
            try:
                logger.info("Loading persisted index from %s...", self.persist_dir)
                storage_context = StorageContext.from_defaults(persist_dir=self.persist_dir)
                self._vector_index = load_index_from_storage(storage_context)
                self._chunks = list(self._vector_index.docstore.docs.values())
                loaded_classified = any("pharma_doc_type" in getattr(chunk, "metadata", {}) for chunk in self._chunks)
                # If the caller wants classification but the persisted index was
                # built without it, discard the cached index and rebuild.
                if classify_docs and not loaded_classified:
                    logger.info("Persisted index lacks classification data. Rebuilding with classification.")
                    self._vector_index = None
                else:
                    self._docs_classified = loaded_classified
                    logger.info("Loaded index with %d chunks.", len(self._chunks))
            except (FileNotFoundError, OSError, ValueError, RuntimeError) as e:
                logger.warning("Failed to load persisted index: %s. Building new index.", e)
                self._vector_index = None

        # Build new index if not loaded
        if self._vector_index is None:
            documents = self.load_pdf(pdf_path)
            if classify_docs:
                documents = self._annotate_pharma_doc_types(documents)
                before = len(documents)
                documents = [d for d in documents if d.metadata.get("pharma_doc_type") != "unknown"]
                dropped = before - len(documents)
                if dropped:
                    logger.info("Skipping %d page(s) classified as 'unknown'.", dropped)
            self._docs_classified = classify_docs
            self._chunks = self._chunk(documents)
            self._vector_index = self._index(self._chunks)

        hybrid_retriever = self._build_retriever(self._vector_index, self._chunks)

        self._query_engine = RetrieverQueryEngine.from_args(
            retriever=hybrid_retriever,
            llm=self.llm,
            text_qa_template=_PHARMA_QA_PROMPT,
        )
        logger.info("RAG pipeline ready.")

    def build_from_multiple_pdfs(
        self,
        pdf_paths: List[str],
        classify_docs: bool = False,
        progress_callback: Optional[callable] = None,
    ) -> None:
        """Index multiple PDF documents into a single unified index.

        This method processes multiple PDFs sequentially, loading and combining
        all pages before chunking and indexing. This creates a single searchable
        index across all documents.

        Args:
            pdf_paths: List of paths to PDF files to ingest.
            classify_docs: When ``True``, each page is classified by the LLM and
                its ``pharma_doc_type`` is stored in chunk metadata.
            progress_callback: Optional callback function called after each PDF
                is loaded. Receives (current_index, total_count, filename).

        Raises:
            ValueError: If pdf_paths is empty.
        """
        if not pdf_paths:
            raise ValueError("No PDF paths provided.")

        # Clear persistence dir since we're building from multiple sources
        if self.persist_dir and os.path.exists(self.persist_dir):
            logger.info("Clearing persisted index for multi-document build...")
            shutil.rmtree(self.persist_dir)

        all_documents: List[Document] = []

        for idx, pdf_path in enumerate(pdf_paths, 1):
            logger.info("Loading PDF %d/%d: %s", idx, len(pdf_paths), pdf_path)
            documents = self.load_pdf(pdf_path)
            all_documents.extend(documents)

            if progress_callback:
                progress_callback(idx, len(pdf_paths), os.path.basename(pdf_path))

        logger.info("Loaded %d total pages from %d PDFs.", len(all_documents), len(pdf_paths))

        # Classify all documents together if requested
        if classify_docs:
            all_documents = self._annotate_pharma_doc_types(all_documents)
            before = len(all_documents)
            all_documents = [d for d in all_documents if d.metadata.get("pharma_doc_type") != "unknown"]
            dropped = before - len(all_documents)
            if dropped:
                logger.info("Skipping %d page(s) classified as 'unknown'.", dropped)
        self._docs_classified = classify_docs

        # Chunk and index all documents
        self._chunks = self._chunk(all_documents)
        self._vector_index = self._index(self._chunks)

        # Build retriever and query engine
        hybrid_retriever = self._build_retriever(self._vector_index, self._chunks)
        self._query_engine = RetrieverQueryEngine.from_args(
            retriever=hybrid_retriever,
            llm=self.llm,
            text_qa_template=_PHARMA_QA_PROMPT,
        )

        # Store list of PDF paths for stats
        self._pdf_path = f"Multiple files ({len(pdf_paths)} PDFs)"

        logger.info("RAG pipeline ready with %d documents.", len(pdf_paths))

    def get_stats(self) -> Dict[str, Any]:
        """Return statistics about the currently indexed document(s).

        Returns:
            A dictionary with the following keys:

            - ``total_pages`` (*int*) – Total page count across all indexed PDFs.
            - ``total_chunks`` (*int*) – Number of chunks in the index.
            - ``total_files`` (*int*) – Number of unique files indexed.
            - ``file_names`` (*list*) – List of unique source filenames.
            - ``doc_type_counts`` (*dict*) – Mapping of ``pharma_doc_type`` →
              chunk count. Only populated when the index was built with
              ``classify_docs=True``; otherwise contains ``{"unclassified": N}``.
            - ``classified`` (*bool*) – Whether document classification was run.

        Raises:
            RuntimeError: If :meth:`build` has not been called yet.
        """
        if not self._chunks:
            raise RuntimeError("Pipeline not built. Call build(pdf_path) first.")

        # Collect unique file names and total pages
        file_names = set()
        total_pages = 0
        for chunk in self._chunks:
            file_name = chunk.metadata.get("file_name")
            if file_name:
                file_names.add(file_name)
            # Track max page number per file for accurate total
            page_num = chunk.metadata.get("page_number", 0)
            total_pages = max(total_pages, page_num)

        # Deduplicate by (file, page) so each page is counted once per type
        page_types: Dict[tuple, str] = {}
        for chunk in self._chunks:
            file_name = chunk.metadata.get("file_name")
            page = chunk.metadata.get("page_number")
            dt = chunk.metadata.get("pharma_doc_type", "unclassified")
            if file_name is not None and page is not None:
                page_types[(file_name, page)] = dt

        doc_type_counts: Dict[str, int] = {}
        for dt in page_types.values():
            doc_type_counts[dt] = doc_type_counts.get(dt, 0) + 1

        return {
            "total_pages": len(page_types),  # Total unique pages across all files
            "total_chunks": len(self._chunks),
            "total_files": len(file_names),
            "file_names": sorted(list(file_names)),
            "doc_type_counts": doc_type_counts,
            "classified": self._docs_classified,
        }

    def get_document_details(self) -> List[Dict[str, Any]]:
        """Return per-file metadata for every indexed document.

        Aggregates chunk-level metadata into one summary record per source file,
        suitable for display in a UI document panel.

        Returns:
            A list of dicts sorted by filename, each containing:

            - ``file_name`` (*str*) – Base filename.
            - ``total_pages`` (*int*) – Number of unique pages indexed.
            - ``total_chunks`` (*int*) – Number of chunks from this file.
            - ``has_ocr`` (*bool*) – ``True`` if any page was OCR-processed.
            - ``scan_ratio`` (*float*) – Fraction of pages that are scanned (0–1).
            - ``pharma_doc_types`` (*dict*) – Mapping of pharma category → page count.
              Empty when documents were not classified.

        Raises:
            RuntimeError: If :meth:`build` has not been called yet.
        """
        if not self._chunks:
            raise RuntimeError("Pipeline not built. Call build() first.")

        file_data: Dict[str, Dict[str, Any]] = {}

        for chunk in self._chunks:
            meta = chunk.metadata
            fname = meta.get("file_name", "unknown")
            page = meta.get("page_number")
            ocr_used = meta.get("ocr_used", False)
            doc_type = meta.get("doc_type", "digital")
            pharma_type = meta.get("pharma_doc_type")

            if fname not in file_data:
                file_data[fname] = {
                    "file_name": fname,
                    "pages": set(),
                    "scanned_pages": set(),
                    "total_chunks": 0,
                    "has_ocr": False,
                    "pharma_page_types": {},
                }

            fd = file_data[fname]
            fd["total_chunks"] += 1
            if page is not None:
                fd["pages"].add(page)
                if ocr_used or doc_type == "scanned":
                    fd["scanned_pages"].add(page)
            if ocr_used:
                fd["has_ocr"] = True
            if pharma_type and page is not None:
                fd["pharma_page_types"].setdefault(pharma_type, set()).add(page)

        result = []
        for fname, fd in sorted(file_data.items()):
            total_pages = len(fd["pages"])
            scanned = len(fd["scanned_pages"])
            result.append(
                {
                    "file_name": fname,
                    "total_pages": total_pages,
                    "total_chunks": fd["total_chunks"],
                    "has_ocr": fd["has_ocr"],
                    "scan_ratio": round(scanned / total_pages, 2) if total_pages else 0.0,
                    "pharma_doc_types": {k: len(v) for k, v in fd["pharma_page_types"].items()},
                }
            )

        return result

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
        classify: bool = False,
    ) -> str:
        """Query the pipeline and return a plain-text answer.

        Args:
            question: Natural-language question to answer.
            expand: If ``True``, use :meth:`expand_query` to generate
                alternative phrasings before retrieval. Increases latency.
            num_expansions: Number of query expansions to generate when
                ``expand=True``.
            classify: If ``True``, classify the query into a pharma document
                category and restrict retrieval to matching chunks.  Requires
                the index to have been built with ``classify_docs=True``; if
                not, classification still runs but no filtering is applied.

        Returns:
            The LLM's answer as a plain string.

        Raises:
            RuntimeError: If :meth:`build` has not been called yet.
        """
        if self._query_engine is None:
            raise RuntimeError("Pipeline not built. Call build(pdf_path) first.")

        search_query = question
        if expand:
            queries = self.expand_query(question, num_expansions=num_expansions)
            search_query = queries[1] if len(queries) > 1 else question

        engine = self._query_engine
        if classify and self._docs_classified:
            query_category = self._classify_query(search_query)
            if query_category != "unknown":
                engine = self._build_filtered_engine(query_category)

        response = engine.query(search_query)
        return str(response)

    def query_with_sources(
        self,
        question: str,
        expand: bool = False,
        num_expansions: int = 3,
        classify: bool = False,
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
            classify: If ``True``, classify the query into a pharma document
                category and restrict retrieval to matching chunks.  Requires
                the index to have been built with ``classify_docs=True``; if
                not, classification still runs but no filtering is applied.
                The detected category is returned under ``query_category``.

        Returns:
            A dictionary with the following keys:

            - ``answer`` (*str*) – LLM-generated answer.
            - ``sources`` (*list[dict]*) – Retrieved chunks, each containing:

              - ``text`` – Raw chunk text (truncated in the UI, full here).
              - ``file`` – Source filename.
              - ``page`` – 1-based page number.
              - ``score`` – Confidence percentage (0–100), relative to top chunk.
              - ``doc_type`` – ``"digital"`` or ``"scanned"``.
              - ``pharma_doc_type`` – Pharma category label (``"unknown"`` if
                documents were not classified during build).

            - ``chunk_count`` (*int*) – Number of chunks retrieved.
            - ``query_category`` (*str | None*) – Pharma category the query was
              classified into, or ``None`` if ``classify=False``.

        Raises:
            RuntimeError: If :meth:`build` has not been called yet.
        """
        if self._query_engine is None:
            raise RuntimeError("Pipeline not built. Call build(pdf_path) first.")

        search_query = question
        if expand:
            queries = self.expand_query(question, num_expansions=num_expansions)
            search_query = queries[1] if len(queries) > 1 else question

        query_category: Optional[str] = None
        engine = self._query_engine
        if classify:
            query_category = self._classify_query(search_query)
            if self._docs_classified and query_category != "unknown":
                engine = self._build_filtered_engine(query_category)

        response = engine.query(search_query)

        raw_scores = [n.score for n in response.source_nodes if n.score is not None]
        max_score = max(raw_scores) if raw_scores else 0.0

        sources: List[Dict[str, Any]] = []
        for rank, node in enumerate(response.source_nodes):
            meta = node.node.metadata
            if max_score > 0 and node.score is not None:
                confidence = round((node.score / max_score) * 100, 1)
            else:
                confidence = round(100.0 / (rank + 1), 1)
            sources.append(
                {
                    "text": node.node.text,
                    "file": meta.get("file_name", "unknown"),
                    "page": meta.get("page_number", "?"),
                    "score": confidence,
                    "doc_type": meta.get("doc_type", "digital"),
                    "pharma_doc_type": meta.get("pharma_doc_type", "unknown"),
                }
            )

        return {
            "answer": str(response),
            "sources": sources,
            "chunk_count": len(sources),
            "query_category": query_category,
        }

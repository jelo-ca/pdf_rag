"""
RAG Pipeline
============
Reusable RAG object for pharmaceutical document QA.

Embedding Model : sentence-transformers/all-MiniLM-L6-v2
Chunking        : Semantic chunking (LlamaIndex)
Retrieval       : Hybrid – vector (FAISS) + BM25, fused via reciprocal rerank
LLM             : Mistral GGUF (local, via llama-cpp-python)
"""

import os
import fitz  # PyMuPDF
from typing import List, Optional

from llama_index.core import Document, Settings, VectorStoreIndex
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import QueryFusionRetriever, VectorIndexRetriever
from llama_index.core.prompts import PromptTemplate
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.retrievers.bm25 import BM25Retriever

try:
    import pytesseract
    from PIL import Image as PILImage
    _OCR_AVAILABLE = True
except ImportError:
    _OCR_AVAILABLE = False

_PHARMA_QA_PROMPT = PromptTemplate(
    "You are a pharmaceutical document assistant. "
    "Answer the question based ONLY on the context provided below. "
    "After your answer, cite the specific source document name and page number(s) you referenced.\n\n"
    "Context:\n{context_str}\n\n"
    "Question: {query_str}\n\n"
    "Answer (with citations):"
)


class RAGPipeline:
    """
    End-to-end RAG pipeline that can be pointed at any PDF.

    Usage
    -----
    rag = RAGPipeline(model_path=r"C:\\LLM Models\\Mistral\\mistral-7b-instruct-v0.2.Q4_K_M.gguf")
    rag.build("path/to/document.pdf")
    answer = rag.query("What are the storage conditions?")
    print(answer)
    """

    def __init__(
        self,
        model_path: str = r"C:\LLM Models\Mistral\mistral-7b-instruct-v0.2.Q4_K_M.gguf",
        embed_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        similarity_top_k: int = 5,
        num_queries: int = 1,
        n_gpu_layers: int = -1,
    ):
        """
        Parameters
        ----------
        model_path       : Absolute path to the local Mistral GGUF file.
        embed_model_name : HuggingFace sentence-transformer model for embeddings.
        similarity_top_k : Top-k results returned by each retriever.
        num_queries      : Number of LLM-generated query variants for QueryFusion
                           (set to 1 to disable internal expansion).
        n_gpu_layers     : GPU layers to offload (-1 = all, 0 = CPU only).
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"GGUF model not found: {model_path}")

        self.similarity_top_k = similarity_top_k
        self.num_queries = num_queries

        # LLM — loaded directly from local GGUF file
        self.llm = LlamaCPP(
            model_path=model_path,
            temperature=0.1,
            max_new_tokens=1024,
            context_window=32768,
            model_kwargs={"n_gpu_layers": n_gpu_layers},
            verbose=False,
        )

        # Embedding model
        self.embed_model = HuggingFaceEmbedding(model_name=embed_model_name)

        # Apply to global LlamaIndex settings
        Settings.embed_model = self.embed_model
        Settings.llm = self.llm

        # Semantic splitter (reused across builds)
        self._splitter = SemanticSplitterNodeParser(embed_model=self.embed_model)

        # State populated by build()
        self._chunks: list = []
        self._vector_index: Optional[VectorStoreIndex] = None
        self._query_engine: Optional[RetrieverQueryEngine] = None
        self._pdf_path: Optional[str] = None

    # ------------------------------------------------------------------
    # PDF Loading
    # ------------------------------------------------------------------

    def _ocr_page(self, page) -> str:
        """Render a PDF page as an image and extract text via Tesseract OCR."""
        pix = page.get_pixmap(dpi=300)
        img = PILImage.frombytes("RGB", [pix.width, pix.height], pix.samples)
        return pytesseract.image_to_string(img)

    def load_pdf(self, pdf_path: str) -> List[Document]:
        """
        Extract text page-by-page from a PDF using PyMuPDF.
        Falls back to Tesseract OCR for scanned pages (< 100 chars extracted).

        Returns a list of LlamaIndex Document objects with rich metadata.
        """
        documents = []
        file_name = os.path.basename(pdf_path)

        with fitz.open(pdf_path) as doc:
            for i, page in enumerate(doc):
                text = page.get_text()
                ocr_used = False

                # Scanned page detection: fall back to OCR if very little text extracted
                if len(text.strip()) < 100 and _OCR_AVAILABLE:
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

        print(f"Loaded '{file_name}': {len(documents)} pages with content.")
        return documents

    # ------------------------------------------------------------------
    # Chunking
    # ------------------------------------------------------------------

    def _chunk(self, documents: List[Document]) -> list:
        """Semantic chunking on a list of Document objects."""
        if not documents:
            raise ValueError("No documents provided for chunking.")
        print("Performing semantic chunking...")
        chunks = self._splitter.get_nodes_from_documents(documents)
        print(f"Total chunks created: {len(chunks)}")
        return chunks

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def _index(self, chunks: list) -> VectorStoreIndex:
        """Create an in-memory VectorStoreIndex from chunk nodes."""
        vector_index = VectorStoreIndex.from_documents(chunks)
        print(f"Indexed {len(chunks)} chunks.")
        return vector_index

    # ------------------------------------------------------------------
    # Retriever
    # ------------------------------------------------------------------

    def _build_retriever(self, vector_index: VectorStoreIndex, chunks: list) -> QueryFusionRetriever:
        """Combine a vector retriever and BM25 retriever via reciprocal rerank fusion."""
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
        print("Hybrid retriever ready.")
        return hybrid_retriever

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(self, pdf_path: str) -> None:
        """
        Full pipeline: load PDF → chunk → embed & index → build retriever + query engine.

        Call this once per document (or call again to swap documents).
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
        print("RAG pipeline ready.\n")

    def expand_query(self, query: str, num_expansions: int = 3) -> List[str]:
        """
        Use the LLM to generate alternative phrasings of a query.

        Returns the original query prepended to the list of expansions.
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
        """
        Query the built RAG pipeline.

        Parameters
        ----------
        question       : Natural-language question.
        expand         : If True, expand the query via LLM before retrieval.
        num_expansions : Number of LLM-generated query expansions (used when expand=True).

        Returns
        -------
        Answer string from the LLM.
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
    ) -> dict:
        """
        Query the pipeline and return a structured result with answer, sources,
        confidence scores, and chunk count.

        Returns
        -------
        {
            "answer"      : str,
            "sources"     : [{"text", "file", "page", "score", "doc_type"}, ...],
            "chunk_count" : int,
        }
        """
        if self._query_engine is None:
            raise RuntimeError("Pipeline not built. Call build(pdf_path) first.")

        search_query = question
        if expand:
            queries = self.expand_query(question, num_expansions=num_expansions)
            search_query = queries[1] if len(queries) > 1 else question

        response = self._query_engine.query(search_query)

        # Confidence: normalize RRF scores to 0–100%.
        # If all scores are 0 or None (common with QueryFusionRetriever), fall back
        # to rank-based confidence: 1st chunk = 100%, 2nd = 50%, 3rd = 33%, etc.
        raw_scores = [n.score for n in response.source_nodes if n.score is not None]
        max_score = max(raw_scores) if raw_scores else 0.0

        sources = []
        for rank, node in enumerate(response.source_nodes):
            meta = node.node.metadata
            if max_score > 0 and node.score is not None:
                confidence = round((node.score / max_score) * 100, 1)
            else:
                confidence = round(100.0 / (rank + 1), 1)  # rank-based fallback
            sources.append({
                "text": node.node.text,
                "file": meta.get("file_name", "unknown"),
                "page": meta.get("page_number", "?"),
                "score": confidence,   # 0–100 confidence % relative to top chunk
                "doc_type": meta.get("doc_type", "digital"),
            })

        return {
            "answer": str(response),
            "sources": sources,
            "chunk_count": len(sources),
        }

"""
OCR Quality Tests  (pipeline.py — load_images, _ocr_page)
==========================================================
Tests OCR output quality on the 5 ``image_test_*`` folders in ``docs/``.

Markers
-------
unit      : metric-helper unit tests; no documents or Tesseract required.
ocr_scan  : real OCR tests; require a Tesseract binary on PATH.

Run unit tests only:
    pytest tests/test_ocr_quality.py -m unit -v

Run OCR tests (Tesseract required):
    pytest tests/test_ocr_quality.py -m ocr_scan -v
"""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from rag.pipeline import (
    _PHARMA_DOC_CATEGORIES,
    RAGPipeline,
    _is_ocr_runtime_available,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DOCS_DIR    = Path(__file__).parent.parent / "docs"
IMAGE_FOLDERS = [f"image_test_{i}" for i in range(1, 6)]

IMAGE_FILE_COUNTS = {
    "image_test_1": 17,
    "image_test_2": 18,
    "image_test_3": 18,
    "image_test_4": 18,
    "image_test_5": 17,
}

# Intrinsic quality thresholds
MIN_CHARS_PER_PAGE       = 50
WORD_DENSITY_THRESHOLD   = 0.50
AVG_WORD_LEN_MIN         = 3.0
AVG_WORD_LEN_MAX         = 12.0
MIN_VOCAB_SIZE           = 100
KW_PRESERVATION_THRESHOLD = 0.30

# Pharma terms expected in both SDS and regulatory submission docs
CRITICAL_KEYWORDS = [
    "pfizer", "product", "date", "company", "lot",
    "certificate", "batch", "manufacturer", "submission", "protocol",
]

needs_tesseract = pytest.mark.skipif(
    not _is_ocr_runtime_available(),
    reason="Tesseract binary not available — skipping OCR scan tests",
)


# ---------------------------------------------------------------------------
# Pure-Python metric helpers
# ---------------------------------------------------------------------------

def _intrinsic_stats(text: str) -> dict:
    words = re.findall(r"\b[a-zA-Z]+\b", text)
    non_ws  = len(re.sub(r"\s", "", text))
    wchars  = sum(len(w) for w in words)
    return {
        "word_count":    len(words),
        "unique_words":  len({w.lower() for w in words}),
        "word_density":  wchars / max(non_ws, 1),
        "avg_word_length": sum(len(w) for w in words) / max(len(words), 1),
    }


def _keyword_preservation(text: str, keywords: list) -> float:
    if not keywords:
        return 0.0
    low = text.lower()
    return sum(1 for kw in keywords if kw in low) / len(keywords)


# ---------------------------------------------------------------------------
# UNIT: metric helper correctness
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestMetricHelpers:

    def test_word_density_pure_words(self):
        stats = _intrinsic_stats("hello world foo bar")
        assert stats["word_density"] == pytest.approx(1.0)

    def test_word_density_with_noise_chars(self):
        stats = _intrinsic_stats("abc !@#$%")
        # "abc !@#$%" → 8 non-whitespace chars (a,b,c,!,@,#,$,%), 3 word chars
        assert stats["word_density"] == pytest.approx(3 / 8, abs=0.01)

    def test_unique_words_case_insensitive(self):
        stats = _intrinsic_stats("Hello hello HELLO world")
        assert stats["unique_words"] == 2  # "hello" and "world"

    def test_avg_word_length_simple(self):
        stats = _intrinsic_stats("cat dog bird")  # lengths 3, 3, 4
        assert stats["avg_word_length"] == pytest.approx(10 / 3, abs=0.01)

    def test_empty_text_returns_zeros(self):
        stats = _intrinsic_stats("")
        assert stats["word_count"] == 0
        assert stats["unique_words"] == 0
        assert stats["word_density"] == pytest.approx(0.0)

    def test_kw_preservation_all_present(self):
        text = " ".join(CRITICAL_KEYWORDS) + " extra words"
        assert _keyword_preservation(text, CRITICAL_KEYWORDS) == pytest.approx(1.0)

    def test_kw_preservation_none_present(self):
        assert _keyword_preservation("lorem ipsum dolor", CRITICAL_KEYWORDS) == pytest.approx(0.0)

    def test_kw_preservation_empty_list_returns_zero(self):
        assert _keyword_preservation("anything here", []) == pytest.approx(0.0)

    def test_kw_preservation_partial(self):
        text = "pfizer product batch"   # 3 of 10 keywords
        rate = _keyword_preservation(text, CRITICAL_KEYWORDS)
        assert pytest.approx(0.3, abs=0.01) == rate


# ---------------------------------------------------------------------------
# OCR SCAN TESTS — require real Tesseract binary
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def ocr_pipeline(tmp_path_factory):
    """Session-scoped pipeline; fitz/pytesseract are real, LLM is mocked."""
    p = tmp_path_factory.mktemp("model") / "model.gguf"
    p.write_bytes(b"")
    with patch("rag.pipeline.LlamaCPP") as mlc, \
         patch("rag.pipeline.HuggingFaceEmbedding") as mec, \
         patch("rag.pipeline.SentenceSplitter") as msc, \
         patch("rag.pipeline.Settings"):
        mlc.return_value = MagicMock(name="llm")
        mec.return_value = MagicMock(name="embed")
        msc.return_value = MagicMock(name="splitter")
        rag = RAGPipeline(model_path=str(p))
    rag.llm.complete.return_value = MagicMock(text="unknown")
    return rag


@pytest.mark.ocr_scan
@needs_tesseract
class TestOCRDocumentProperties:
    """Metadata and structural checks on docs returned by load_images."""

    @pytest.mark.parametrize("folder", IMAGE_FOLDERS)
    def test_load_images_returns_at_least_one_doc(self, ocr_pipeline, folder):
        docs = ocr_pipeline.load_images(str(DOCS_DIR / folder))
        assert len(docs) >= 1

    @pytest.mark.parametrize("folder", IMAGE_FOLDERS)
    def test_ocr_used_is_true_on_all_pages(self, ocr_pipeline, folder):
        docs = ocr_pipeline.load_images(str(DOCS_DIR / folder))
        assert all(d.metadata["ocr_used"] for d in docs)

    @pytest.mark.parametrize("folder", IMAGE_FOLDERS)
    def test_doc_type_is_scanned(self, ocr_pipeline, folder):
        docs = ocr_pipeline.load_images(str(DOCS_DIR / folder))
        assert all(d.metadata["doc_type"] == "scanned" for d in docs)

    @pytest.mark.parametrize("folder", IMAGE_FOLDERS)
    def test_source_id_matches_folder_colon_page_pattern(self, ocr_pipeline, folder):
        docs = ocr_pipeline.load_images(str(DOCS_DIR / folder))
        pattern = re.compile(rf"^{re.escape(folder)}:p\d+$")
        for doc in docs:
            assert pattern.match(doc.metadata["source_id"]), (
                f"Bad source_id: {doc.metadata['source_id']}"
            )

    @pytest.mark.parametrize("folder", IMAGE_FOLDERS)
    def test_total_pages_metadata_equals_image_count(self, ocr_pipeline, folder):
        expected = IMAGE_FILE_COUNTS[folder]
        docs = ocr_pipeline.load_images(str(DOCS_DIR / folder))
        for doc in docs:
            assert doc.metadata["total_pages"] == expected


@pytest.mark.ocr_scan
@needs_tesseract
class TestOCRIntrinsicQuality:
    """Readability thresholds — no reference document required."""

    @pytest.mark.parametrize("folder", IMAGE_FOLDERS)
    def test_every_page_has_min_chars(self, ocr_pipeline, folder):
        docs = ocr_pipeline.load_images(str(DOCS_DIR / folder))
        short = [(d.metadata.get("page_number"), len(d.text))
                 for d in docs if len(d.text) < MIN_CHARS_PER_PAGE]
        assert not short, f"{folder}: pages below {MIN_CHARS_PER_PAGE} chars: {short}"

    @pytest.mark.parametrize("folder", IMAGE_FOLDERS)
    def test_word_density_above_threshold(self, ocr_pipeline, folder):
        text = "\n".join(d.text for d in ocr_pipeline.load_images(str(DOCS_DIR / folder)))
        stats = _intrinsic_stats(text)
        assert stats["word_density"] >= WORD_DENSITY_THRESHOLD, (
            f"{folder}: word density {stats['word_density']:.2%} < {WORD_DENSITY_THRESHOLD:.0%}"
        )

    @pytest.mark.parametrize("folder", IMAGE_FOLDERS)
    def test_avg_word_length_in_plausible_range(self, ocr_pipeline, folder):
        text = "\n".join(d.text for d in ocr_pipeline.load_images(str(DOCS_DIR / folder)))
        avg = _intrinsic_stats(text)["avg_word_length"]
        assert AVG_WORD_LEN_MIN <= avg <= AVG_WORD_LEN_MAX, (
            f"{folder}: avg word length {avg:.2f} outside [{AVG_WORD_LEN_MIN}, {AVG_WORD_LEN_MAX}]"
        )

    @pytest.mark.parametrize("folder", IMAGE_FOLDERS)
    def test_vocabulary_size_above_minimum(self, ocr_pipeline, folder):
        text = "\n".join(d.text for d in ocr_pipeline.load_images(str(DOCS_DIR / folder)))
        unique = _intrinsic_stats(text)["unique_words"]
        assert unique >= MIN_VOCAB_SIZE, (
            f"{folder}: only {unique} unique words (< {MIN_VOCAB_SIZE})"
        )

    @pytest.mark.parametrize("folder", IMAGE_FOLDERS)
    def test_critical_keyword_preservation_above_threshold(self, ocr_pipeline, folder):
        text = "\n".join(d.text for d in ocr_pipeline.load_images(str(DOCS_DIR / folder)))
        rate = _keyword_preservation(text, CRITICAL_KEYWORDS)
        missing = [kw for kw in CRITICAL_KEYWORDS if kw not in text.lower()]
        assert rate >= KW_PRESERVATION_THRESHOLD, (
            f"{folder}: keyword preservation {rate:.0%} < {KW_PRESERVATION_THRESHOLD:.0%}. "
            f"Missing: {missing}"
        )


@pytest.mark.ocr_scan
@needs_tesseract
class TestOCRClassification:
    """Classification correctness on OCR-extracted image text."""

    @pytest.mark.parametrize("folder", IMAGE_FOLDERS)
    def test_all_docs_receive_a_pharma_doc_type(self, ocr_pipeline, folder):
        docs = ocr_pipeline.load_images(str(DOCS_DIR / folder))
        ocr_pipeline._annotate_pharma_doc_types(docs)
        missing = [i for i, d in enumerate(docs) if "pharma_doc_type" not in d.metadata]
        assert not missing, f"{folder}: docs at indices {missing} have no pharma_doc_type"

    @pytest.mark.parametrize("folder", IMAGE_FOLDERS)
    def test_all_pharma_doc_types_are_valid(self, ocr_pipeline, folder):
        docs = ocr_pipeline.load_images(str(DOCS_DIR / folder))
        ocr_pipeline._annotate_pharma_doc_types(docs)
        invalid = [
            d.metadata["pharma_doc_type"]
            for d in docs
            if d.metadata.get("pharma_doc_type") not in _PHARMA_DOC_CATEGORIES
        ]
        assert not invalid, f"{folder}: invalid categories: {invalid}"

    @pytest.mark.parametrize("folder", IMAGE_FOLDERS)
    def test_classification_is_deterministic(self, ocr_pipeline, folder):
        """Running annotation twice on identical docs must produce identical results."""
        import copy
        docs  = ocr_pipeline.load_images(str(DOCS_DIR / folder))
        docs2 = copy.deepcopy(docs)
        ocr_pipeline.llm.complete.return_value = MagicMock(text="unknown")
        ocr_pipeline._annotate_pharma_doc_types(docs)
        ocr_pipeline.llm.complete.return_value = MagicMock(text="unknown")
        ocr_pipeline._annotate_pharma_doc_types(docs2)
        for d1, d2 in zip(docs, docs2):
            assert d1.metadata["pharma_doc_type"] == d2.metadata["pharma_doc_type"], (
                f"{folder} p{d1.metadata.get('page_number')}: "
                f"'{d1.metadata['pharma_doc_type']}' vs '{d2.metadata['pharma_doc_type']}'"
            )

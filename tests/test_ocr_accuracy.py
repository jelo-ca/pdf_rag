"""OCR Accuracy Test Suite
=========================
Tests OCR quality and classification correctness for the 5 ``image_test_*``
folders in ``docs/``.  All OCR is performed by the real pipeline
(``src/rag/pipeline.py``) — no OCR mocking.  Only LLM / embedding / splitter
are stubbed so the suite runs without GPU hardware.

Markers
-------
unit      -- pure-Python metric helpers and keyword-classification unit tests;
             no real documents or OCR required.
ocr_scan  -- OCR accuracy, document property, and classification tests that
             require a Tesseract binary to be installed.  Skipped automatically
             when ``_is_ocr_runtime_available()`` returns False.
"""
from __future__ import annotations

import re
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List
from unittest.mock import MagicMock, patch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from rag.pipeline import (
    _KEYWORD_MAP,
    _PHARMA_DOC_CATEGORIES,
    RAGPipeline,
    _is_ocr_runtime_available,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DOCS_DIR = Path(__file__).parent.parent / "docs"
ARTIFACTS_DIR = Path(__file__).parent.parent / "artifacts"

IMAGE_FOLDERS: List[str] = [
    "image_test_1",
    "image_test_2",
    "image_test_3",
    "image_test_4",
    "image_test_5",
]

PDF_COMPANIONS: Dict[str, str] = {f"image_test_{i}": f"test_{i}.pdf" for i in range(1, 6)}

IMAGE_FILE_COUNTS: Dict[str, int] = {
    "image_test_1": 17,
    "image_test_2": 18,
    "image_test_3": 18,
    "image_test_4": 18,
    "image_test_5": 17,
}

# General pharma / regulatory terms expected in both SDS and regulatory submission docs.
# image_test folders contain FDA regulatory submissions; we test for universally
# present pharma terms rather than SDS-specific terms.
CRITICAL_KEYWORDS: List[str] = [
    "pfizer",
    "product",
    "date",
    "company",
    "lot",
    "certificate",
    "batch",
    "manufacturer",
    "submission",
    "protocol",
]

# Intrinsic OCR quality thresholds (no ground-truth reference required)
MIN_CHARS_PER_PAGE = 50          # every page must have ≥ 50 characters
WORD_DENSITY_THRESHOLD = 0.50    # ≥ 50 % of non-whitespace chars inside words
AVG_WORD_LEN_MIN = 3.0           # mean word length must be ≥ 3 chars
AVG_WORD_LEN_MAX = 12.0          # mean word length must be ≤ 12 chars (no noise)
MIN_VOCAB_SIZE = 100             # unique word count per folder ≥ 100

# Kept for potential future companion-PDF comparison work.
CER_THRESHOLD = 0.40
WER_THRESHOLD = 0.40
TOKEN_F1_THRESHOLD = 0.60
KW_PRESERVATION_THRESHOLD = 0.30  # lowered: image_test docs are regulatory, not SDS

# Skip marker for every test that requires a real Tesseract binary.
pytesseract_available = pytest.mark.skipif(
    not _is_ocr_runtime_available(),
    reason="pytesseract and Tesseract binary not available",
)


# ---------------------------------------------------------------------------
# Metric helpers (module-level, pure Python — no pipeline dependency)
# ---------------------------------------------------------------------------


def compute_cer(ref: str, hyp: str) -> float:
    """Character Error Rate via SequenceMatcher (1 - similarity ratio).

    Both strings are whitespace-normalised before comparison.
    Returns 1.0 when the reference is empty.
    """
    ref_norm = " ".join(ref.split())
    hyp_norm = " ".join(hyp.split())
    if not ref_norm:
        return 1.0
    return 1.0 - SequenceMatcher(None, ref_norm, hyp_norm).ratio()


def compute_wer(ref: str, hyp: str) -> float:
    """Word Error Rate via dynamic-programming edit distance on word lists.

    Returns 1.0 when the reference contains no words.
    """
    ref_words = ref.lower().split()
    hyp_words = hyp.lower().split()
    if not ref_words:
        return 1.0

    n, m = len(ref_words), len(hyp_words)
    # DP edit distance
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[:]
        dp[0] = i
        for j in range(1, m + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                dp[j] = prev[j - 1]
            else:
                dp[j] = 1 + min(prev[j], dp[j - 1], prev[j - 1])
    return min(dp[m] / n, 1.0)


def compute_token_f1(ref: str, hyp: str) -> float:
    """SQuAD-style token-level F1 using Counter intersection (duplicates counted)."""
    ref_tokens = Counter(re.findall(r"\b\w+\b", ref.lower()))
    hyp_tokens = Counter(re.findall(r"\b\w+\b", hyp.lower()))
    common = sum((ref_tokens & hyp_tokens).values())
    if common == 0:
        return 0.0
    precision = common / max(sum(hyp_tokens.values()), 1)
    recall = common / max(sum(ref_tokens.values()), 1)
    return 2 * precision * recall / (precision + recall)


def keyword_preservation_rate(text: str, keywords: List[str]) -> float:
    """Fraction of *keywords* found (case-insensitive) in *text*."""
    if not keywords:
        return 0.0
    lowered = text.lower()
    return sum(1 for kw in keywords if kw in lowered) / len(keywords)


def _intrinsic_stats(text: str) -> dict:
    """Compute intrinsic OCR quality stats that need no reference document.

    Returns a dict with:
    - ``word_count``     – total English words found
    - ``unique_words``   – vocabulary size (lower-cased)
    - ``word_density``   – fraction of non-whitespace chars that are inside words
    - ``avg_word_length``– mean word length in characters
    """
    words = re.findall(r"\b[a-zA-Z]+\b", text)
    non_ws_chars = len(re.sub(r"\s", "", text))
    word_chars = sum(len(w) for w in words)
    return {
        "word_count": len(words),
        "unique_words": len(set(w.lower() for w in words)),
        "word_density": word_chars / max(non_ws_chars, 1),
        "avg_word_length": sum(len(w) for w in words) / max(len(words), 1),
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def pipeline(tmp_path_factory):
    """RAGPipeline with every heavy dependency patched out.

    Mirrors the fixture in ``test_rag_pipeline.py`` (lines 33-49).
    ``fitz``, ``pytesseract``, and ``_is_ocr_runtime_available`` are
    intentionally left real so that integration tests run real OCR.
    """
    p = tmp_path_factory.mktemp("model") / "model.gguf"
    p.write_bytes(b"")

    with patch("rag.pipeline.LlamaCPP") as mock_llm_cls, \
         patch("rag.pipeline.HuggingFaceEmbedding") as mock_embed_cls, \
         patch("rag.pipeline.SentenceSplitter") as mock_splitter_cls, \
         patch("rag.pipeline.Settings"):

        mock_llm_cls.return_value = MagicMock(name="llm")
        mock_embed_cls.return_value = MagicMock(name="embed")
        mock_splitter_cls.return_value = MagicMock(name="splitter")

        rag = RAGPipeline(model_path=str(p))

    # Default LLM mock: any fallback call returns "unknown" (a valid category)
    # so keyword-matched pages can be distinguished from LLM-fallback pages.
    rag.llm.complete.return_value = MagicMock(text="unknown")

    return rag


# ---------------------------------------------------------------------------
# Helpers used across multiple test classes
# ---------------------------------------------------------------------------


def _ocr_text(pipeline: RAGPipeline, folder_name: str) -> str:
    """Return concatenated OCR text for an image folder."""
    docs = pipeline.load_images(str(DOCS_DIR / folder_name))
    return "\n".join(d.text for d in docs)


# ---------------------------------------------------------------------------
# UNIT TESTS — pure Python, no real documents or OCR
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMetricHelpersUnit:
    """Verify CER, WER, Token F1, and keyword-preservation helpers."""

    # -- CER --
    def test_cer_identical(self):
        assert compute_cer("hello world", "hello world") == pytest.approx(0.0)

    def test_cer_completely_different(self):
        assert compute_cer("abcd", "wxyz") == pytest.approx(1.0)

    def test_cer_partial(self):
        # One-character transposition — CER should be small but non-zero
        cer = compute_cer("hello world", "hello wrold")
        assert 0.0 < cer < 0.20

    def test_cer_empty_reference(self):
        assert compute_cer("", "something") == pytest.approx(1.0)

    def test_cer_empty_hypothesis(self):
        cer = compute_cer("some reference text", "")
        assert cer == pytest.approx(1.0)

    # -- WER --
    def test_wer_identical(self):
        assert compute_wer("the quick fox", "the quick fox") == pytest.approx(0.0)

    def test_wer_one_substitution(self):
        # 1 substitution out of 3 reference words
        wer = compute_wer("the quick fox", "the slow fox")
        assert wer == pytest.approx(1 / 3, abs=0.01)

    def test_wer_empty_reference(self):
        assert compute_wer("", "anything") == pytest.approx(1.0)

    # -- Token F1 --
    def test_token_f1_identical(self):
        assert compute_token_f1("cat dog bird", "cat dog bird") == pytest.approx(1.0)

    def test_token_f1_no_overlap(self):
        assert compute_token_f1("cat dog", "fish bird") == pytest.approx(0.0)

    def test_token_f1_partial(self):
        f1 = compute_token_f1("cat dog fish", "cat bird fish")
        assert 0.0 < f1 < 1.0

    # -- Keyword preservation --
    def test_kw_preservation_all_present(self):
        text = " ".join(CRITICAL_KEYWORDS) + " extra filler words"
        assert keyword_preservation_rate(text, CRITICAL_KEYWORDS) == pytest.approx(1.0)

    def test_kw_preservation_none_present(self):
        text = "lorem ipsum dolor sit amet consectetur adipiscing"
        assert keyword_preservation_rate(text, CRITICAL_KEYWORDS) == pytest.approx(0.0)

    def test_kw_preservation_empty_list(self):
        assert keyword_preservation_rate("anything", []) == pytest.approx(0.0)


@pytest.mark.unit
class TestKeywordClassificationUnit:
    """Verify the pipeline's keyword fast-path via ``pipeline._classify_document``.

    The LLM mock is pre-configured to return ``"unknown"``, so tests can
    distinguish pages resolved by keyword (LLM not called) from those that
    fall through to the LLM.
    """

    @pytest.mark.parametrize("category,keywords", list(_KEYWORD_MAP.items()))
    def test_keyword_hit_returns_correct_category(self, pipeline, category, keywords):
        """First keyword for every ``_KEYWORD_MAP`` entry triggers keyword path."""
        trigger = keywords[0]
        text = trigger + " " + ("x " * 50)  # keyword well within first 300 chars
        pipeline.llm.complete.reset_mock()

        result = pipeline._classify_document(text)

        assert result == category
        pipeline.llm.complete.assert_not_called()

    def test_no_keyword_falls_through_to_llm(self, pipeline):
        """Text with no keyword triggers the LLM fallback."""
        text = "this document contains no recognisable pharmaceutical keyword signals"
        pipeline.llm.complete.reset_mock()

        result = pipeline._classify_document(text)

        pipeline.llm.complete.assert_called_once()
        assert result == "unknown"

    def test_boundary_keyword_fully_inside_300_chars(self, pipeline):
        """Keyword whose last char falls at position 299 must match."""
        keyword = list(_KEYWORD_MAP["certificate_of_quality"])[0]
        # Place keyword so it ends exactly at char 299 (0-indexed)
        start = 300 - len(keyword)
        padding = "x" * start
        text = padding + keyword
        assert len(text[:300]) == 300
        pipeline.llm.complete.reset_mock()

        result = pipeline._classify_document(text)

        assert result == "certificate_of_quality"
        pipeline.llm.complete.assert_not_called()

    def test_boundary_keyword_starts_past_300_chars(self, pipeline):
        """Keyword starting at char 300 (outside the header) must fall through to LLM."""
        keyword = list(_KEYWORD_MAP["certificate_of_quality"])[0]
        padding = "x" * 300  # keyword begins at position 300, past the window
        text = padding + keyword
        pipeline.llm.complete.reset_mock()

        result = pipeline._classify_document(text)

        pipeline.llm.complete.assert_called_once()
        _ = result  # result is whatever the LLM mock returns

    def test_result_always_in_pharma_categories(self, pipeline):
        """``_classify_document`` must always return a valid pharma category."""
        for text in [
            "Certificate of Analysis — Batch 12345",
            "completely unrelated document with no signals whatsoever",
            "",
        ]:
            result = pipeline._classify_document(text)
            assert result in _PHARMA_DOC_CATEGORIES, (
                f"Got '{result}' which is not in _PHARMA_DOC_CATEGORIES"
            )


# ---------------------------------------------------------------------------
# OCR TESTS — require real Tesseract binary
# ---------------------------------------------------------------------------


@pytest.mark.ocr_scan
@pytesseract_available
class TestOCRDocumentProperties:
    """Verify metadata fields on documents returned by ``pipeline.load_images``."""

    @pytest.mark.parametrize("folder_name", IMAGE_FOLDERS)
    def test_produces_documents(self, pipeline, folder_name):
        docs = pipeline.load_images(str(DOCS_DIR / folder_name))
        assert len(docs) >= 1, f"{folder_name}: load_images returned no documents"

    @pytest.mark.parametrize("folder_name", IMAGE_FOLDERS)
    def test_all_pages_ocr_used_true(self, pipeline, folder_name):
        docs = pipeline.load_images(str(DOCS_DIR / folder_name))
        assert all(d.metadata["ocr_used"] for d in docs), (
            f"{folder_name}: some pages have ocr_used=False"
        )

    @pytest.mark.parametrize("folder_name", IMAGE_FOLDERS)
    def test_all_pages_doc_type_scanned(self, pipeline, folder_name):
        docs = pipeline.load_images(str(DOCS_DIR / folder_name))
        assert all(d.metadata["doc_type"] == "scanned" for d in docs), (
            f"{folder_name}: some pages have doc_type != 'scanned'"
        )

    @pytest.mark.parametrize("folder_name", IMAGE_FOLDERS)
    def test_page_count_leq_image_count(self, pipeline, folder_name):
        docs = pipeline.load_images(str(DOCS_DIR / folder_name))
        image_count = IMAGE_FILE_COUNTS[folder_name]
        assert len(docs) <= image_count, (
            f"{folder_name}: {len(docs)} docs > {image_count} images "
            "(empty-page skipping appears broken)"
        )

    @pytest.mark.parametrize("folder_name", IMAGE_FOLDERS)
    def test_source_id_format(self, pipeline, folder_name):
        docs = pipeline.load_images(str(DOCS_DIR / folder_name))
        pattern = re.compile(rf"^{re.escape(folder_name)}:p\d+$")
        for doc in docs:
            sid = doc.metadata["source_id"]
            assert pattern.match(sid), (
                f"{folder_name}: source_id '{sid}' does not match expected format"
            )

    @pytest.mark.parametrize("folder_name", IMAGE_FOLDERS)
    def test_total_pages_metadata(self, pipeline, folder_name):
        docs = pipeline.load_images(str(DOCS_DIR / folder_name))
        expected = IMAGE_FILE_COUNTS[folder_name]
        for doc in docs:
            assert doc.metadata["total_pages"] == expected, (
                f"{folder_name} p{doc.metadata['page_number']}: "
                f"total_pages={doc.metadata['total_pages']} != {expected}"
            )


@pytest.mark.ocr_scan
@pytesseract_available
class TestOCRIntrinsicQuality:
    """Intrinsic OCR quality tests that need no reference document.

    The image_test folders contain FDA regulatory submission scans (not the same
    documents as the SDS PDFs in docs/).  These tests measure readability of the
    OCR output directly — word density, vocabulary richness, and per-page text
    length — to catch catastrophic OCR failures without requiring ground-truth text.
    """

    @pytest.mark.parametrize("folder_name", IMAGE_FOLDERS)
    def test_min_chars_per_page(self, pipeline, folder_name):
        """Every page must contain at least MIN_CHARS_PER_PAGE characters."""
        docs = pipeline.load_images(str(DOCS_DIR / folder_name))
        short = [
            (d.metadata.get("page_number"), len(d.text))
            for d in docs
            if len(d.text) < MIN_CHARS_PER_PAGE
        ]
        assert not short, (
            f"{folder_name}: pages below {MIN_CHARS_PER_PAGE} chars: {short}"
        )

    @pytest.mark.parametrize("folder_name", IMAGE_FOLDERS)
    def test_word_density_adequate(self, pipeline, folder_name):
        """≥ 50 % of non-whitespace characters must be inside recognised words."""
        text = _ocr_text(pipeline, folder_name)
        stats = _intrinsic_stats(text)
        assert stats["word_density"] >= WORD_DENSITY_THRESHOLD, (
            f"{folder_name}: word density {stats['word_density']:.2%} "
            f"< threshold {WORD_DENSITY_THRESHOLD:.0%} (OCR may be producing noise)"
        )

    @pytest.mark.parametrize("folder_name", IMAGE_FOLDERS)
    def test_avg_word_length_plausible(self, pipeline, folder_name):
        """Mean word length must fall in [3, 12] chars — ruling out noise tokens."""
        text = _ocr_text(pipeline, folder_name)
        stats = _intrinsic_stats(text)
        avg = stats["avg_word_length"]
        assert AVG_WORD_LEN_MIN <= avg <= AVG_WORD_LEN_MAX, (
            f"{folder_name}: avg word length {avg:.2f} outside "
            f"[{AVG_WORD_LEN_MIN}, {AVG_WORD_LEN_MAX}]"
        )

    @pytest.mark.parametrize("folder_name", IMAGE_FOLDERS)
    def test_vocabulary_size(self, pipeline, folder_name):
        """Unique word count across the folder must reach MIN_VOCAB_SIZE."""
        text = _ocr_text(pipeline, folder_name)
        stats = _intrinsic_stats(text)
        assert stats["unique_words"] >= MIN_VOCAB_SIZE, (
            f"{folder_name}: only {stats['unique_words']} unique words "
            f"(< {MIN_VOCAB_SIZE}) — OCR output may be degenerate"
        )

    @pytest.mark.parametrize("folder_name", IMAGE_FOLDERS)
    def test_keyword_preservation(self, pipeline, folder_name):
        """At least KW_PRESERVATION_THRESHOLD fraction of CRITICAL_KEYWORDS appear."""
        ocr = _ocr_text(pipeline, folder_name)
        rate = keyword_preservation_rate(ocr, CRITICAL_KEYWORDS)
        missing = [kw for kw in CRITICAL_KEYWORDS if kw not in ocr.lower()]
        assert rate >= KW_PRESERVATION_THRESHOLD, (
            f"{folder_name}: keyword preservation {rate:.0%} "
            f"< {KW_PRESERVATION_THRESHOLD:.0%}. Missing: {missing}"
        )


@pytest.mark.ocr_scan
@pytesseract_available
class TestOCRClassification:
    """Verify classification of OCR text through the actual pipeline methods."""

    @pytest.mark.parametrize("folder_name", IMAGE_FOLDERS)
    def test_all_docs_have_pharma_doc_type(self, pipeline, folder_name):
        docs = pipeline.load_images(str(DOCS_DIR / folder_name))
        pipeline._annotate_pharma_doc_types(docs)
        assert all("pharma_doc_type" in d.metadata for d in docs), (
            f"{folder_name}: some docs missing pharma_doc_type after annotation"
        )

    @pytest.mark.parametrize("folder_name", IMAGE_FOLDERS)
    def test_all_pharma_doc_types_are_valid(self, pipeline, folder_name):
        docs = pipeline.load_images(str(DOCS_DIR / folder_name))
        pipeline._annotate_pharma_doc_types(docs)
        invalid = [
            d.metadata["pharma_doc_type"]
            for d in docs
            if d.metadata.get("pharma_doc_type") not in _PHARMA_DOC_CATEGORIES
        ]
        assert not invalid, (
            f"{folder_name}: invalid pharma_doc_type values: {invalid}"
        )

    @pytest.mark.parametrize("folder_name", IMAGE_FOLDERS)
    def test_classification_is_deterministic(self, pipeline, folder_name):
        """Annotating the same docs twice must produce identical results."""
        import copy

        docs = pipeline.load_images(str(DOCS_DIR / folder_name))

        # Deep-copy so second run starts from same state
        docs_copy = copy.deepcopy(docs)

        pipeline.llm.complete.return_value = MagicMock(text="unknown")
        pipeline._annotate_pharma_doc_types(docs)
        pipeline.llm.complete.return_value = MagicMock(text="unknown")
        pipeline._annotate_pharma_doc_types(docs_copy)

        for d1, d2 in zip(docs, docs_copy):
            assert d1.metadata["pharma_doc_type"] == d2.metadata["pharma_doc_type"], (
                f"{folder_name} p{d1.metadata.get('page_number')}: "
                f"got '{d1.metadata['pharma_doc_type']}' then '{d2.metadata['pharma_doc_type']}'"
            )

    @pytest.mark.parametrize("folder_name", IMAGE_FOLDERS)
    def test_keyword_matched_pages_skip_llm(self, pipeline, folder_name):
        """Pages classified by keyword must not trigger an LLM call."""
        docs = pipeline.load_images(str(DOCS_DIR / folder_name))
        pipeline.llm.complete.reset_mock()
        pipeline.llm.complete.return_value = MagicMock(text="unknown")

        annotated = pipeline._annotate_pharma_doc_types(docs)

        # Pages with a non-"unknown" category were resolved by keyword (Stage 1)
        keyword_resolved = [
            d for d in annotated
            if d.metadata.get("pharma_doc_type", "unknown") != "unknown"
        ]
        llm_call_count = pipeline.llm.complete.call_count
        non_keyword_count = len(docs) - len(keyword_resolved)

        # LLM must have been called at most once per non-keyword page (batched)
        import math
        max_expected_llm_calls = math.ceil(non_keyword_count / 5)
        assert llm_call_count <= max_expected_llm_calls, (
            f"{folder_name}: {llm_call_count} LLM calls for {non_keyword_count} "
            f"non-keyword pages (max expected {max_expected_llm_calls})"
        )


# ---------------------------------------------------------------------------
# VISUALIZATION — collect all metrics and render report
# ---------------------------------------------------------------------------


@pytest.mark.ocr_scan
@pytesseract_available
def test_ocr_accuracy_visualization(pipeline):
    """Generate artifacts/ocr_accuracy_report.png and two CSV files.

    Collects intrinsic OCR quality metrics (word density, vocabulary size,
    avg word length, keyword preservation, page coverage) and classification
    distribution per folder, then renders a 6-panel figure.
    """
    ARTIFACTS_DIR.mkdir(exist_ok=True)

    folder_records = []
    page_records = []

    for folder_name in IMAGE_FOLDERS:
        ocr_docs = pipeline.load_images(str(DOCS_DIR / folder_name))
        ocr_text = "\n".join(d.text for d in ocr_docs)

        # --- Per-folder intrinsic quality metrics ---
        stats = _intrinsic_stats(ocr_text)
        kw = keyword_preservation_rate(ocr_text, CRITICAL_KEYWORDS)
        coverage = len(ocr_docs) / max(IMAGE_FILE_COUNTS[folder_name], 1)

        # --- Classification via actual pipeline ---
        pipeline.llm.complete.return_value = MagicMock(text="unknown")
        annotated = pipeline._annotate_pharma_doc_types(ocr_docs)
        cat_counts = Counter(
            d.metadata.get("pharma_doc_type", "unknown") for d in annotated
        )

        record = {
            "folder": folder_name,
            "word_density": stats["word_density"],
            "avg_word_length": stats["avg_word_length"],
            "unique_words": stats["unique_words"],
            "words_per_page": stats["word_count"] / max(len(ocr_docs), 1),
            "kw_preservation": kw,
            "page_coverage": coverage,
            "total_pages": len(ocr_docs),
        }
        for cat in _PHARMA_DOC_CATEGORIES:
            record[f"cat_{cat}"] = cat_counts.get(cat, 0)
        folder_records.append(record)

        # --- Per-page stats ---
        for doc in annotated:
            page_stats = _intrinsic_stats(doc.text)
            page_records.append({
                "folder": folder_name,
                "page_number": doc.metadata.get("page_number", 0),
                "text_length": len(doc.text),
                "word_count": page_stats["word_count"],
                "word_density": page_stats["word_density"],
                "pharma_doc_type": doc.metadata.get("pharma_doc_type", "unknown"),
            })

    if not folder_records:
        pytest.skip("No image folders found — skipping visualization")

    df = pd.DataFrame(folder_records)
    df_pages = pd.DataFrame(page_records)

    # -----------------------------------------------------------------------
    # 6-panel figure
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle(
        "OCR Quality Report — Pharmaceutical Document Suite",
        fontsize=14,
        fontweight="bold",
    )
    ax = axes.flatten()

    labels = [f.replace("image_test_", "Folder ") for f in df["folder"].tolist()]
    x = np.arange(len(labels))
    bar_w = 0.55
    palette = plt.colormaps["tab10"].colors

    def _label_bars(axis, bars, vals, fmt="{:.2f}"):
        for bar, val in zip(bars, vals):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                fmt.format(val),
                ha="center", va="bottom", fontsize=8,
            )

    # Panel 0 — Words per page (avg)
    bars = ax[0].bar(x, df["words_per_page"], width=bar_w, color="#1565C0")
    ax[0].set_title("Average Words per Page (higher = richer OCR)")
    ax[0].set_xticks(x); ax[0].set_xticklabels(labels, rotation=15, ha="right")
    ax[0].set_ylabel("Words / page")
    _label_bars(ax[0], bars, df["words_per_page"], "{:.0f}")

    # Panel 1 — Word density per folder
    bars = ax[1].bar(x, df["word_density"], width=bar_w, color="#E65100")
    ax[1].axhline(WORD_DENSITY_THRESHOLD, color="red", linestyle="--", lw=1.5,
                  label=f"Threshold {WORD_DENSITY_THRESHOLD:.0%}")
    ax[1].set_title("Word Density (lower may indicate OCR noise)")
    ax[1].set_xticks(x); ax[1].set_xticklabels(labels, rotation=15, ha="right")
    ax[1].set_ylabel("Density")
    ax[1].set_ylim(0, 1.1)
    ax[1].legend(fontsize=9)
    _label_bars(ax[1], bars, df["word_density"], "{:.2%}")

    # Panel 2 — Avg word length per folder
    bars = ax[2].bar(x, df["avg_word_length"], width=bar_w, color="#2E7D32")
    ax[2].axhline(AVG_WORD_LEN_MIN, color="orange", linestyle="--", lw=1.2,
                  label=f"Min {AVG_WORD_LEN_MIN:.0f}")
    ax[2].axhline(AVG_WORD_LEN_MAX, color="red", linestyle="--", lw=1.2,
                  label=f"Max {AVG_WORD_LEN_MAX:.0f}")
    ax[2].set_title("Avg Word Length (plausible: 3–12 chars)")
    ax[2].set_xticks(x); ax[2].set_xticklabels(labels, rotation=15, ha="right")
    ax[2].set_ylabel("Chars")
    ax[2].legend(fontsize=9)
    _label_bars(ax[2], bars, df["avg_word_length"], "{:.2f}")

    # Panel 3 — Keyword preservation + page coverage (grouped bars)
    w2 = bar_w / 2
    ax[3].bar(x - w2 / 2, df["kw_preservation"], width=w2,
              label="Keyword Preservation", color="#1565C0")
    ax[3].bar(x + w2 / 2, df["page_coverage"], width=w2,
              label="Page Coverage", color="#6A1B9A")
    ax[3].axhline(KW_PRESERVATION_THRESHOLD, color="red", linestyle="--", lw=1.5,
                  label=f"KW threshold {KW_PRESERVATION_THRESHOLD:.0%}")
    ax[3].set_title("Keyword Preservation & Page Coverage")
    ax[3].set_xticks(x); ax[3].set_xticklabels(labels, rotation=15, ha="right")
    ax[3].set_ylabel("Rate")
    ax[3].set_ylim(0, 1.15)
    ax[3].legend(fontsize=9)

    # Panel 4 — Classification category distribution (stacked bars)
    active_cats = [
        c for c in _PHARMA_DOC_CATEGORIES
        if df[[f"cat_{c}"]].sum().sum() > 0
    ]
    bottom = np.zeros(len(df))
    for i, cat in enumerate(active_cats):
        vals = df[f"cat_{cat}"].values.astype(float)
        ax[4].bar(x, vals, bottom=bottom, label=cat,
                  color=palette[i % len(palette)], width=bar_w)
        bottom += vals
    ax[4].set_title("Classification Distribution per Folder")
    ax[4].set_xticks(x); ax[4].set_xticklabels(labels, rotation=15, ha="right")
    ax[4].set_ylabel("Page Count")
    ax[4].legend(fontsize=8, loc="upper right")

    # Panel 5 — Per-page text length scatter (coloured by folder)
    folder_colors = {f: palette[i % len(palette)] for i, f in enumerate(IMAGE_FOLDERS)}
    for folder_name in IMAGE_FOLDERS:
        subset = df_pages[df_pages["folder"] == folder_name]
        if subset.empty:
            continue
        ax[5].scatter(
            subset["page_number"], subset["text_length"],
            color=folder_colors[folder_name], alpha=0.7, s=35,
            label=folder_name.replace("image_test_", "Folder "),
        )
    ax[5].axhline(MIN_CHARS_PER_PAGE, color="red", linestyle="--", lw=1.5,
                  label=f"Min {MIN_CHARS_PER_PAGE} chars")
    ax[5].set_xlabel("Page Number")
    ax[5].set_ylabel("Text Length (chars)")
    ax[5].set_title("Per-Page Extracted Text Length")
    ax[5].legend(fontsize=8)

    plt.tight_layout()

    # -----------------------------------------------------------------------
    # Save artifacts
    # -----------------------------------------------------------------------
    chart_path = ARTIFACTS_DIR / "ocr_accuracy_report.png"
    fig.savefig(str(chart_path), dpi=150, bbox_inches="tight")
    plt.close(fig)

    csv_folder = ARTIFACTS_DIR / "ocr_accuracy_metrics.csv"
    csv_pages = ARTIFACTS_DIR / "ocr_accuracy_pages.csv"
    df.to_csv(str(csv_folder), index=False)
    df_pages.to_csv(str(csv_pages), index=False)

    assert chart_path.exists() and chart_path.stat().st_size > 0, (
        "ocr_accuracy_report.png was not written"
    )
    assert csv_folder.exists()
    assert csv_pages.exists()

    # Print summary to stdout so it is visible in CI logs
    summary_cols = [
        "folder", "word_density", "avg_word_length",
        "unique_words", "words_per_page", "kw_preservation", "page_coverage",
    ]
    print("\n=== OCR Quality Summary ===")
    print(df[summary_cols].to_string(index=False, float_format="{:.3f}".format))
    print(f"\nVisualization  : {chart_path}")
    print(f"Metrics CSV    : {csv_folder}")
    print(f"Per-page CSV   : {csv_pages}")

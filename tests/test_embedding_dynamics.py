"""
Pipeline Classification Dynamics Test
======================================
Wraps ``pipeline._classify_document`` (the real two-stage keyword + LLM
classifier) to measure how accuracy, speed, and the keyword-hit vs.
LLM-fallback ratio change as progressively more documents are classified
and stored in a temporary embedding store.

Heavy ML dependencies are NOT needed — ``conftest.py`` stubs them out, and
the LLM is replaced with a lightweight deterministic mock.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Pull in the real pipeline internals we are wrapping
# ---------------------------------------------------------------------------
from rag.pipeline import (  # noqa: E402
    _KEYWORD_MAP,
    _PHARMA_DOC_CATEGORIES,
    RAGPipeline,
)

DOCUMENT_TYPES: list[str] = [c for c in _PHARMA_DOC_CATEGORIES if c != "unknown"]
EMBEDDING_DIM: int = 384  # all-MiniLM-L6-v2 dimension


# ---------------------------------------------------------------------------
# Synthetic document corpus
# ---------------------------------------------------------------------------

# Documents whose headers contain keywords from _KEYWORD_MAP
# -> should be resolved by the keyword fast-path (no LLM call)
_KEYWORD_CORPUS: dict[str, list[str]] = {
    "cover_letter": [
        "Dear Supplier, please find enclosed the updated documentation for batch 2024-001.",
        "We herewith enclose the quality documentation requested by your team.",
        "Dear sir, herewith enclosed please find the shipment records.",
    ],
    "certificate_of_quality": [
        "Certificate of Quality — Batch No: 12345 — Product: Excipient X — Conforms to spec.",
        "Certificate of Analysis — sample id 9912 — all specifications met.",
        "C.O.A — Batch 77201 — complies with EP monograph.",
    ],
    "packaging_specification": [
        "Packaging Specification Rev. 3 — HDPE bottle 250 mL — closure torque 15 Nm.",
        "Pack spec for blister PVC/PVDC — dimensions 100 x 80 mm.",
        "Label specification V2 — printed black text on white background.",
    ],
    "bse_tse_declaration": [
        "BSE/TSE Declaration — no materials of bovine or ovine origin are used.",
        "Transmissible spongiform encephalopathy statement — raw materials: plant only.",
        "Bovine spongiform encephalopathy risk assessment — not applicable.",
    ],
    "material_description": [
        "Material Description — Chemical: Microcrystalline Cellulose — CAS 9004-34-6.",
        "Product Description — Polysorbate 80 — function: emulsifier — grade: NF.",
        "Substance Description — lactose monohydrate — particle size D90 < 150 um.",
    ],
    "supplier_qualification": [
        "Supplier Qualification Report — audit date 2023-05 — status: Approved.",
        "Vendor Qualification — site inspection score 94/100 — ISO 9001 certified.",
        "Audit Report — Plant A — GMP compliant per EU Directive 2003/94/EC.",
    ],
    "chain_of_custody": [
        "Chain of Custody — transferred from Manufacturer X to Distributor Y.",
        "Chain-of-custody document — sample sealed, temperature logged.",
        "Custody Transfer Record — batch 20240310 — cold chain maintained.",
    ],
}

# Documents with no keyword signals -> fall through to LLM path
_AMBIGUOUS_CORPUS: dict[str, list[str]] = {
    "cover_letter": [
        "Attached are the relevant documents for your review. Please acknowledge receipt.",
        "Please find the requested files in this transmission. Contact us for queries.",
    ],
    "certificate_of_quality": [
        "Batch 12345 tested against specification. All results within acceptance criteria.",
        "Product lot 77201 released by QA department per internal procedure QP-04.",
    ],
    "packaging_specification": [
        "Primary container dimensions verified against approved drawing PK-0023 Rev 2.",
        "Blister configuration changed from PVC to PETG per change control CC-112.",
    ],
    "bse_tse_declaration": [
        "No animal-derived raw materials are used in the manufacture of this product.",
        "All excipients are of synthetic origin — no prion contamination risk identified.",
    ],
    "material_description": [
        "Molecular weight: 342.30 g/mol. Solubility in water: 210 g/L at 25 degrees C.",
        "Functional excipient used as a binder in solid oral dosage forms.",
    ],
    "supplier_qualification": [
        "Site audit completed 2023-05. Findings: 0 critical, 2 minor. Status: Approved.",
        "Quality agreement signed 2022-11. Next review due 2025-11. Status: active.",
    ],
    "chain_of_custody": [
        "Sample sealed by QC inspector, transferred under continuous cold chain supervision.",
        "Temperature logger attached. Min: 2.1C, Max: 7.9C during transit. No excursions.",
    ],
}


# ---------------------------------------------------------------------------
# Mock LLM
# ---------------------------------------------------------------------------

class _LLMResponse:
    def __init__(self, text: str) -> None:
        self.text = text


class MockLLM:
    """
    Returns the correct category for any document whose first 60 characters
    appear as a key in *text_to_label*.  Counts every ``.complete()`` call
    so tests can assert on LLM-path vs. keyword-path invocation counts.
    """

    def __init__(self, text_to_label: dict[str, str]) -> None:
        self._map = text_to_label
        self.call_count = 0

    def complete(self, prompt: str) -> _LLMResponse:
        self.call_count += 1
        for snippet, label in self._map.items():
            if snippet in prompt:
                return _LLMResponse(label)
        return _LLMResponse("unknown")


def _build_llm_map() -> dict[str, str]:
    """
    Map first-60-char snippet -> label for every document that will reach
    the LLM fallback path (ambiguous docs with no keyword_map signal).
    """
    mapping: dict[str, str] = {}
    for label, texts in _AMBIGUOUS_CORPUS.items():
        for text in texts:
            mapping[text[:60]] = label
    return mapping


# ---------------------------------------------------------------------------
# MinimalClassifier — thin shell that hosts the real pipeline methods
# ---------------------------------------------------------------------------

class MinimalClassifier:
    """
    Exposes ``_classify_document`` and ``_parse_category`` from the real
    ``RAGPipeline`` class while substituting the LLM with a mock.

    The unbound method call ``RAGPipeline._classify_document(self, text)``
    uses Python's normal attribute lookup so ``self.llm`` and
    ``self._parse_category`` are resolved against this class.
    """

    def __init__(self, llm: MockLLM) -> None:
        self.llm = llm

    def classify_document(self, text: str) -> str:
        return RAGPipeline._classify_document(self, text)  # type: ignore[arg-type]

    @staticmethod
    def _parse_category(text: str) -> str:
        return RAGPipeline._parse_category(text)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# ClassificationStore — temporary database of classified + embedded docs
# ---------------------------------------------------------------------------

class ClassificationStore:
    """
    Accumulates classified documents together with synthetic embedding vectors.
    Both artefacts are persisted to a temp directory after every ``add()`` call.
    """

    _EMBEDDINGS_FILE = "embeddings.npy"
    _META_FILE = "metadata.json"

    def __init__(self, store_dir: str) -> None:
        self.store_dir = Path(store_dir)
        self._embeddings: list[np.ndarray] = []
        self._metadata: list[dict] = []

    def add(self, text: str, predicted: str, true_label: str, embedding: np.ndarray) -> None:
        self._embeddings.append(embedding.astype(np.float32))
        self._metadata.append(
            {
                "text_snippet": text[:80],
                "predicted": predicted,
                "true": true_label,
                "correct": predicted == true_label,
            }
        )
        self._persist()

    def _persist(self) -> None:
        np.save(str(self.store_dir / self._EMBEDDINGS_FILE), np.array(self._embeddings, dtype=np.float32))
        with open(self.store_dir / self._META_FILE, "w", encoding="utf-8") as fh:
            json.dump(self._metadata, fh, indent=2)

    def accuracy(self) -> float:
        if not self._metadata:
            return 0.0
        return sum(m["correct"] for m in self._metadata) / len(self._metadata) * 100

    def accuracy_for(self, doc_type: str) -> float:
        subset = [m for m in self._metadata if m["true"] == doc_type]
        if not subset:
            return 0.0
        return sum(m["correct"] for m in subset) / len(subset) * 100

    def __len__(self) -> int:
        return len(self._metadata)


# ---------------------------------------------------------------------------
# Synthetic embeddings (centroid + Gaussian noise)
# ---------------------------------------------------------------------------

def _centroid(doc_type: str) -> np.ndarray:
    seed = sum(i * ord(c) for i, c in enumerate(doc_type, 1))
    v = np.random.default_rng(seed).standard_normal(EMBEDDING_DIM).astype(np.float32)
    return v / np.linalg.norm(v)


def make_embedding(rng: np.random.Generator, doc_type: str, noise: float = 0.08) -> np.ndarray:
    v = _centroid(doc_type) + rng.standard_normal(EMBEDDING_DIM).astype(np.float32) * noise
    return v / np.linalg.norm(v)


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

_BLUE = "#2196F3"
_RED = "#F44336"
_ORANGE = "#FF7043"
_GREEN = "#4CAF50"
_AMBER = "#FF9800"


def _plot_classification_dynamics(df: pd.DataFrame, output_path: Path) -> None:
    """Three-panel chart: accuracy, latency, and keyword vs. LLM path split."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        "Pipeline Classification Dynamics\n(pipeline._classify_document — keyword path vs. LLM fallback)",
        fontsize=12,
        fontweight="bold",
    )

    x = df["n_classified"]

    # Panel 1: Accuracy
    ax = axes[0]
    ax.plot(x, df["accuracy_pct"], "o-", color=_BLUE, lw=2, ms=6, label="Accuracy (%)")
    ax.fill_between(x, df["accuracy_pct"], alpha=0.12, color=_BLUE)
    ax.set_xlabel("Documents Classified")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Classification Accuracy")
    ax.set_ylim(0, 107)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    for pos in (0, -1):
        ax.annotate(
            f"{df['accuracy_pct'].iloc[pos]:.1f}%",
            xy=(x.iloc[pos], df["accuracy_pct"].iloc[pos]),
            xytext=(6, -14),
            textcoords="offset points",
            fontsize=8,
            color=_BLUE,
        )

    # Panel 2: Latency
    ax = axes[1]
    ax.plot(x, df["avg_latency_ms"], "s-", color=_RED, lw=2, ms=6, label="Avg latency (ms)")
    ax.fill_between(x, df["avg_latency_ms"], df["p95_latency_ms"], alpha=0.18, color=_RED, label="P95 band")
    ax.plot(x, df["p95_latency_ms"], "^--", color=_ORANGE, lw=1.5, ms=5, label="P95 latency (ms)")
    ax.set_xlabel("Documents Classified")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Classification Speed")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # Panel 3: Keyword vs. LLM path split
    ax = axes[2]
    ax.stackplot(
        x,
        df["keyword_hit_pct"],
        df["llm_call_pct"],
        labels=["Keyword path (%)", "LLM fallback (%)"],
        colors=[_GREEN, _AMBER],
        alpha=0.82,
    )
    ax.set_xlabel("Documents Classified")
    ax.set_ylabel("Share of Classifications (%)")
    ax.set_title("Keyword vs. LLM Path Split")
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.2)
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_per_category_accuracy(df: pd.DataFrame, output_path: Path) -> None:
    """Multi-line chart: per-category accuracy vs. documents classified."""
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_title(
        "Per-Category Classification Accuracy vs. Documents Classified",
        fontsize=13,
        fontweight="bold",
    )
    colors = matplotlib.colormaps["tab10"](np.linspace(0, 1, len(DOCUMENT_TYPES)))
    x = df["n_classified"]

    for doc_type, color in zip(DOCUMENT_TYPES, colors):
        col = f"acc_{doc_type}"
        if col in df.columns:
            ax.plot(x, df[col], "o-", color=color, lw=2, ms=5,
                    label=doc_type.replace("_", " ").title())

    ax.set_xlabel("Documents Classified")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(-5, 110)
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=9)
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def temp_store_dir():
    """Yield a fresh temp directory; remove it unconditionally afterwards."""
    d = tempfile.mkdtemp(prefix="rag_classify_test_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPipelineClassificationDynamics:
    """
    Wraps ``pipeline._classify_document`` to measure accuracy, speed, and
    path distribution (keyword vs. LLM) as the temporary embedding store grows.
    """

    # -----------------------------------------------------------------------
    # Test 1: Overall accuracy and speed
    # -----------------------------------------------------------------------

    def test_accuracy_and_speed_over_growing_store(self, temp_store_dir: str, tmp_path: Path) -> None:
        """
        Progressively classifies documents from both corpora (keyword-rich and
        ambiguous) using the real ``_classify_document`` two-stage logic with a
        mock LLM.  After each batch the cumulative accuracy, latency, and
        keyword/LLM path counts are recorded.

        Assertions
        ----------
        * Final accuracy >= 85%
        * LLM path is used for ambiguous and "unknown" docs (call_count > 0)
        * Keyword path is used for keyword-rich docs
        * Output PNG written to tmp_path/ and mirrored to artifacts/

        Graph: three panels — accuracy, latency, keyword vs. LLM split.
        """
        rng = np.random.default_rng(42)
        mock_llm = MockLLM(_build_llm_map())
        classifier = MinimalClassifier(mock_llm)
        store = ClassificationStore(temp_store_dir)

        # Build flat doc list: all keyword docs then all ambiguous docs per type
        all_docs: list[tuple[str, str]] = []
        for doc_type in DOCUMENT_TYPES:
            for text in _KEYWORD_CORPUS.get(doc_type, []):
                all_docs.append((text, doc_type))
            for text in _AMBIGUOUS_CORPUS.get(doc_type, []):
                all_docs.append((text, doc_type))

        # Shuffle for a realistic interleaved stream
        indices = rng.permutation(len(all_docs))
        all_docs = [all_docs[i] for i in indices]

        BATCH_SIZE = 4
        results: list[dict] = []

        for batch_start in range(0, len(all_docs), BATCH_SIZE):
            batch = all_docs[batch_start : batch_start + BATCH_SIZE]
            latencies_ms: list[float] = []

            for text, true_label in batch:
                t0 = time.perf_counter()
                predicted = classifier.classify_document(text)
                latencies_ms.append((time.perf_counter() - t0) * 1_000)

                store.add(text, predicted, true_label, make_embedding(rng, true_label))

            n = len(store)
            n_llm = mock_llm.call_count
            n_keyword = n - n_llm

            results.append(
                {
                    "batch": batch_start // BATCH_SIZE + 1,
                    "n_classified": n,
                    "accuracy_pct": store.accuracy(),
                    "avg_latency_ms": float(np.mean(latencies_ms)),
                    "p95_latency_ms": float(np.percentile(latencies_ms, 95)),
                    "keyword_hit_pct": n_keyword / n * 100,
                    "llm_call_pct": n_llm / n * 100,
                }
            )

        df = pd.DataFrame(results)

        # ── Assertions ──────────────────────────────────────────────────
        final_acc = df["accuracy_pct"].iloc[-1]
        assert final_acc >= 95.0, f"Final accuracy {final_acc:.1f}% is too low"
        assert mock_llm.call_count > 0, "LLM should have been invoked for ambiguous docs"
        assert len(store) - mock_llm.call_count > 0, "Keyword path should have fired at least once"
        assert len(store) == len(all_docs)

        # ── Output graph ────────────────────────────────────────────────
        chart_path = tmp_path / "classification_dynamics.png"
        _plot_classification_dynamics(df, chart_path)
        assert chart_path.exists() and chart_path.stat().st_size > 0

        artifacts_dir = Path(__file__).parent.parent / "artifacts"
        if artifacts_dir.exists():
            shutil.copy(chart_path, artifacts_dir / "classification_dynamics.png")

        print(
            f"\n[Pipeline Classification]  n={len(store)}"
            f"  accuracy={final_acc:.1f}%"
            f"  keyword_hits={len(store) - mock_llm.call_count}"
            f"  llm_calls={mock_llm.call_count}"
        )

    # -----------------------------------------------------------------------
    # Test 2: Per-category accuracy breakdown
    # -----------------------------------------------------------------------

    def test_per_category_accuracy_over_growing_store(self, temp_store_dir: str, tmp_path: Path) -> None:
        """
        Tracks classification accuracy per document type as the store grows,
        revealing which categories rely on the keyword fast-path and which
        need the LLM fallback.

        Assertions
        ----------
        * Every category achieves >= 50% accuracy by the final batch
          (keyword-matched categories will hit 100%; ambiguous-only categories
          must rely on the mock LLM being consistent)
        * Output PNG written to tmp_path/ and mirrored to artifacts/

        Graph: one line per document type.
        """
        rng = np.random.default_rng(99)
        mock_llm = MockLLM(_build_llm_map())
        classifier = MinimalClassifier(mock_llm)
        store = ClassificationStore(temp_store_dir)

        all_docs: list[tuple[str, str]] = []
        for doc_type in DOCUMENT_TYPES:
            for text in _KEYWORD_CORPUS.get(doc_type, []):
                all_docs.append((text, doc_type))
            for text in _AMBIGUOUS_CORPUS.get(doc_type, []):
                all_docs.append((text, doc_type))

        indices = rng.permutation(len(all_docs))
        all_docs = [all_docs[i] for i in indices]

        BATCH_SIZE = 5
        results: list[dict] = []

        for batch_start in range(0, len(all_docs), BATCH_SIZE):
            batch = all_docs[batch_start : batch_start + BATCH_SIZE]
            for text, true_label in batch:
                predicted = classifier.classify_document(text)
                store.add(text, predicted, true_label, make_embedding(rng, true_label))

            row: dict = {"n_classified": len(store)}
            for doc_type in DOCUMENT_TYPES:
                row[f"acc_{doc_type}"] = store.accuracy_for(doc_type)
            results.append(row)

        df = pd.DataFrame(results)

        # ── Assertions ──────────────────────────────────────────────────
        for doc_type in DOCUMENT_TYPES:
            final_acc = df[f"acc_{doc_type}"].iloc[-1]
            assert final_acc >= 50.0, (
                f"Category '{doc_type}' final accuracy {final_acc:.1f}% is too low"
            )

        # ── Output graph ────────────────────────────────────────────────
        chart_path = tmp_path / "per_category_accuracy.png"
        _plot_per_category_accuracy(df, chart_path)
        assert chart_path.exists() and chart_path.stat().st_size > 0

        artifacts_dir = Path(__file__).parent.parent / "artifacts"
        if artifacts_dir.exists():
            shutil.copy(chart_path, artifacts_dir / "per_category_accuracy.png")

        print(f"\n[Per-category]  n={len(store)}")
        for doc_type in DOCUMENT_TYPES:
            print(f"  {doc_type:30s}: {df[f'acc_{doc_type}'].iloc[-1]:.0f}%")

    # -----------------------------------------------------------------------
    # Test 3: Keyword vs LLM path counts per category
    # -----------------------------------------------------------------------

    def test_keyword_vs_llm_path_per_category(self, temp_store_dir: str) -> None:
        """
        Verifies that documents containing keyword-map signals are resolved
        by the fast path (no LLM call) and documents without them use the LLM.

        Assertions
        ----------
        * Every keyword-corpus doc resolves without an LLM call.
        * Every ambiguous-corpus doc triggers exactly one LLM call.
        """
        mock_llm = MockLLM(_build_llm_map())
        classifier = MinimalClassifier(mock_llm)
        store = ClassificationStore(temp_store_dir)
        rng = np.random.default_rng(0)

        expected_llm_calls = 0

        for doc_type in DOCUMENT_TYPES:
            # Keyword corpus — all named-category docs have keyword signals
            for text in _KEYWORD_CORPUS.get(doc_type, []):
                llm_before = mock_llm.call_count
                predicted = classifier.classify_document(text)
                store.add(text, predicted, doc_type, make_embedding(rng, doc_type))

                assert mock_llm.call_count == llm_before, (
                    f"Keyword doc should NOT trigger LLM: {text[:50]}"
                )

            # Ambiguous corpus — all should use LLM
            for text in _AMBIGUOUS_CORPUS.get(doc_type, []):
                expected_llm_calls += 1
                predicted = classifier.classify_document(text)
                store.add(text, predicted, doc_type, make_embedding(rng, doc_type))
                assert mock_llm.call_count == expected_llm_calls, (
                    f"Ambiguous doc should trigger LLM: {text[:50]}"
                )

        assert mock_llm.call_count == expected_llm_calls

    # -----------------------------------------------------------------------
    # Test 4: Store persistence round-trip
    # -----------------------------------------------------------------------

    def test_store_persists_to_disk(self, temp_store_dir: str) -> None:
        """
        Classifies a small set of documents and verifies the store serialises
        both embeddings and metadata to disk correctly.

        Assertions
        ----------
        * embeddings.npy and metadata.json exist after classification
        * Reloaded embedding count matches
        * Reloaded metadata preserves predicted/true labels and correct flag
        """
        mock_llm = MockLLM(_build_llm_map())
        classifier = MinimalClassifier(mock_llm)
        store = ClassificationStore(temp_store_dir)
        rng = np.random.default_rng(7)

        docs = [
            (_KEYWORD_CORPUS["cover_letter"][0], "cover_letter"),
            (_AMBIGUOUS_CORPUS["certificate_of_quality"][0], "certificate_of_quality"),
            (_KEYWORD_CORPUS["bse_tse_declaration"][0], "bse_tse_declaration"),
        ]
        for text, true_label in docs:
            predicted = classifier.classify_document(text)
            store.add(text, predicted, true_label, make_embedding(rng, true_label))

        emb_path = Path(temp_store_dir) / ClassificationStore._EMBEDDINGS_FILE
        meta_path = Path(temp_store_dir) / ClassificationStore._META_FILE

        assert emb_path.exists(), "embeddings.npy not written"
        assert meta_path.exists(), "metadata.json not written"

        loaded_embs = np.load(str(emb_path))
        with open(meta_path, encoding="utf-8") as fh:
            loaded_meta = json.load(fh)

        assert loaded_embs.shape == (len(docs), EMBEDDING_DIM)
        assert len(loaded_meta) == len(docs)

        for entry in loaded_meta:
            assert "predicted" in entry
            assert "true" in entry
            assert "correct" in entry
            assert entry["correct"] == (entry["predicted"] == entry["true"])

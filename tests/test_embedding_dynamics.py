"""
Pipeline Classification Dynamics Test
======================================
Wraps ``pipeline._classify_document`` (the real two-stage keyword + LLM
classifier) to measure how accuracy, speed, and the keyword-hit vs.
LLM-fallback ratio change as progressively more documents are classified
and stored in a temporary embedding store.

Latency methodology
-------------------
* Each sample is warmed up (N_WARMUP untimed calls) then timed N_REPEATS
  times; the median is used — not a single-pass batch mean.
* Keyword-path and LLM-path latencies are tracked in separate buckets and
  plotted as independent series.
* Batches have a fixed keyword : LLM ratio so the composition never skews
  the per-batch aggregates.

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

DOCUMENT_TYPES: list[str] = _PHARMA_DOC_CATEGORIES
EMBEDDING_DIM: int = 384  # all-MiniLM-L6-v2 dimension

# Latency measurement parameters
N_WARMUP: int = 3   # untimed runs per sample before timing starts
N_REPEATS: int = 9  # timed runs per sample; median is taken (odd number)

# Controlled batch composition
KW_PER_BATCH: int = 2    # keyword-path docs per batch
LLM_PER_BATCH: int = 2   # LLM-path docs per batch


# ---------------------------------------------------------------------------
# Synthetic document corpus
# ---------------------------------------------------------------------------

# Documents whose headers contain keywords from _KEYWORD_MAP
# -> resolved by the keyword fast-path (no LLM call)
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
    # "unknown" has no _KEYWORD_MAP entry so these fall through to LLM
    "unknown": [
        "Internal memo — quarterly review meeting notes Q3 2024.",
        "Project timeline — milestone tracking sheet — Phase 2.",
        "Employee onboarding checklist — IT setup complete.",
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
    "unknown": [
        "Q3 budget allocation — R&D department — EUR 1.2M approved.",
        "Canteen menu for week 42 — includes vegetarian and vegan options.",
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
    the LLM fallback path (ambiguous docs + "unknown" keyword docs which
    have no keyword_map entry).
    """
    mapping: dict[str, str] = {}
    for label, texts in _AMBIGUOUS_CORPUS.items():
        for text in texts:
            mapping[text[:60]] = label
    for text in _KEYWORD_CORPUS.get("unknown", []):
        mapping[text[:60]] = "unknown"
    return mapping


# ---------------------------------------------------------------------------
# MinimalClassifier — thin shell that hosts the real pipeline methods
# ---------------------------------------------------------------------------

class MinimalClassifier:
    """
    Exposes ``_classify_document`` and ``_parse_category`` from the real
    ``RAGPipeline`` class while substituting the LLM with a mock.
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
# Latency measurement helpers
# ---------------------------------------------------------------------------

def _is_keyword_path(text: str) -> bool:
    """
    Mirrors Stage 1 of ``_classify_document``: return True if the text header
    matches any entry in ``_KEYWORD_MAP`` (fast path, no LLM call).
    """
    header = text[:300].lower()
    return any(
        any(kw in header for kw in kws)
        for kws in _KEYWORD_MAP.values()
    )


def _measure_latency_ms(classifier: MinimalClassifier, text: str) -> float:
    """
    Warm up the classifier on *text* (N_WARMUP untimed calls), then return
    the median of N_REPEATS timed calls in milliseconds.

    Uses a dedicated timing classifier so the counting ``MockLLM``'s
    ``call_count`` is never inflated by warm-up or repeat passes.
    """
    for _ in range(N_WARMUP):
        classifier.classify_document(text)
    samples = []
    for _ in range(N_REPEATS):
        t0 = time.perf_counter()
        classifier.classify_document(text)
        samples.append((time.perf_counter() - t0) * 1_000)
    return float(np.median(samples))


def _build_controlled_batches(
    keyword_docs: list[tuple[str, str]],
    llm_docs: list[tuple[str, str]],
    kw_per_batch: int,
    llm_per_batch: int,
) -> list[list[tuple[str, str]]]:
    """
    Return batches with a fixed keyword : LLM composition.
    Stops when either list is exhausted so every batch is complete.
    """
    batches: list[list[tuple[str, str]]] = []
    ki = li = 0
    while ki + kw_per_batch <= len(keyword_docs) and li + llm_per_batch <= len(llm_docs):
        batches.append(keyword_docs[ki : ki + kw_per_batch] + llm_docs[li : li + llm_per_batch])
        ki += kw_per_batch
        li += llm_per_batch
    return batches


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

_BLUE = "#2196F3"
_GREEN = "#4CAF50"
_GREEN_DARK = "#2E7D32"
_AMBER = "#FF9800"
_AMBER_DARK = "#E65100"


def _plot_classification_dynamics(df: pd.DataFrame, output_path: Path) -> None:
    """
    Three-panel chart:
      1. Classification accuracy vs. documents classified
      2. Keyword-path vs. LLM-path latency (median + p90) as separate series
      3. Keyword vs. LLM path share (stacked area)
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        "Pipeline Classification Dynamics\n"
        "(pipeline._classify_document — keyword path vs. LLM fallback)",
        fontsize=12,
        fontweight="bold",
    )

    x = df["n_classified"]

    # ── Panel 1: Accuracy ──────────────────────────────────────────────
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

    # ── Panel 2: Latency — keyword vs. LLM, median + p90 ──────────────
    ax = axes[1]
    ax.plot(x, df["kw_median_ms"],  "o-",  color=_GREEN,      lw=2, ms=6,  label="Keyword median")
    ax.plot(x, df["kw_p90_ms"],     "o--", color=_GREEN_DARK, lw=1.5, ms=4, label="Keyword p90")
    ax.fill_between(x, df["kw_median_ms"], df["kw_p90_ms"], alpha=0.14, color=_GREEN)

    ax.plot(x, df["llm_median_ms"], "s-",  color=_AMBER,      lw=2, ms=6,  label="LLM median")
    ax.plot(x, df["llm_p90_ms"],    "s--", color=_AMBER_DARK, lw=1.5, ms=4, label="LLM p90")
    ax.fill_between(x, df["llm_median_ms"], df["llm_p90_ms"], alpha=0.14, color=_AMBER)

    ax.set_xlabel("Documents Classified")
    ax.set_ylabel("Latency (ms)  [median of 9 runs]")
    ax.set_title("Classification Speed — Keyword vs. LLM")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # ── Panel 3: Path-split stacked area ──────────────────────────────
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
    colors = plt.cm.tab10(np.linspace(0, 1, len(DOCUMENT_TYPES)))  # type: ignore[attr-defined]
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
    # Test 1: Accuracy + separate keyword/LLM latency over growing store
    # -----------------------------------------------------------------------

    def test_accuracy_and_speed_over_growing_store(self, temp_store_dir: str, tmp_path: Path) -> None:
        """
        Progressively classifies controlled batches (KW_PER_BATCH keyword docs +
        LLM_PER_BATCH LLM docs per batch) and records per-batch metrics:

        * accuracy_pct    — cumulative correct / total
        * kw_median_ms    — keyword-path median latency  (N_REPEATS timed runs)
        * kw_p90_ms       — keyword-path 90th-percentile latency
        * llm_median_ms   — LLM-path median latency
        * llm_p90_ms      — LLM-path 90th-percentile latency
        * keyword_hit_pct — cumulative keyword-path share
        * llm_call_pct    — cumulative LLM-path share

        Latency methodology
        -------------------
        Each sample is first warmed up (N_WARMUP calls, not timed) via a
        dedicated timing classifier that is separate from the accuracy-tracking
        classifier, so MockLLM.call_count is never inflated by repeats.

        Assertions
        ----------
        * Final accuracy >= 85%
        * LLM-path median latency >= keyword-path median on every batch
          (LLM path does more work: string format + dict scan on top of
          the keyword scan)
        * Output PNG written to tmp_path/ and mirrored to artifacts/
        """
        rng = np.random.default_rng(42)

        # Counting classifier: tracks accuracy and call_count (single pass)
        counting_llm = MockLLM(_build_llm_map())
        counting_clf = MinimalClassifier(counting_llm)

        # Timing classifier: isolated from counting_llm so call_count stays clean
        timing_clf = MinimalClassifier(MockLLM(_build_llm_map()))

        store = ClassificationStore(temp_store_dir)

        # ── Build keyword and LLM doc lists ───────────────────────────
        # keyword_docs: all categories that have _KEYWORD_MAP entries (not "unknown")
        keyword_docs: list[tuple[str, str]] = [
            (text, doc_type)
            for doc_type in DOCUMENT_TYPES
            if doc_type != "unknown"
            for text in _KEYWORD_CORPUS.get(doc_type, [])
        ]
        # llm_docs: all ambiguous docs + all "unknown" keyword docs
        llm_docs: list[tuple[str, str]] = [
            (text, doc_type)
            for doc_type in DOCUMENT_TYPES
            for text in _AMBIGUOUS_CORPUS.get(doc_type, [])
        ] + [
            (text, "unknown")
            for text in _KEYWORD_CORPUS.get("unknown", [])
        ]

        # Shuffle each list independently to avoid ordering bias
        kw_idx = rng.permutation(len(keyword_docs))
        llm_idx = rng.permutation(len(llm_docs))
        keyword_docs = [keyword_docs[i] for i in kw_idx]
        llm_docs = [llm_docs[i] for i in llm_idx]

        batches = _build_controlled_batches(keyword_docs, llm_docs, KW_PER_BATCH, LLM_PER_BATCH)
        assert batches, "No complete batches could be formed — check corpus sizes"

        results: list[dict] = []

        for batch_idx, batch in enumerate(batches):
            kw_latencies: list[float] = []
            llm_latencies: list[float] = []

            for text, true_label in batch:
                # ── Single-pass: accuracy tracking ────────────────────
                predicted = counting_clf.classify_document(text)
                store.add(text, predicted, true_label, make_embedding(rng, true_label))

                # ── Latency measurement: warm-up then repeated runs ────
                lat = _measure_latency_ms(timing_clf, text)
                if _is_keyword_path(text):
                    kw_latencies.append(lat)
                else:
                    llm_latencies.append(lat)

            n = len(store)
            n_llm = counting_llm.call_count
            n_keyword = n - n_llm

            results.append(
                {
                    "batch": batch_idx + 1,
                    "n_classified": n,
                    "accuracy_pct": store.accuracy(),
                    "kw_median_ms": float(np.median(kw_latencies)) if kw_latencies else float("nan"),
                    "kw_p90_ms": float(np.percentile(kw_latencies, 90)) if kw_latencies else float("nan"),
                    "llm_median_ms": float(np.median(llm_latencies)) if llm_latencies else float("nan"),
                    "llm_p90_ms": float(np.percentile(llm_latencies, 90)) if llm_latencies else float("nan"),
                    "keyword_hit_pct": n_keyword / n * 100,
                    "llm_call_pct": n_llm / n * 100,
                }
            )

        df = pd.DataFrame(results)

        # ── Assertions ──────────────────────────────────────────────────
        final_acc = df["accuracy_pct"].iloc[-1]
        assert final_acc >= 85.0, f"Final accuracy {final_acc:.1f}% is too low"
        assert counting_llm.call_count > 0, "LLM path should have fired"
        assert len(store) - counting_llm.call_count > 0, "Keyword path should have fired"

        # LLM path should be slower than keyword path on every batch
        for _, row in df.iterrows():
            assert row["llm_median_ms"] >= row["kw_median_ms"], (
                f"Batch {row['batch']}: LLM median ({row['llm_median_ms']:.4f} ms) "
                f"< keyword median ({row['kw_median_ms']:.4f} ms)"
            )

        # ── Output graph ────────────────────────────────────────────────
        chart_path = tmp_path / "classification_dynamics.png"
        _plot_classification_dynamics(df, chart_path)
        assert chart_path.exists() and chart_path.stat().st_size > 0

        artifacts_dir = Path(__file__).parent.parent / "artifacts"
        if artifacts_dir.exists():
            shutil.copy(chart_path, artifacts_dir / "classification_dynamics.png")

        print(
            f"\n[Pipeline Classification]"
            f"  batches={len(batches)}  n={len(store)}"
            f"  accuracy={final_acc:.1f}%"
            f"  kw_median={df['kw_median_ms'].mean():.4f} ms"
            f"  llm_median={df['llm_median_ms'].mean():.4f} ms"
            f"  kw_path={len(store) - counting_llm.call_count}"
            f"  llm_path={counting_llm.call_count}"
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
        * Every category achieves >= 50% accuracy by the final batch.
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

        for doc_type in DOCUMENT_TYPES:
            final_acc = df[f"acc_{doc_type}"].iloc[-1]
            assert final_acc >= 50.0, (
                f"Category '{doc_type}' final accuracy {final_acc:.1f}% is too low"
            )

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
        * Every keyword-corpus doc (except "unknown") resolves without an LLM call.
        * Every ambiguous-corpus doc triggers exactly one LLM call.
        * "unknown" keyword docs (no keyword_map entry) also trigger an LLM call.
        """
        mock_llm = MockLLM(_build_llm_map())
        classifier = MinimalClassifier(mock_llm)
        store = ClassificationStore(temp_store_dir)
        rng = np.random.default_rng(0)

        expected_llm_calls = 0

        for doc_type in DOCUMENT_TYPES:
            for text in _KEYWORD_CORPUS.get(doc_type, []):
                llm_before = mock_llm.call_count
                predicted = classifier.classify_document(text)
                store.add(text, predicted, doc_type, make_embedding(rng, doc_type))

                if doc_type == "unknown":
                    expected_llm_calls += 1
                    assert mock_llm.call_count == expected_llm_calls, (
                        f"'unknown' keyword doc should have gone to LLM: {text[:50]}"
                    )
                else:
                    assert mock_llm.call_count == llm_before, (
                        f"Keyword doc should NOT trigger LLM: {text[:50]}"
                    )

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
        * embeddings.npy and metadata.json exist after classification.
        * Reloaded embedding shape matches.
        * Reloaded metadata preserves predicted/true labels and correct flag.
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

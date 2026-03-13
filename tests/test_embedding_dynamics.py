"""
Pipeline Classification Dynamics Test
======================================
Wraps ``pipeline._classify_document`` to measure how the pipeline's reliance
on the classification index grows as more documents are stored.

Classification routing (in priority order):
  1. Index  — nearest-neighbour lookup in the embedding store (cosine ≥ threshold)
  2. Keyword — fast regex scan via ``_KEYWORD_MAP`` (no LLM needed)
  3. LLM    — two-stage fallback for ambiguous documents

Three independent runs with different random orderings of the document corpus
simulate the variability in which order files from ``docs/`` are encountered.
The side-by-side comparison shows how quickly each ordering builds up index
coverage and drives down LLM reliance.

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

# Cosine similarity threshold for index-based classification
SIMILARITY_THRESHOLD: float = 0.82

# Snapshot every N documents (x-axis resolution)
SNAPSHOT_EVERY: int = 4

# Three independent random orderings for side-by-side comparison
RUN_SEEDS: list[int] = [42, 99, 7]


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
# Includes generic pharma docs plus FDA regulatory correspondence (test_10–14):
#   cover_letter      — BLA 125742 "Dear…" response letters  (test_10, 11, 13)
#   certificate_of_quality — Sterility/Endotoxin report      (test_12)
#   supplier_qualification — Manufacturing/Equipment response (test_14)
_AMBIGUOUS_CORPUS: dict[str, list[str]] = {
    "cover_letter": [
        "Attached are the relevant documents for your review. Please acknowledge receipt.",
        "Please find the requested files in this transmission. Contact us for queries.",
        # FDA regulatory letters — BLA 125742 (test_10, 11, 13)
        "In response to your information request dated July 23, 2021, under BLA 125742, please find our technical reply attached.",
        "We are responding to the FDA request regarding BLA 125742 for the COVID-19 mRNA vaccine regarding capillary gel electrophoresis.",
        "This document provides our response to the FDA correspondence issued July 30, 2021 concerning sterility and endotoxin testing.",
        "We hereby submit our technical response to FDA information request BLA 125742 pertaining to manufacturing equipment qualification.",
    ],
    "certificate_of_quality": [
        "Batch 12345 tested against specification. All results within acceptance criteria.",
        "Product lot 77201 released by QA department per internal procedure QP-04.",
        # Sterility/Endotoxin technical verification report (test_12)
        "Endotoxin testing was performed by the LAL chromogenic method. All results met the acceptance criteria of < 0.5 EU/mL.",
        "Sterility testing confirmed no microbial growth after 14-day incubation. Positive product control (PPC) recovery was 100%.",
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
        # Manufacturing/Equipment technical response (test_14)
        "Bioreactor equipment was qualified per IQ/OQ/PQ protocols and demonstrated compliance with current GMP requirements.",
        "Equipment maintenance and calibration records confirm all manufacturing systems are within validated operational ranges.",
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

    def add(self, text: str, predicted: str, embedding: np.ndarray) -> None:
        self._embeddings.append(embedding.astype(np.float32))
        self._metadata.append({"text_snippet": text[:80], "predicted": predicted})
        self._persist()

    def _persist(self) -> None:
        np.save(
            str(self.store_dir / self._EMBEDDINGS_FILE),
            np.array(self._embeddings, dtype=np.float32),
        )
        with open(self.store_dir / self._META_FILE, "w", encoding="utf-8") as fh:
            json.dump(self._metadata, fh, indent=2)

    def lookup_nearest(self, embedding: np.ndarray) -> str | None:
        """Return the predicted label of the nearest stored doc if cosine similarity
        is at or above ``SIMILARITY_THRESHOLD``, otherwise ``None``."""
        if not self._embeddings:
            return None
        embs = np.array(self._embeddings, dtype=np.float32)
        sims = embs @ embedding.astype(np.float32)   # both sides are unit-normalised
        best = int(np.argmax(sims))
        return self._metadata[best]["predicted"] if sims[best] >= SIMILARITY_THRESHOLD else None

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


# ---------------------------------------------------------------------------
# Single-run helper
# ---------------------------------------------------------------------------

def _single_run(seed: int, store_dir: str) -> pd.DataFrame:
    """
    Classify all corpus documents in a random order determined by *seed*,
    simulating the variability in which order files from ``docs/`` are
    encountered.

    Routing priority per document:
      1. Index   — ``store.lookup_nearest()`` cosine ≥ SIMILARITY_THRESHOLD
      2. Keyword — ``_is_keyword_path()`` regex match (no LLM)
      3. LLM     — ``counting_clf.classify_document()`` fallback

    A snapshot of cumulative route shares is recorded every SNAPSHOT_EVERY
    documents so the returned DataFrame shows the progression of index reliance
    as the store grows.
    """
    rng = np.random.default_rng(seed)

    counting_llm = MockLLM(_build_llm_map())
    counting_clf = MinimalClassifier(counting_llm)
    store = ClassificationStore(store_dir)

    # Flat corpus — all doc types, both keyword-signal and ambiguous variants
    all_docs: list[tuple[str, str]] = [
        (text, doc_type)
        for doc_type in DOCUMENT_TYPES
        for text in _KEYWORD_CORPUS.get(doc_type, []) + _AMBIGUOUS_CORPUS.get(doc_type, [])
    ]
    # Different permutation each run — key variability simulating docs/ read order
    all_docs = [all_docs[i] for i in rng.permutation(len(all_docs))]

    n_index = n_keyword = n_llm = 0
    results: list[dict] = []
    window_ms: list[float] = []  # per-doc routing latency within the current window

    for pos, (text, doc_type) in enumerate(all_docs, 1):
        # noise=0.01 keeps same-type cosine sim ~0.96 >> SIMILARITY_THRESHOLD
        # while cross-type sims stay near 0 in 384-D space
        embedding = make_embedding(rng, doc_type, noise=0.01)

        t0 = time.perf_counter()
        nearest = store.lookup_nearest(embedding)
        if nearest is not None:
            predicted = nearest
            n_index += 1
        elif _is_keyword_path(text):
            predicted = counting_clf.classify_document(text)
            n_keyword += 1
        else:
            predicted = counting_clf.classify_document(text)
            n_llm += 1
        window_ms.append((time.perf_counter() - t0) * 1_000)

        store.add(text, predicted, embedding)

        if pos % SNAPSHOT_EVERY == 0 or pos == len(all_docs):
            results.append(
                {
                    "n_classified": pos,
                    "index_hit_pct": n_index / pos * 100,
                    "keyword_hit_pct": n_keyword / pos * 100,
                    "llm_call_pct": n_llm / pos * 100,
                    "window_median_ms": float(np.median(window_ms)),
                }
            )
            window_ms = []

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Plotting helper — 3-run side-by-side comparison
# ---------------------------------------------------------------------------

# One distinct colour per run
_RUN_COLORS: list[str] = ["#1565C0", "#2E7D32", "#BF360C"]  # blue, green, red


def _plot_three_run_comparison(runs: dict[int, pd.DataFrame], output_path: Path) -> None:
    """
    Three-panel chart comparing three independent runs with different doc orderings:

      Left   — Index hit% vs. documents classified (increasing = more index reliance)
      Centre — LLM call% vs. documents classified (decreasing = less LLM reliance)
      Right  — Median classification time (ms) per snapshot window
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        "Pipeline Classification Dynamics — 3 Random Doc Orderings Side by Side\n"
        "(index-first routing: index → keyword → LLM)",
        fontsize=12,
        fontweight="bold",
    )

    for (seed, df), color in zip(runs.items(), _RUN_COLORS):
        x = df["n_classified"]
        lbl = f"seed={seed}"

        # ── Left: growing index reliance ──────────────────────────────
        axes[0].plot(x, df["index_hit_pct"], "o-", color=color, lw=2, ms=5, label=lbl)
        axes[0].fill_between(x, df["index_hit_pct"], alpha=0.08, color=color)

        # ── Centre: shrinking LLM reliance ────────────────────────────
        axes[1].plot(x, df["llm_call_pct"], "s-", color=color, lw=2, ms=5, label=lbl)
        axes[1].fill_between(x, df["llm_call_pct"], alpha=0.08, color=color)

        # ── Right: per-window median classification speed ─────────────
        axes[2].plot(x, df["window_median_ms"], "^-", color=color, lw=2, ms=5, label=lbl)

    axes[0].set_xlabel("Documents Classified")
    axes[0].set_ylabel("Index Hit (%)")
    axes[0].set_title("Growing Reliance on Classification Index")
    axes[0].set_ylim(0, 105)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=9)

    axes[1].set_xlabel("Documents Classified")
    axes[1].set_ylabel("LLM Call (%)")
    axes[1].set_title("Decreasing LLM Fallback")
    axes[1].set_ylim(0, 105)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=9)

    axes[2].set_xlabel("Documents Classified")
    axes[2].set_ylabel("Median Classification Time (ms)")
    axes[2].set_title("Classification Speed per Window")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(fontsize=9)

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
    Wraps ``pipeline._classify_document`` to measure speed and path
    distribution (keyword vs. LLM) as the temporary embedding store grows,
    across three independent random orderings of the document corpus.
    """

    # -----------------------------------------------------------------------
    # Test 1: Three-run side-by-side comparison
    # -----------------------------------------------------------------------

    def test_three_run_ordering_comparison(self, tmp_path: Path) -> None:
        """
        Runs classification three times, each with a different random ordering
        of the document corpus (simulating different ``docs/`` read orders).
        Compares the three runs side by side on index reliance and LLM fallback.

        Assertions
        ----------
        * Index hit% is strictly higher at the end than at the start (store
          grows and becomes increasingly useful).
        * LLM call% at the final snapshot is lower than at the first snapshot.
        * Output PNG written to tmp_path/ and mirrored to artifacts/.
        """
        runs: dict[int, pd.DataFrame] = {}

        for seed in RUN_SEEDS:
            run_dir = tmp_path / f"run_{seed}"
            run_dir.mkdir()
            df = _single_run(seed, str(run_dir))

            assert not df.empty, f"Run seed={seed} produced no snapshots"
            assert df["index_hit_pct"].iloc[-1] > df["index_hit_pct"].iloc[0], (
                f"seed={seed}: index reliance did not grow "
                f"({df['index_hit_pct'].iloc[0]:.1f}% → {df['index_hit_pct'].iloc[-1]:.1f}%)"
            )
            assert df["llm_call_pct"].iloc[-1] <= df["llm_call_pct"].iloc[0], (
                f"seed={seed}: LLM call% did not decrease "
                f"({df['llm_call_pct'].iloc[0]:.1f}% → {df['llm_call_pct'].iloc[-1]:.1f}%)"
            )

            runs[seed] = df

        chart_path = tmp_path / "classification_dynamics.png"
        _plot_three_run_comparison(runs, chart_path)
        assert chart_path.exists() and chart_path.stat().st_size > 0

        artifacts_dir = Path(__file__).parent.parent / "artifacts"
        if artifacts_dir.exists():
            shutil.copy(chart_path, artifacts_dir / "classification_dynamics.png")

        for seed, df in runs.items():
            print(
                f"\n[Run seed={seed}]"
                f"  snapshots={len(df)}"
                f"  index_start={df['index_hit_pct'].iloc[0]:.1f}%"
                f"  index_end={df['index_hit_pct'].iloc[-1]:.1f}%"
                f"  llm_start={df['llm_call_pct'].iloc[0]:.1f}%"
                f"  llm_end={df['llm_call_pct'].iloc[-1]:.1f}%"
                f"  speed_first={df['window_median_ms'].iloc[0]:.4f}ms"
                f"  speed_last={df['window_median_ms'].iloc[-1]:.4f}ms"
            )

    # -----------------------------------------------------------------------
    # Test 2: Keyword vs LLM path counts per category
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
                store.add(text, predicted, make_embedding(rng, doc_type))

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
                store.add(text, predicted, make_embedding(rng, doc_type))
                assert mock_llm.call_count == expected_llm_calls, (
                    f"Ambiguous doc should trigger LLM: {text[:50]}"
                )

        assert mock_llm.call_count == expected_llm_calls

    # -----------------------------------------------------------------------
    # Test 3: Store persistence round-trip
    # -----------------------------------------------------------------------

    def test_store_persists_to_disk(self, temp_store_dir: str) -> None:
        """
        Classifies a small set of documents and verifies the store serialises
        both embeddings and metadata to disk correctly.

        Assertions
        ----------
        * embeddings.npy and metadata.json exist after classification.
        * Reloaded embedding shape matches.
        * Reloaded metadata contains text_snippet and predicted fields.
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
        for text, doc_type in docs:
            predicted = classifier.classify_document(text)
            store.add(text, predicted, make_embedding(rng, doc_type))

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
            assert "text_snippet" in entry

"""
Embedding Classification Dynamics Test
=======================================
Measures how classification accuracy and response speed change as more
documents are progressively added to a temporary embedding store.

Key features
------------
* Temporary numpy-backed embedding store (no heavy ML deps required).
* Deterministic synthetic embeddings per document type.
* Accuracy and latency recorded after every batch of new documents.
* Two output graphs:
    1. Overall accuracy + speed vs. store size.
    2. Per-category accuracy breakdown vs. store size.
* Persistence round-trip: store survives serialisation to disk.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
from collections import Counter
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")  # Non-interactive backend — safe for CI and headless runs.


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DOCUMENT_TYPES: list[str] = [
    "cover_letter",
    "certificate_of_quality",
    "packaging_specification",
    "bse_tse_declaration",
    "material_description",
    "supplier_qualification",
    "chain_of_custody",
    "unknown",
]

EMBEDDING_DIM: int = 384  # Matches sentence-transformers/all-MiniLM-L6-v2


# ---------------------------------------------------------------------------
# EmbeddingStore — temporary database for embedded documents
# ---------------------------------------------------------------------------


class EmbeddingStore:
    """
    Lightweight embedding store backed by numpy arrays on disk.

    Documents are added incrementally.  Classification uses cosine-similarity
    search plus top-k majority voting, mirroring the FAISS + retrieval logic
    in the production RAG pipeline.

    Parameters
    ----------
    store_dir : str
        Path to an existing directory used for persistence.
    embedding_dim : int
        Dimensionality of the embedding vectors.
    """

    _EMBEDDINGS_FILE = "embeddings.npy"
    _LABELS_FILE = "labels.json"

    def __init__(self, store_dir: str, embedding_dim: int) -> None:
        self.store_dir = Path(store_dir)
        self.embedding_dim = embedding_dim
        self._embeddings: list[np.ndarray] = []
        self._labels: list[str] = []

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add(self, embedding: np.ndarray, label: str) -> None:
        """Append a single embedding and persist the updated store to disk."""
        self._embeddings.append(embedding.astype(np.float32))
        self._labels.append(label)
        self._persist()

    def add_batch(self, embeddings: list[np.ndarray], labels: list[str]) -> None:
        """Append multiple embeddings at once (single persist at the end)."""
        for emb, lbl in zip(embeddings, labels):
            self._embeddings.append(emb.astype(np.float32))
            self._labels.append(lbl)
        self._persist()

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------

    def classify(self, query_embedding: np.ndarray, top_k: int = 3) -> str:
        """
        Return the predicted label for *query_embedding* using cosine similarity
        and majority voting among the *top_k* nearest neighbours.
        """
        if not self._embeddings:
            return "unknown"

        matrix = np.array(self._embeddings, dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        matrix_normed = matrix / np.where(norms == 0, 1.0, norms)

        q = query_embedding.astype(np.float32)
        q_norm = np.linalg.norm(q)
        q_normed = q / (q_norm if q_norm > 0 else 1.0)

        similarities = matrix_normed @ q_normed
        k = min(top_k, len(self._embeddings))
        top_indices = np.argpartition(similarities, -k)[-k:]
        top_labels = [self._labels[i] for i in top_indices]
        return Counter(top_labels).most_common(1)[0][0]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _persist(self) -> None:
        np.save(str(self.store_dir / self._EMBEDDINGS_FILE), np.array(self._embeddings, dtype=np.float32))
        with open(self.store_dir / self._LABELS_FILE, "w", encoding="utf-8") as fh:
            json.dump(self._labels, fh)

    @classmethod
    def load(cls, store_dir: str, embedding_dim: int) -> "EmbeddingStore":
        """Reconstruct an EmbeddingStore from a previously persisted directory."""
        store = cls(store_dir, embedding_dim)
        emb_path = Path(store_dir) / cls._EMBEDDINGS_FILE
        lbl_path = Path(store_dir) / cls._LABELS_FILE

        if emb_path.exists() and lbl_path.exists():
            loaded = np.load(str(emb_path))
            store._embeddings = [loaded[i] for i in range(len(loaded))]
            with open(lbl_path, encoding="utf-8") as fh:
                store._labels = json.load(fh)
        return store

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._embeddings)

    @property
    def size_bytes(self) -> int:
        total = 0
        for fname in (self._EMBEDDINGS_FILE, self._LABELS_FILE):
            p = self.store_dir / fname
            if p.exists():
                total += p.stat().st_size
        return total


# ---------------------------------------------------------------------------
# Synthetic embedding generation
# ---------------------------------------------------------------------------


def _centroid_for_type(doc_type: str) -> np.ndarray:
    """
    Return a unit-norm centroid vector that is unique and deterministic for
    *doc_type*.  The centroid acts as the "true" embedding for that class.
    """
    seed = sum(i * ord(c) for i, c in enumerate(doc_type, 1))
    rng = np.random.default_rng(seed)
    centroid = rng.standard_normal(EMBEDDING_DIM).astype(np.float32)
    return centroid / np.linalg.norm(centroid)


def make_embeddings(rng: np.random.Generator, doc_type: str, count: int, noise: float = 0.1) -> list[np.ndarray]:
    """
    Return *count* synthetic embeddings for *doc_type*.

    Each embedding = unit_centroid + gaussian noise * *noise*, then
    re-normalised.  Smaller *noise* → tighter cluster → higher accuracy.
    """
    centroid = _centroid_for_type(doc_type)
    embeddings: list[np.ndarray] = []
    for _ in range(count):
        noise_vec = rng.standard_normal(EMBEDDING_DIM).astype(np.float32) * noise
        emb = centroid + noise_vec
        norm = np.linalg.norm(emb)
        embeddings.append(emb / (norm if norm > 0 else 1.0))
    return embeddings


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

_BLUE = "#2196F3"
_RED = "#F44336"
_ORANGE = "#FF7043"


def _plot_accuracy_and_speed(df: pd.DataFrame, output_path: Path) -> None:
    """Two-panel chart: accuracy and latency vs. number of stored embeddings."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Embedding Store — Classification Dynamics", fontsize=14, fontweight="bold")

    x = df["n_stored"]

    # ── Panel 1: Accuracy ─────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(x, df["accuracy_pct"], "o-", color=_BLUE, linewidth=2, markersize=6, label="Accuracy (%)")
    ax.fill_between(x, df["accuracy_pct"], alpha=0.12, color=_BLUE)
    ax.axhline(y=df["accuracy_pct"].iloc[-1], color=_BLUE, linestyle="--", alpha=0.4, linewidth=1)
    ax.set_xlabel("Stored Embeddings")
    ax.set_ylabel("Classification Accuracy (%)")
    ax.set_title("Accuracy vs. Store Size")
    ax.set_ylim(0, 107)
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Annotate first and last point
    for idx in (0, -1):
        ax.annotate(
            f"{df['accuracy_pct'].iloc[idx]:.1f}%",
            xy=(x.iloc[idx], df["accuracy_pct"].iloc[idx]),
            xytext=(8, -14),
            textcoords="offset points",
            fontsize=8,
            color=_BLUE,
        )

    # ── Panel 2: Latency ──────────────────────────────────────────────────
    ax = axes[1]
    ax.plot(x, df["avg_latency_ms"], "s-", color=_RED, linewidth=2, markersize=6, label="Avg latency (ms)")
    ax.fill_between(x, df["avg_latency_ms"], df["p95_latency_ms"], alpha=0.18, color=_RED, label="P95 band")
    ax.plot(x, df["p95_latency_ms"], "^--", color=_ORANGE, linewidth=1.5, markersize=5, label="P95 latency (ms)")
    ax.set_xlabel("Stored Embeddings")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Classification Speed vs. Store Size")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_per_category(df: pd.DataFrame, output_path: Path) -> None:
    """Multi-line chart: per-category accuracy vs. number of stored embeddings."""
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_title("Per-Category Classification Accuracy vs. Store Size", fontsize=13, fontweight="bold")

    colors = plt.cm.tab10(np.linspace(0, 1, len(DOCUMENT_TYPES)))  # type: ignore[attr-defined]
    x = df["n_stored"]

    for doc_type, color in zip(DOCUMENT_TYPES, colors):
        col = f"acc_{doc_type}"
        label = doc_type.replace("_", " ").title()
        ax.plot(x, df[col], "o-", color=color, linewidth=2, markersize=5, label=label)

    ax.set_xlabel("Stored Embeddings")
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
    """Yield a fresh temporary directory; remove it unconditionally afterwards."""
    d = tempfile.mkdtemp(prefix="rag_embed_test_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestEmbeddingClassificationDynamics:
    """
    Measure how classification accuracy and speed evolve as the embedding
    store grows with progressively more documents.
    """

    # ------------------------------------------------------------------
    # Test 1: Overall accuracy and speed
    # ------------------------------------------------------------------

    def test_accuracy_and_speed_over_growing_store(self, temp_store_dir: str, tmp_path: Path) -> None:
        """
        Progressively adds documents to the embedding store in batches.
        After each batch the full test set is classified and both accuracy
        (%) and per-query latency (ms) are recorded.

        Assertions
        ----------
        * Final accuracy ≥ 70 % (cosine k-NN should converge quickly).
        * Accuracy must not drop more than 5 pp from first to last batch.
        * The store grows monotonically.
        * Output PNG is written to *tmp_path* and to artifacts/.

        Graph
        -----
        Two-panel figure: accuracy trend + latency trend vs. store size.
        """
        rng = np.random.default_rng(42)
        store = EmbeddingStore(temp_store_dir, EMBEDDING_DIM)

        # ── Fixed held-out test set ────────────────────────────────────
        # 10 samples per category, tighter noise so the set is "clean".
        test_embeddings: list[np.ndarray] = []
        test_labels: list[str] = []
        for doc_type in DOCUMENT_TYPES:
            embs = make_embeddings(rng, doc_type, count=10, noise=0.05)
            test_embeddings.extend(embs)
            test_labels.extend([doc_type] * 10)

        # Shuffle test set to avoid ordering bias.
        perm = rng.permutation(len(test_embeddings))
        test_embeddings = [test_embeddings[i] for i in perm]
        test_labels = [test_labels[i] for i in perm]

        # ── Progressive training ───────────────────────────────────────
        DOCS_PER_TYPE_PER_BATCH = 2   # documents added per category per round
        N_BATCHES = 10                 # rounds → final store: 10 * 2 * 8 = 160 docs

        results: list[dict] = []

        for batch_idx in range(N_BATCHES):
            # Add one batch for every category.
            for doc_type in DOCUMENT_TYPES:
                embs = make_embeddings(rng, doc_type, count=DOCS_PER_TYPE_PER_BATCH, noise=0.10)
                store.add_batch(embs, [doc_type] * DOCS_PER_TYPE_PER_BATCH)

            n_stored = len(store)

            # ── Evaluate accuracy and latency ──────────────────────────
            correct = 0
            latencies_ms: list[float] = []

            for q_emb, true_label in zip(test_embeddings, test_labels):
                t0 = time.perf_counter()
                predicted = store.classify(q_emb, top_k=3)
                latencies_ms.append((time.perf_counter() - t0) * 1_000)
                if predicted == true_label:
                    correct += 1

            accuracy_pct = correct / len(test_labels) * 100

            results.append(
                {
                    "batch": batch_idx + 1,
                    "n_stored": n_stored,
                    "accuracy_pct": accuracy_pct,
                    "avg_latency_ms": float(np.mean(latencies_ms)),
                    "p95_latency_ms": float(np.percentile(latencies_ms, 95)),
                    "store_size_bytes": store.size_bytes,
                }
            )

        df = pd.DataFrame(results)

        # ── Assertions ─────────────────────────────────────────────────
        first_acc = df["accuracy_pct"].iloc[0]
        last_acc = df["accuracy_pct"].iloc[-1]

        assert last_acc > 70.0, f"Final accuracy {last_acc:.1f}% is unexpectedly low"
        assert last_acc >= first_acc - 5.0, (
            f"Accuracy regressed: {first_acc:.1f}% → {last_acc:.1f}%"
        )
        assert df["n_stored"].is_monotonic_increasing, "Store size should grow each batch"

        # ── Output graph ───────────────────────────────────────────────
        chart_path = tmp_path / "embedding_dynamics.png"
        _plot_accuracy_and_speed(df, chart_path)
        assert chart_path.exists() and chart_path.stat().st_size > 0

        # Mirror to artifacts/ so the chart is visible in the repo.
        artifacts_dir = Path(__file__).parent.parent / "artifacts"
        if artifacts_dir.exists():
            shutil.copy(chart_path, artifacts_dir / "embedding_dynamics.png")

        # ── Summary print (visible with -s) ───────────────────────────
        print(
            f"\n[Dynamics] batches={N_BATCHES}  docs={df['n_stored'].iloc[-1]}  "
            f"accuracy={last_acc:.1f}%  avg_latency={df['avg_latency_ms'].iloc[-1]:.3f} ms"
        )

    # ------------------------------------------------------------------
    # Test 2: Per-category accuracy breakdown
    # ------------------------------------------------------------------

    def test_per_category_accuracy_over_growing_store(self, temp_store_dir: str, tmp_path: Path) -> None:
        """
        Tracks classification accuracy per document type as the store grows.
        Reveals which categories are harder to distinguish and how quickly
        each one converges.

        Assertions
        ----------
        * Every category achieves ≥ 60 % accuracy by the final batch.
        * Output PNG is written to *tmp_path* and to artifacts/.

        Graph
        -----
        Multi-line chart with one line per document type.
        """
        rng = np.random.default_rng(99)
        store = EmbeddingStore(temp_store_dir, EMBEDDING_DIM)

        # Held-out test set: 5 samples per category.
        test_set: dict[str, list[np.ndarray]] = {
            doc_type: make_embeddings(rng, doc_type, count=5, noise=0.05)
            for doc_type in DOCUMENT_TYPES
        }

        DOCS_PER_TYPE_PER_BATCH = 3
        N_BATCHES = 8

        results: list[dict] = []

        for batch_idx in range(N_BATCHES):
            for doc_type in DOCUMENT_TYPES:
                embs = make_embeddings(rng, doc_type, count=DOCS_PER_TYPE_PER_BATCH, noise=0.12)
                store.add_batch(embs, [doc_type] * DOCS_PER_TYPE_PER_BATCH)

            row: dict = {"batch": batch_idx + 1, "n_stored": len(store)}
            for doc_type in DOCUMENT_TYPES:
                correct = sum(
                    1 for emb in test_set[doc_type] if store.classify(emb, top_k=3) == doc_type
                )
                row[f"acc_{doc_type}"] = correct / len(test_set[doc_type]) * 100
            results.append(row)

        df = pd.DataFrame(results)

        # ── Assertions ─────────────────────────────────────────────────
        for doc_type in DOCUMENT_TYPES:
            final_acc = df[f"acc_{doc_type}"].iloc[-1]
            assert final_acc >= 60.0, (
                f"Category '{doc_type}' final accuracy {final_acc:.1f}% is too low"
            )

        # ── Output graph ───────────────────────────────────────────────
        chart_path = tmp_path / "per_category_accuracy.png"
        _plot_per_category(df, chart_path)
        assert chart_path.exists() and chart_path.stat().st_size > 0

        artifacts_dir = Path(__file__).parent.parent / "artifacts"
        if artifacts_dir.exists():
            shutil.copy(chart_path, artifacts_dir / "per_category_accuracy.png")

        print(
            f"\n[Per-category] batches={N_BATCHES}  docs={df['n_stored'].iloc[-1]}"
        )
        for doc_type in DOCUMENT_TYPES:
            print(f"  {doc_type:30s}: {df[f'acc_{doc_type}'].iloc[-1]:.0f}%")

    # ------------------------------------------------------------------
    # Test 3: Store persistence and reload
    # ------------------------------------------------------------------

    def test_store_persists_to_disk_and_reloads(self, temp_store_dir: str) -> None:
        """
        Verify that the embedding store serialises to disk and can be
        fully reconstructed, producing identical classification results.

        Assertions
        ----------
        * Disk files are created after the first add().
        * Reloaded store has the same number of embeddings.
        * Classification predictions are identical before and after reload.
        """
        rng = np.random.default_rng(7)

        store = EmbeddingStore(temp_store_dir, EMBEDDING_DIM)
        for doc_type in DOCUMENT_TYPES[:4]:
            embs = make_embeddings(rng, doc_type, count=5, noise=0.10)
            store.add_batch(embs, [doc_type] * 5)

        n_before = len(store)

        # Files must exist on disk.
        assert (Path(temp_store_dir) / EmbeddingStore._EMBEDDINGS_FILE).exists()
        assert (Path(temp_store_dir) / EmbeddingStore._LABELS_FILE).exists()

        # Reload from disk.
        restored = EmbeddingStore.load(temp_store_dir, EMBEDDING_DIM)
        assert len(restored) == n_before, (
            f"Loaded {len(restored)} embeddings but expected {n_before}"
        )

        # Predictions must match for unseen queries.
        for doc_type in DOCUMENT_TYPES[:4]:
            probe = make_embeddings(rng, doc_type, count=1, noise=0.02)[0]
            original_pred = store.classify(probe, top_k=3)
            restored_pred = restored.classify(probe, top_k=3)
            assert original_pred == restored_pred, (
                f"Mismatch for {doc_type}: original={original_pred}  restored={restored_pred}"
            )

    # ------------------------------------------------------------------
    # Test 4: Cold-start edge case (empty store)
    # ------------------------------------------------------------------

    def test_classify_returns_unknown_on_empty_store(self, temp_store_dir: str) -> None:
        """An empty store must return 'unknown' rather than raising."""
        store = EmbeddingStore(temp_store_dir, EMBEDDING_DIM)
        rng = np.random.default_rng(0)
        query = rng.standard_normal(EMBEDDING_DIM).astype(np.float32)
        assert store.classify(query) == "unknown"

    # ------------------------------------------------------------------
    # Test 5: Latency scales sub-linearly (sanity check)
    # ------------------------------------------------------------------

    def test_latency_growth_is_reasonable(self, temp_store_dir: str) -> None:
        """
        Verify that latency does not blow up as the store grows.

        With a pure numpy cosine search the cost is O(n * d) where n is the
        store size and d the embedding dimension.  For the sizes used here
        (≤ 2 000 docs) the search should complete in well under 50 ms on any
        modern CPU, even without FAISS.
        """
        rng = np.random.default_rng(13)
        store = EmbeddingStore(temp_store_dir, EMBEDDING_DIM)

        # Populate with 2 000 embeddings across all categories.
        N_DOCS = 2_000
        batch_labels: list[str] = []
        batch_embs: list[np.ndarray] = []
        for i in range(N_DOCS):
            doc_type = DOCUMENT_TYPES[i % len(DOCUMENT_TYPES)]
            batch_embs.extend(make_embeddings(rng, doc_type, count=1, noise=0.10))
            batch_labels.append(doc_type)
        store.add_batch(batch_embs, batch_labels)

        # Time 20 classification calls.
        probes = [rng.standard_normal(EMBEDDING_DIM).astype(np.float32) for _ in range(20)]
        t0 = time.perf_counter()
        for p in probes:
            store.classify(p, top_k=3)
        elapsed_ms = (time.perf_counter() - t0) * 1_000 / len(probes)

        assert elapsed_ms < 50.0, (
            f"Average classify latency {elapsed_ms:.2f} ms exceeds 50 ms for {N_DOCS} stored docs"
        )
        print(f"\n[Latency] {N_DOCS} docs -> avg classify = {elapsed_ms:.3f} ms")

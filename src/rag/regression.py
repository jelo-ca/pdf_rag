"""Utilities for pandas-driven regression testing of the RAG pipeline.

This module provides a lightweight harness that:
1) executes a suite of query test cases,
2) records retrieval/answer metrics in a DataFrame,
3) compares current runs to a baseline to flag regressions.
"""

# pylint: disable=too-many-lines,too-many-locals,too-many-statements

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


@dataclass
class RegressionThresholds:
    """Thresholds that define what counts as a regression."""

    min_answer_similarity: float = 0.65
    max_response_time_increase_ms: float = 3000.0
    max_confidence_drop: float = 25.0


class RAGRegressionHarness:
    """Runs repeatable query regression tests against a RAG pipeline."""

    def __init__(
        self,
        rag_pipeline: Any,
        thresholds: Optional[RegressionThresholds] = None,
    ) -> None:
        self.rag = rag_pipeline
        self.thresholds = thresholds or RegressionThresholds()

    @staticmethod
    def create_test_suite(test_cases: List[Dict[str, Any]]) -> pd.DataFrame:
        """Create a normalized DataFrame of regression test cases.

        Expected keys per test case:
            - test_id (str)
            - query (str)

        Optional keys:
            - expected_query_category (str)
            - min_sources (int)
            - required_terms (str): pipe-separated terms, e.g. "batch|lot|expiry"
            - criticality (str)
            - classify (bool)
            - expand (bool)
            - num_expansions (int)
        """
        if not test_cases:
            raise ValueError("test_cases cannot be empty.")

        suite_df = pd.DataFrame(test_cases).copy()
        required = {"test_id", "query"}
        missing = required.difference(suite_df.columns)
        if missing:
            missing_list = ", ".join(sorted(missing))
            raise ValueError(f"Missing required test case columns: {missing_list}")

        defaults: Dict[str, Any] = {
            "expected_query_category": None,
            "min_sources": 1,
            "required_terms": "",
            "criticality": "medium",
            "classify": False,
            "expand": False,
            "num_expansions": 3,
        }
        for column, value in defaults.items():
            if column not in suite_df.columns:
                suite_df[column] = value
            else:
                if value is None:
                    suite_df[column] = suite_df[column].where(suite_df[column].notna(), None)
                else:
                    suite_df[column] = suite_df[column].fillna(value)

        suite_df["classify"] = suite_df["classify"].map(bool)
        suite_df["expand"] = suite_df["expand"].map(bool)
        suite_df["min_sources"] = suite_df["min_sources"].map(int)
        suite_df["num_expansions"] = suite_df["num_expansions"].map(int)

        unchecked = suite_df["classify"] & suite_df["expected_query_category"].isna()
        if unchecked.any():
            ids = suite_df.loc[unchecked, "test_id"].tolist()
            warnings.warn(
                f"Tests with classify=True but no expected_query_category (category will not be "
                f"validated): {ids}",
                stacklevel=2,
            )

        return suite_df

    def run(self, test_suite: pd.DataFrame) -> pd.DataFrame:
        """Execute each test in the suite and return a results DataFrame."""
        rows: List[Dict[str, Any]] = []

        for _, test in test_suite.iterrows():
            t0 = time.perf_counter()
            payload = self.rag.query_with_sources(
                test["query"],
                classify=bool(test["classify"]),
                expand=bool(test["expand"]),
                num_expansions=int(test["num_expansions"]),
                retrieval_mode=str(test.get("retrieval_mode", "hybrid")),
            )
            t1 = time.perf_counter()
            timestamp = datetime.now(timezone.utc)

            answer = str(payload.get("answer", ""))
            sources = payload.get("sources", []) or []
            query_category = payload.get("query_category")

            source_scores = [s.get("score") for s in sources if s.get("score") is not None]
            avg_confidence = float(sum(source_scores) / len(source_scores)) if source_scores else 0.0

            min_sources = int(test.get("min_sources", 1))
            has_min_sources = len(sources) >= min_sources
            category_ok = self._category_match(query_category, test.get("expected_query_category"))
            terms_ok = self._required_terms_match(answer, str(test.get("required_terms", "")))

            # Stage timing fields — fall back to wall-clock total if pipeline
            # predates the timing instrumentation.
            wall_total_ms = (t1 - t0) * 1_000
            total_ms = payload.get("total_ms", wall_total_ms)

            row = {
                "timestamp": timestamp,
                "test_id": test["test_id"],
                "criticality": test.get("criticality", "medium"),
                "query": test["query"],
                "retrieval_mode": payload.get("retrieval_mode", test.get("retrieval_mode", "hybrid")),
                "answer": answer,
                "answer_length": len(answer),
                "query_category": query_category,
                "num_sources": len(sources),
                "avg_confidence": round(avg_confidence, 3),
                "expand_ms":   round(payload.get("expand_ms",   0.0), 3),
                "classify_ms": round(payload.get("classify_ms", 0.0), 3),
                "retrieve_ms": round(payload.get("retrieve_ms", 0.0), 3),
                "generate_ms": round(payload.get("generate_ms", 0.0), 3),
                "total_ms":    round(total_ms, 3),
                "response_time_ms": round(total_ms, 3),  # backward-compat alias
                "has_min_sources": has_min_sources,
                "category_match": category_ok,
                "required_terms_match": terms_ok,
                "passed": has_min_sources and category_ok and terms_ok and len(answer.strip()) > 0,
            }
            rows.append(row)

        return pd.DataFrame(rows)

    def compare_to_baseline(
        self,
        current_results: pd.DataFrame,
        baseline_results: pd.DataFrame,
    ) -> pd.DataFrame:
        """Join current and baseline results and compute regression flags."""
        merge_columns = [
            "test_id",
            "answer",
            "passed",
            "avg_confidence",
            "response_time_ms",
            "num_sources",
        ]
        missing_current = [c for c in merge_columns if c not in current_results.columns]
        missing_baseline = [c for c in merge_columns if c not in baseline_results.columns]
        if missing_current:
            raise ValueError(f"current_results missing columns: {', '.join(missing_current)}")
        if missing_baseline:
            raise ValueError(f"baseline_results missing columns: {', '.join(missing_baseline)}")

        current_ids = set(current_results["test_id"])
        baseline_ids = set(baseline_results["test_id"])
        new_ids = sorted(current_ids - baseline_ids)
        dropped_ids = sorted(baseline_ids - current_ids)
        if new_ids:
            warnings.warn(
                f"Test IDs in current run but not in baseline (excluded from comparison): {new_ids}",
                stacklevel=2,
            )
        if dropped_ids:
            warnings.warn(
                f"Test IDs in baseline but not in current run (excluded from comparison): {dropped_ids}",
                stacklevel=2,
            )

        merged = current_results.merge(
            baseline_results[merge_columns],
            on="test_id",
            suffixes=("_current", "_baseline"),
            how="inner",
        )

        merged["answer_similarity"] = merged.apply(
            lambda row: self._answer_similarity(
                str(row["answer_current"]),
                str(row["answer_baseline"]),
            ),
            axis=1,
        )
        merged["confidence_delta"] = merged["avg_confidence_current"] - merged["avg_confidence_baseline"]
        merged["response_time_delta_ms"] = (
            merged["response_time_ms_current"] - merged["response_time_ms_baseline"]
        )
        merged["num_sources_delta"] = merged["num_sources_current"] - merged["num_sources_baseline"]

        merged["regression_detected"] = merged.apply(self._is_regression, axis=1)
        return merged

    @staticmethod
    def summarize(results_df: pd.DataFrame) -> Dict[str, Any]:
        """Return aggregate metrics from a run."""
        if results_df.empty:
            return {
                "total_tests": 0,
                "pass_rate": 0.0,
                "avg_response_time_ms": 0.0,
                "avg_confidence": 0.0,
            }

        total = len(results_df)
        passed = int(results_df["passed"].sum()) if "passed" in results_df else 0
        return {
            "total_tests": total,
            "passed_tests": passed,
            "failed_tests": total - passed,
            "pass_rate": round((passed / total) * 100.0, 2),
            "avg_response_time_ms": round(float(results_df["response_time_ms"].mean()), 2),
            "avg_confidence": round(float(results_df["avg_confidence"].mean()), 2),
        }

    @staticmethod
    def save_results(results_df: pd.DataFrame, csv_path: str) -> None:
        """Persist regression outputs to CSV."""
        results_df.to_csv(csv_path, index=False)

    @staticmethod
    def load_results(csv_path: str) -> pd.DataFrame:
        """Load previously saved results CSV."""
        return pd.read_csv(csv_path)

    @staticmethod
    def visualize_results(
        results_df: pd.DataFrame,
        output_path: str,
        title: str = "RAG Regression Run Summary",
    ) -> str:
        """Generate a PNG dashboard from one regression run.

        The output includes:
            - pass/fail count,
            - response-time distribution,
            - confidence distribution,
            - per-test response time.
        """
        required = {"test_id", "passed", "response_time_ms", "avg_confidence"}
        missing = sorted(required.difference(results_df.columns))
        if missing:
            raise ValueError(f"results_df missing required columns: {', '.join(missing)}")
        if results_df.empty:
            raise ValueError("results_df is empty; nothing to visualize.")

        # Delay pyplot import to avoid backend issues during module import in headless CI.
        import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel

        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(2, 2, figsize=(14, 9))
        fig.suptitle(title, fontsize=14, fontweight="bold")

        pass_count = int(results_df["passed"].sum())
        fail_count = int((~results_df["passed"].astype(bool)).sum())
        axes[0, 0].bar(["Passed", "Failed"], [pass_count, fail_count], color=["#2e7d32", "#c62828"])
        axes[0, 0].set_title("Pass / Fail")
        axes[0, 0].set_ylabel("Test Count")

        axes[0, 1].hist(results_df["response_time_ms"], bins=min(10, max(3, len(results_df))), color="#1565c0")
        axes[0, 1].set_title("Response Time Distribution")
        axes[0, 1].set_xlabel("Milliseconds")
        axes[0, 1].set_ylabel("Frequency")

        axes[1, 0].hist(results_df["avg_confidence"], bins=min(10, max(3, len(results_df))), color="#6a1b9a")
        axes[1, 0].set_title("Confidence Distribution")
        axes[1, 0].set_xlabel("Confidence")
        axes[1, 0].set_ylabel("Frequency")

        sorted_df = results_df.sort_values("response_time_ms", ascending=False)
        top_n = min(15, len(sorted_df))
        y_labels = sorted_df["test_id"].head(top_n).astype(str).tolist()
        x_vals = sorted_df["response_time_ms"].head(top_n).tolist()
        axes[1, 1].barh(y_labels, x_vals, color="#ef6c00")
        axes[1, 1].invert_yaxis()
        axes[1, 1].set_title("Per-Test Response Time (Top Slowest)")
        axes[1, 1].set_xlabel("Milliseconds")

        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        return str(out_path)

    @staticmethod
    def visualize_comparison(
        comparison_df: pd.DataFrame,
        output_path: str,
        title: str = "RAG Baseline Comparison",
    ) -> str:
        """Generate a PNG dashboard for baseline-vs-current comparisons."""
        required = {
            "test_id",
            "answer_similarity",
            "confidence_delta",
            "response_time_delta_ms",
            "regression_detected",
        }
        missing = sorted(required.difference(comparison_df.columns))
        if missing:
            raise ValueError(f"comparison_df missing required columns: {', '.join(missing)}")
        if comparison_df.empty:
            raise ValueError("comparison_df is empty; nothing to visualize.")

        # Delay pyplot import to avoid backend issues during module import in headless CI.
        import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel

        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(2, 2, figsize=(14, 9))
        fig.suptitle(title, fontsize=14, fontweight="bold")

        reg_count = int(comparison_df["regression_detected"].sum())
        ok_count = int(len(comparison_df) - reg_count)
        axes[0, 0].bar(["No Regression", "Regression"], [ok_count, reg_count], color=["#2e7d32", "#c62828"])
        axes[0, 0].set_title("Regression Detection")

        axes[0, 1].hist(
            comparison_df["answer_similarity"],
            bins=min(10, max(3, len(comparison_df))),
            color="#00838f",
        )
        axes[0, 1].axvline(0.65, color="#c62828", linestyle="--", linewidth=1)
        axes[0, 1].set_title("Answer Similarity")
        axes[0, 1].set_xlabel("Similarity (0-1)")

        axes[1, 0].bar(
            comparison_df["test_id"].astype(str),
            comparison_df["confidence_delta"],
            color="#5d4037",
        )
        axes[1, 0].axhline(0.0, color="#424242", linewidth=1)
        axes[1, 0].tick_params(axis="x", rotation=45)
        axes[1, 0].set_title("Confidence Delta (Current - Baseline)")

        axes[1, 1].bar(
            comparison_df["test_id"].astype(str),
            comparison_df["response_time_delta_ms"],
            color="#283593",
        )
        axes[1, 1].axhline(0.0, color="#424242", linewidth=1)
        axes[1, 1].tick_params(axis="x", rotation=45)
        axes[1, 1].set_title("Latency Delta ms (Current - Baseline)")

        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        return str(out_path)

    @staticmethod
    def _category_match(actual: Optional[str], expected: Optional[str]) -> bool:
        if not expected:
            return True
        return (actual or "").strip().lower() == expected.strip().lower()

    @staticmethod
    def _required_terms_match(answer: str, required_terms: str) -> bool:
        """Check whether the answer satisfies the required_terms expression.

        Pipe (``|``) separates OR alternatives; any one matching group is
        sufficient.  Comma (``,``) joins AND requirements within a group; all
        comma-separated substrings must appear for that group to match.

        Examples::

            "batch|lot"          → passes if "batch" OR "lot" is present
            "batch,lot"          → passes only if BOTH "batch" AND "lot" are present
            "H401,H411|aquatic"  → passes if (H401 AND H411) OR "aquatic" is present
        """
        if not required_terms:
            return True
        answer_lower = answer.lower()
        or_groups = [g.strip() for g in required_terms.split("|") if g.strip()]
        if not or_groups:
            return True
        return any(
            all(part.strip().lower() in answer_lower for part in group.split(",") if part.strip())
            for group in or_groups
        )

    @staticmethod
    def _answer_similarity(answer_a: str, answer_b: str) -> float:
        """Text similarity in [0, 1] for baseline drift detection."""
        if not answer_a and not answer_b:
            return 1.0
        return SequenceMatcher(None, answer_a, answer_b).ratio()

    def _is_regression(self, row: pd.Series) -> bool:
        """Apply threshold policy to one merged baseline/current row."""
        failed_now = (not bool(row["passed_current"])) and bool(row["passed_baseline"])
        low_similarity = float(row["answer_similarity"]) < self.thresholds.min_answer_similarity
        high_latency_increase = (
            float(row["response_time_delta_ms"]) > self.thresholds.max_response_time_increase_ms
        )
        confidence_drop = float(row["confidence_delta"]) < (-1.0 * self.thresholds.max_confidence_drop)
        return failed_now or low_similarity or high_latency_increase or confidence_drop

    # ==========================================================================
    # Cumulative corpus-growth regression methods
    # ==========================================================================
    #
    # Background
    # ----------
    # A RAG pipeline's quality is not static: it changes as more documents are
    # added to the vector index.  Adding a new document can:
    #   - Introduce semantically similar chunks that compete with existing ones,
    #     lowering retrieval precision for previously-indexed content ("cross-
    #     contamination").
    #   - Dilute BM25 and vector similarity scores, reducing avg_confidence.
    #   - Increase FAISS ANN-search time, raising response latency.
    #
    # The three methods below (run_at_milestone, summarize_milestones,
    # visualize_growth_trends) form a data-collection + analysis pipeline:
    #
    #   run_at_milestone  →  per-test rows tagged with corpus metadata
    #       ↓ concatenate across milestones
    #   milestone_df      →  flat DataFrame: one row per (test, milestone)
    #       ↓ aggregate
    #   summarize_milestones  →  one row per milestone for trend analysis
    #       ↓ visualize
    #   visualize_growth_trends  →  6-panel PNG showing metric trajectories
    # ==========================================================================

    def run_at_milestone(
        self,
        test_suite: pd.DataFrame,
        milestone_index: int,
        doc_count: int,
        doc_just_added: str,
        new_test_ids: set,
    ) -> pd.DataFrame:
        """Run the suite and stamp every row with corpus-growth milestone metadata.

        This is the fundamental measurement step in cumulative regression testing.
        You call this method once per corpus-growth event (i.e. each time a new
        document is added to the index), passing:
          - the test suite that covers all documents currently in the index, and
          - metadata describing *which* document was just added.

        The method calls ``run()`` internally then attaches four extra columns to
        every result row so that rows from multiple milestones can be safely
        concatenated into a single growth-trend DataFrame for analysis and charts.

        Why milestone columns matter
        ----------------------------
        Without milestone context, a flat results CSV cannot tell you *when* a
        test started failing.  By stamping ``milestone_index`` and
        ``doc_just_added`` on every row, you can pivot the combined DataFrame to
        answer questions like:
          "Did T1-001 (COVID-19 product name) still pass after test_7.pdf was
           added to the index?"
        That pivot is what ``visualize_growth_trends`` uses for the heatmap panel.

        Parameters
        ----------
        test_suite:
            DataFrame from ``create_test_suite()``.  Must include only tests
            whose source documents have been indexed at this milestone; tests for
            documents that have not yet been indexed must be excluded so that the
            pipeline is not asked about content it has not seen.
        milestone_index:
            1-based integer position in the cumulative sequence.  Milestone 1 =
            first document indexed, milestone 2 = second document added, etc.
            This is the x-axis value in every growth-trend line chart.
        doc_count:
            Total number of documents in the index right now.  Equals
            ``milestone_index`` within a single phase, but may differ when the
            cumulative run spans two separate phases (PDF + OCR image folders)
            and the global ``milestone_index`` is a combined counter.
        doc_just_added:
            Human-readable name (file name or folder name) of the document added
            at this milestone, e.g. ``"test_3.pdf"`` or ``"image_test_2"``.
            Stored on every result row so you can filter "what changed at step N"
            in the CSV without needing to join another table.
        new_test_ids:
            Set of ``test_id`` values whose source document was just introduced
            at this milestone.  Rows whose ``test_id`` is in this set get
            ``is_new_test = True``; all other rows get ``is_new_test = False``.

            This split enables two separate analyses of pipeline health:
              - **New-doc performance**: does the retriever surface content from
                a freshly-indexed document?  Tracked via ``is_new_test = True``
                rows across the milestone when the document was added.
              - **Existing-doc drift**: do queries that previously worked still
                return correct answers after the corpus was expanded?  Tracked
                via ``is_new_test = False`` rows across all milestones.  A
                declining pass rate for this group is the key early-warning
                indicator of cross-contamination.

        Returns
        -------
        pd.DataFrame
            Standard ``run()`` output with four prepended milestone columns:

            ===============  ================================================
            milestone_index  1-based corpus-growth step (x-axis in charts)
            doc_count        number of docs in the index at this step
            doc_just_added   document name added at this step
            is_new_test      True iff test_id belongs to the just-added doc
            ===============  ================================================

            Concatenate the return values from successive ``run_at_milestone``
            calls to build the full growth-trend dataset::

                all_results = []
                for step in plan:
                    ...build index...
                    all_results.append(
                        harness.run_at_milestone(suite, milestone_index=i, ...)
                    )
                milestone_df = pd.concat(all_results, ignore_index=True)
        """
        # Execute the full test suite against the current pipeline state.
        # run() times each query, checks min_sources / required_terms / category,
        # and records answer text + confidence — all the per-test raw metrics.
        results = self.run(test_suite)

        # Prepend milestone identifiers at columns 0-2 so that when someone opens
        # the CSV in Excel or pandas the "where in the sequence" context appears
        # before the individual test result columns, making it immediately readable.
        results.insert(0, "milestone_index", milestone_index)
        # doc_count may differ from milestone_index when PDFs and image folders
        # are counted in separate phases; storing it explicitly keeps the CSV
        # self-contained without needing to know the phase structure.
        results.insert(1, "doc_count", doc_count)
        # doc_just_added lets you group all rows where a specific document was
        # the newly-added one, answering "what happened the moment test_7.pdf
        # entered the corpus?"
        results.insert(2, "doc_just_added", doc_just_added)

        # Mark which tests belong to the document that was just indexed.
        # Tests NOT in new_test_ids have been in scope since earlier milestones
        # and are the critical population for detecting retrieval drift.
        results["is_new_test"] = results["test_id"].isin(new_test_ids)

        return results

    @staticmethod
    def summarize_milestones(milestone_df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate per-milestone metrics from a full cumulative regression run.

        Groups the raw per-test-per-milestone rows produced by successive calls
        to ``run_at_milestone`` into one summary row per milestone.  This is the
        primary DataFrame for high-level study of corpus-growth effects and is
        also written as a standalone CSV artifact alongside the raw results.

        Each summary row answers: "At corpus size N (after adding document D),
        how well did the pipeline perform overall, for tests of the just-added
        document alone, and for tests of documents already in the index?"

        The ``existing_test_pass_rate`` column is the most sensitive indicator
        of retrieval drift: if it falls as the corpus grows, it means that adding
        a new document is harming retrieval for previously-indexed content.  A
        stable or rising value confirms the index remains well-partitioned.

        Parameters
        ----------
        milestone_df:
            Concatenated output of multiple ``run_at_milestone`` calls.
            Required columns: milestone_index, doc_count, doc_just_added,
            passed, is_new_test, avg_confidence, response_time_ms.

        Returns
        -------
        pd.DataFrame  sorted by milestone_index, one row per milestone.

        Columns produced
        ----------------
        =========================  =========================================
        milestone_index            1-based corpus-growth step
        doc_count                  number of docs in index at this step
        doc_just_added             name of document added at this step
        total_tests                tests executed at this milestone
        passed_tests               number that passed
        pass_rate                  overall pass rate (%)
        new_test_pass_rate         pass rate for tests of just-added doc;
                                   tells you if the retriever finds new content
        existing_test_pass_rate    pass rate for previously-in-scope tests;
                                   NaN at milestone 1 where all tests are new;
                                   a falling value = cross-contamination signal
        avg_confidence             mean retrieval confidence across all tests
        avg_response_time_ms       mean response time across all tests
        median_response_time_ms    median response time (robust to outliers)
        =========================  =========================================
        """
        # Validate required columns up front so callers get a clear error
        # rather than a cryptic KeyError deep inside a groupby operation.
        required = {
            "milestone_index",
            "doc_count",
            "doc_just_added",
            "passed",
            "is_new_test",
            "avg_confidence",
            "response_time_ms",
        }
        missing = sorted(required.difference(milestone_df.columns))
        if missing:
            raise ValueError(f"milestone_df missing required columns: {', '.join(missing)}")

        rows: List[Dict[str, Any]] = []

        # group by milestone_index (sort=True ensures ascending order so the
        # resulting DataFrame naturally reads as a time series left-to-right).
        for milestone_index, grp in milestone_df.groupby("milestone_index", sort=True):
            # All rows in this group share the same milestone metadata, so
            # taking iloc[0] is safe for the scalar fields.
            doc_count = int(grp["doc_count"].iloc[0])
            doc_just_added = str(grp["doc_just_added"].iloc[0])
            total = len(grp)
            passed = int(grp["passed"].sum())

            # Split the group into the tests that were newly introduced at this
            # milestone vs. those that have been in scope since an earlier step.
            new_mask = grp["is_new_test"].astype(bool)
            new_grp = grp[new_mask]         # tests for just-added document
            existing_grp = grp[~new_mask]   # tests for already-indexed documents

            # Pass rates per sub-group.  Use float("nan") when a group is empty
            # (e.g. milestone 1 has no "existing" tests yet) so that plotting
            # libraries correctly omit those points rather than plotting zeros.
            new_pass_rate = (
                round(float(new_grp["passed"].mean()) * 100.0, 2)
                if len(new_grp) > 0
                else float("nan")
            )
            existing_pass_rate = (
                round(float(existing_grp["passed"].mean()) * 100.0, 2)
                if len(existing_grp) > 0
                else float("nan")
            )

            rows.append(
                {
                    "milestone_index": milestone_index,
                    "doc_count": doc_count,
                    "doc_just_added": doc_just_added,
                    "total_tests": total,
                    "passed_tests": passed,
                    # Overall pass rate: the headline KPI at this corpus size.
                    "pass_rate": round((passed / total) * 100.0, 2),
                    # New-doc pass rate: measures indexing success for fresh content.
                    "new_test_pass_rate": new_pass_rate,
                    # Existing-doc pass rate: the cross-contamination sentinel.
                    "existing_test_pass_rate": existing_pass_rate,
                    # Confidence and latency aggregates for trend charting.
                    "avg_confidence": round(float(grp["avg_confidence"].mean()), 3),
                    "avg_response_time_ms": round(float(grp["response_time_ms"].mean()), 2),
                    # Median latency is less sensitive to a single slow query than
                    # the mean, making it a better indicator of typical performance.
                    "median_response_time_ms": round(float(grp["response_time_ms"].median()), 2),
                }
            )

        return pd.DataFrame(rows)

    # pylint: disable-next=too-many-locals,too-many-statements
    @staticmethod
    def visualize_growth_trends(
        milestone_df: pd.DataFrame,
        output_path: str,
        title: str = "RAG Corpus Growth Regression",
    ) -> str:
        """Generate a 6-panel PNG showing how pipeline metrics evolve as the corpus grows.

        This is the primary visualization for studying regression-under-load.  It
        takes the concatenated ``run_at_milestone`` output and produces a single
        PNG with six panels, each answering a different question about pipeline
        health as documents accumulate in the index.

        Layout (2 rows × 3 columns)::

            ┌─────────────────┬──────────────────┬──────────────────┐
            │  [0,0]          │  [0,1]           │  [0,2]           │
            │  Pass-rate      │  Confidence      │  Latency trend   │
            │  trend          │  trend           │  (mean + median) │
            ├─────────────────┼──────────────────┼──────────────────┤
            │  [1,0]          │  [1,1]           │  [1,2]           │
            │  Pass/Fail      │  Latency box-    │  Source-count    │
            │  heatmap        │  plot per step   │  trend           │
            └─────────────────┴──────────────────┴──────────────────┘

        Panel descriptions
        ------------------
        **[0,0] Pass Rate Trend** — three line series: overall, new-doc tests,
        existing-doc tests.  A stable or rising "existing" line means new
        documents do not contaminate retrieval for already-indexed content.
        A falling "existing" line is the key early-warning indicator of
        cross-contamination or capacity-related retrieval drift.

        **[0,1] Confidence Trend** — mean retrieval confidence per milestone for
        new vs existing tests.  Downward slope in the existing-test series
        indicates BM25 + vector scores are diluting as the corpus expands — the
        retriever is surfacing chunks from new documents that partially match old
        queries, dragging the average score down.

        **[0,2] Latency Trend** — mean and median response time per milestone.
        Should grow gradually (larger FAISS index = slightly slower ANN search).
        A sudden spike at a specific milestone usually corresponds to a
        document that is unusually large or semantically dense.

        **[1,0] Per-Test Pass/Fail Heatmap** — rows = unique test_ids, columns
        = milestone_index values.  Green = passed, Red = failed, White = test
        not yet in corpus.  Reading *across a row* shows whether a specific test
        ever regressed.  Reading *down a column* shows the suite health at each
        step.  This is the most granular view: you can identify exactly which
        test failed at which corpus size, enabling targeted debugging.

        **[1,1] Response-Time Box Plot** — distribution of response times at each
        milestone.  Widening boxes indicate growing latency variance — the
        pipeline is becoming less predictable as the corpus grows.  Outliers at
        a specific milestone typically correspond to a slow-to-retrieve document.

        **[1,2] Source-Count Trend** — average number of source chunks returned
        per query at each milestone.  Too few = low recall.  Too many = precision
        dropping (irrelevant chunks surfacing).  A sudden increase when adding a
        specific document means that document introduces semantically ambiguous
        content that matches many queries.

        Parameters
        ----------
        milestone_df:
            Concatenated ``run_at_milestone`` output covering all milestones.
        output_path:
            File path for the output PNG.  Parent directories are created
            automatically.
        title:
            Figure suptitle displayed at the top of the chart.

        Returns
        -------
        str  Absolute path of the written PNG file.
        """
        # Validate required columns before any plotting logic so the error
        # message is actionable rather than a deep matplotlib traceback.
        required = {
            "milestone_index",
            "test_id",
            "passed",
            "is_new_test",
            "avg_confidence",
            "response_time_ms",
            "num_sources",
        }
        missing = sorted(required.difference(milestone_df.columns))
        if missing:
            raise ValueError(f"milestone_df missing required columns: {', '.join(missing)}")
        if milestone_df.empty:
            raise ValueError("milestone_df is empty; nothing to visualize.")

        # Delay heavy imports so that importing this module in CI (where
        # matplotlib may run headless) does not trigger backend-selection
        # side-effects at module load time.
        import matplotlib.colors as mcolors  # pylint: disable=import-outside-toplevel
        import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel
        import numpy as np  # pylint: disable=import-outside-toplevel

        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Collect the sorted list of milestone indices once; used as the shared
        # x-axis domain across all line charts and the box-plot grouping key.
        milestones = sorted(milestone_df["milestone_index"].unique())

        # ------------------------------------------------------------------ #
        # Pre-compute per-milestone summary statistics for the line charts.   #
        # Each entry in summary_rows becomes one point on the trend lines.    #
        # ------------------------------------------------------------------ #
        summary_rows: List[Dict[str, Any]] = []
        for m in milestones:
            # Slice to just this milestone's results.
            grp = milestone_df[milestone_df["milestone_index"] == m]
            # Split new-doc tests from existing-doc tests using the flag set
            # by run_at_milestone.  This split drives three distinct trend lines
            # in panels [0,0] and [0,1].
            new_mask = grp["is_new_test"].astype(bool)
            new_grp = grp[new_mask]
            existing_grp = grp[~new_mask]

            summary_rows.append(
                {
                    "milestone": m,
                    # Pass rates: multiply mean (0-1) by 100 for a percentage y-axis.
                    "overall_pass_rate": grp["passed"].mean() * 100,
                    "new_pass_rate": (
                        new_grp["passed"].mean() * 100 if len(new_grp) else float("nan")
                    ),
                    "existing_pass_rate": (
                        existing_grp["passed"].mean() * 100 if len(existing_grp) else float("nan")
                    ),
                    # Confidence aggregates — used for panel [0,1].
                    "mean_confidence": grp["avg_confidence"].mean(),
                    "new_confidence": (
                        new_grp["avg_confidence"].mean() if len(new_grp) else float("nan")
                    ),
                    "existing_confidence": (
                        existing_grp["avg_confidence"].mean() if len(existing_grp) else float("nan")
                    ),
                    # Latency — mean for trend line, median as a robust alternative.
                    "mean_latency": grp["response_time_ms"].mean(),
                    "median_latency": grp["response_time_ms"].median(),
                    # Source count — measures retrieval recall/precision over time.
                    "mean_sources": grp["num_sources"].mean(),
                }
            )
        summary = pd.DataFrame(summary_rows)

        # ------------------------------------------------------------------ #
        # Build the heatmap matrix for panel [1,0].                           #
        # Rows = test_ids (ordered by first appearance), Cols = milestones.   #
        # Cell values: 1.0 = passed, 0.0 = failed, NaN = not yet in corpus.  #
        # ------------------------------------------------------------------ #
        # Preserve insertion order of test_ids so that tests for the same doc
        # appear grouped together, making the heatmap easier to interpret.
        seen_ids: dict = {}
        for _, row in milestone_df.sort_values("milestone_index").iterrows():
            seen_ids.setdefault(row["test_id"], None)
        all_test_ids = list(seen_ids)

        # Initialise with NaN (= white on the masked colormap) to represent
        # "test not yet applicable at this milestone."
        heatmap_data = pd.DataFrame(
            float("nan"), index=all_test_ids, columns=milestones, dtype=float
        )
        # Fill in the actual pass/fail values from the milestone results.
        for m in milestones:
            grp = milestone_df[milestone_df["milestone_index"] == m]
            for _, row in grp.iterrows():
                # float() converts numpy bool_ to a plain Python float so that
                # the masked-array logic below works correctly.
                heatmap_data.loc[row["test_id"], m] = float(row["passed"])

        # ------------------------------------------------------------------ #
        # Collect per-milestone latency arrays for the box plot.              #
        # Each element is a 1-D array of response_time_ms values at that step #
        # so that matplotlib can draw the full distribution.                  #
        # ------------------------------------------------------------------ #
        box_data = [
            milestone_df[milestone_df["milestone_index"] == m]["response_time_ms"].values
            for m in milestones
        ]

        # ------------------------------------------------------------------ #
        # Create figure: 2 rows × 3 columns, wide aspect for readability.    #
        # ------------------------------------------------------------------ #
        fig, axes = plt.subplots(2, 3, figsize=(18, 11))
        fig.suptitle(title, fontsize=14, fontweight="bold")

        # ---- Panel [0,0]: Pass Rate Trend ----------------------------------
        # Three series answer three questions simultaneously:
        #   1. Overall: headline KPI — is the pipeline regressing globally?
        #   2. New-doc: can the retriever find content just added to the index?
        #   3. Existing-doc: does adding a new doc hurt previously-passing tests?
        ax = axes[0, 0]
        ax.plot(
            summary["milestone"], summary["overall_pass_rate"],
            "o-", label="Overall", color="#1565c0", linewidth=2,
        )
        ax.plot(
            summary["milestone"], summary["new_pass_rate"],
            "s--", label="New-doc tests", color="#2e7d32", linewidth=1.5,
        )
        ax.plot(
            summary["milestone"], summary["existing_pass_rate"],
            "^--", label="Existing-doc tests", color="#c62828", linewidth=1.5,
        )
        ax.set_title("Pass Rate Trend")
        ax.set_xlabel("Milestone (docs indexed)")
        ax.set_ylabel("Pass Rate (%)")
        # Pin y-axis at 0-105 so that a 100% score is clearly distinguishable
        # from a 99% score (it would otherwise be clipped at the edge).
        ax.set_ylim(0, 105)
        ax.axhline(100, color="#888888", linestyle=":", linewidth=0.8)
        ax.legend(fontsize=8)

        # ---- Panel [0,1]: Confidence Trend ---------------------------------
        # Same three-series structure as the pass-rate panel.  A downward slope
        # in the "existing" series means the hybrid retriever (BM25 + FAISS) is
        # returning chunks from newer documents that partially match old queries,
        # dragging down the average confidence score for well-established tests.
        ax = axes[0, 1]
        ax.plot(
            summary["milestone"], summary["mean_confidence"],
            "o-", label="Overall", color="#6a1b9a", linewidth=2,
        )
        ax.plot(
            summary["milestone"], summary["new_confidence"],
            "s--", label="New-doc tests", color="#2e7d32", linewidth=1.5,
        )
        ax.plot(
            summary["milestone"], summary["existing_confidence"],
            "^--", label="Existing-doc tests", color="#c62828", linewidth=1.5,
        )
        ax.set_title("Retrieval Confidence Trend")
        ax.set_xlabel("Milestone (docs indexed)")
        ax.set_ylabel("Avg Confidence Score")
        ax.legend(fontsize=8)

        # ---- Panel [0,2]: Latency Trend ------------------------------------
        # Plotting both mean and median because response-time distributions are
        # typically right-skewed (a single slow query can inflate the mean).
        # If mean and median diverge at a specific milestone it signals that one
        # query became an outlier, not that the whole suite slowed down.
        ax = axes[0, 2]
        ax.plot(
            summary["milestone"], summary["mean_latency"],
            "o-", label="Mean latency", color="#ef6c00", linewidth=2,
        )
        ax.plot(
            summary["milestone"], summary["median_latency"],
            "s--", label="Median latency", color="#e64a19", linewidth=1.5,
        )
        ax.set_title("Response Latency Trend")
        ax.set_xlabel("Milestone (docs indexed)")
        ax.set_ylabel("Response Time (ms)")
        ax.legend(fontsize=8)

        # ---- Panel [1,0]: Per-Test Pass/Fail Heatmap -----------------------
        # This is the highest-resolution view: each cell shows the outcome of
        # one specific test at one specific corpus size.  White cells mark steps
        # where the test's source document was not yet in the index.
        #
        # How to read the heatmap:
        #   • A row that is all-green = that test has never failed.
        #   • A row with a red cell at milestone N = that test regressed when
        #     document D was added (D = doc_just_added at milestone N).
        #   • A column with many red cells = that corpus state was globally bad.
        ax = axes[1, 0]
        n_tests = len(all_test_ids)
        n_milestones = len(milestones)

        # Binary colormap: 0 = red (fail), 1 = green (pass).  NaN cells will be
        # rendered white by set_bad(), representing "not yet applicable."
        cmap = mcolors.ListedColormap(["#c62828", "#2e7d32"])
        cmap.set_bad(color="white")

        heatmap_array = heatmap_data.values.astype(float)
        # np.ma.masked_invalid converts NaN to masked values so imshow renders
        # them using set_bad() colour rather than clamping to 0 (= red).
        masked = np.ma.masked_invalid(heatmap_array)
        ax.imshow(masked, cmap=cmap, vmin=0, vmax=1, aspect="auto")

        ax.set_title("Per-Test Pass/Fail Heatmap\n(green=pass, red=fail, white=not yet indexed)")
        ax.set_xlabel("Milestone")
        ax.set_ylabel("Test ID")
        ax.set_xticks(range(n_milestones))
        ax.set_xticklabels([str(m) for m in milestones], fontsize=7)

        # Limit y-tick labels when there are many tests to avoid overlapping.
        if n_tests <= 30:
            ax.set_yticks(range(n_tests))
            ax.set_yticklabels(all_test_ids, fontsize=6)
        else:
            step = max(1, n_tests // 15)
            ax.set_yticks(range(0, n_tests, step))
            ax.set_yticklabels(
                [all_test_ids[i] for i in range(0, n_tests, step)], fontsize=6
            )

        # ---- Panel [1,1]: Latency Box Plot ---------------------------------
        # Box plots show the full distribution of response times at each
        # milestone rather than just the mean.  Widening IQR (taller boxes)
        # means the pipeline is becoming less predictable; narrow boxes indicate
        # consistent, well-behaved latency.
        ax = axes[1, 1]
        bp = ax.boxplot(
            box_data,
            tick_labels=[str(m) for m in milestones],
            patch_artist=True,
        )
        for patch in bp["boxes"]:
            patch.set_facecolor("#bbdefb")
        ax.set_title("Latency Distribution per Milestone")
        ax.set_xlabel("Milestone")
        ax.set_ylabel("Response Time (ms)")
        ax.tick_params(axis="x", rotation=45)

        # ---- Panel [1,2]: Source-Count Trend -------------------------------
        # num_sources = number of chunks the retriever returned for a query.
        # Fewer chunks = the retriever is being selective (good precision) but
        # risks missing relevant content (low recall).  More chunks = higher
        # recall but the answer may incorporate noise from irrelevant documents.
        # A sudden jump at milestone N usually indicates that document N contains
        # semantically ambiguous content that matches many existing queries.
        ax = axes[1, 2]
        ax.plot(
            summary["milestone"], summary["mean_sources"],
            "o-", color="#00838f", linewidth=2,
        )
        ax.set_title("Avg Source Chunks Returned per Query")
        ax.set_xlabel("Milestone (docs indexed)")
        ax.set_ylabel("Avg num_sources")
        # Horizontal reference at 1.0: queries should return at least one source.
        ax.axhline(1.0, color="#888888", linestyle=":", linewidth=0.8, label="min=1")
        ax.legend(fontsize=8)

        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        return str(out_path)

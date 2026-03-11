"""Utilities for pandas-driven regression testing of the RAG pipeline.

This module provides a lightweight harness that:
1) executes a suite of query test cases,
2) records retrieval/answer metrics in a DataFrame,
3) compares current runs to a baseline to flag regressions.
"""

from __future__ import annotations

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

        return suite_df

    def run(self, test_suite: pd.DataFrame) -> pd.DataFrame:
        """Execute each test in the suite and return a results DataFrame."""
        rows: List[Dict[str, Any]] = []

        for _, test in test_suite.iterrows():
            started = datetime.now(timezone.utc)
            payload = self.rag.query_with_sources(
                test["query"],
                classify=bool(test["classify"]),
                expand=bool(test["expand"]),
                num_expansions=int(test["num_expansions"]),
            )
            ended = datetime.now(timezone.utc)

            answer = str(payload.get("answer", ""))
            sources = payload.get("sources", []) or []
            query_category = payload.get("query_category")

            source_scores = [s.get("score") for s in sources if s.get("score") is not None]
            avg_confidence = float(sum(source_scores) / len(source_scores)) if source_scores else 0.0

            min_sources = int(test.get("min_sources", 1))
            has_min_sources = len(sources) >= min_sources
            category_ok = self._category_match(query_category, test.get("expected_query_category"))
            terms_ok = self._required_terms_match(answer, str(test.get("required_terms", "")))

            row = {
                "timestamp": ended,
                "test_id": test["test_id"],
                "criticality": test.get("criticality", "medium"),
                "query": test["query"],
                "answer": answer,
                "answer_length": len(answer),
                "query_category": query_category,
                "num_sources": len(sources),
                "avg_confidence": round(avg_confidence, 3),
                "response_time_ms": (ended - started).total_seconds() * 1000.0,
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
        import matplotlib.pyplot as plt

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
        import matplotlib.pyplot as plt

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
        terms = [t.strip().lower() for t in required_terms.split("|") if t.strip()]
        if not terms:
            return True
        answer_lower = answer.lower()
        return any(term in answer_lower for term in terms)

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

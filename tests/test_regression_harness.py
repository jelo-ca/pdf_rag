"""Tests for pandas-based regression harness."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import matplotlib
import pandas as pd

from rag.regression import RAGRegressionHarness

matplotlib.use("Agg")


@dataclass
class FakeRAG:
    """Small deterministic fake that mimics query_with_sources."""

    responses: Dict[str, Dict[str, Any]]

    def query_with_sources(
        self,
        question: str,
        classify: bool = False,
        expand: bool = False,
        num_expansions: int = 3,
    ) -> Dict[str, Any]:
        _ = classify, expand, num_expansions
        return self.responses[question]


def test_create_test_suite_applies_defaults() -> None:
    suite = RAGRegressionHarness.create_test_suite(
        [
            {"test_id": "TC-001", "query": "What is the batch number?"},
        ]
    )

    assert isinstance(suite, pd.DataFrame)
    assert suite.iloc[0]["min_sources"] == 1
    assert suite.iloc[0]["required_terms"] == ""
    assert bool(suite.iloc[0]["classify"]) is False


def test_run_generates_pass_fail_metrics() -> None:
    # Responses mirror what the real pipeline returns for Sample1.pdf
    # (Safety Data Sheet — Pfizer-BioNTech COVID-19 Vaccine).
    # classify=False so query_category is None for all cases.
    fake = FakeRAG(
        responses={
            "What is the product name and product code?": {
                "answer": "The product name is Pfizer-BioNTech COVID-19 Vaccine (Comirnaty) with product code PF00092.",
                "query_category": None,
                "sources": [{"score": 95.0, "file": "Sample1.pdf", "page": 1}],
            },
            "What are the storage conditions for this product?": {
                "answer": "Store as directed by product packaging.",
                "query_category": None,
                "sources": [{"score": 88.0, "file": "Sample1.pdf", "page": 5}],
            },
            "What fire extinguishing media should be used?": {
                "answer": "Use dry chemical, CO2, alcohol-resistant foam or water spray.",
                "query_category": None,
                "sources": [{"score": 85.0, "file": "Sample1.pdf", "page": 4}],
            },
            "Is this product regulated for transport?": {
                "answer": "The product is not regulated for transport under USDOT, EUADR, IATA, or IMDG regulations.",
                "query_category": None,
                "sources": [{"score": 90.0, "file": "Sample1.pdf", "page": 11}],
            },
            "What is the batch number or lot number?": {
                # Correct pipeline behaviour: acknowledge absence rather than hallucinate.
                "answer": "There is no batch number or lot number mentioned in this Safety Data Sheet.",
                "query_category": None,
                "sources": [],
            },
        }
    )
    harness = RAGRegressionHarness(fake)

    suite = RAGRegressionHarness.create_test_suite(
        [
            {
                "test_id": "TC-001",
                "query": "What is the product name and product code?",
                "required_terms": "pfizer|comirnaty|covid|PF00092",
                "min_sources": 1,
                "classify": False,
                "criticality": "high",
            },
            {
                "test_id": "TC-002",
                "query": "What are the storage conditions for this product?",
                "required_terms": "store|directed|packaging",
                "min_sources": 1,
                "classify": False,
                "criticality": "high",
            },
            {
                "test_id": "TC-003",
                "query": "What fire extinguishing media should be used?",
                "required_terms": "CO2|chemical|foam|water",
                "min_sources": 1,
                "classify": False,
                "criticality": "medium",
            },
            {
                "test_id": "TC-004",
                "query": "Is this product regulated for transport?",
                "required_terms": "not regulated|not applicable",
                "min_sources": 1,
                "classify": False,
                "criticality": "medium",
            },
            {
                # Hallucination-resistance check — SDS has no batch/lot number.
                "test_id": "TC-005",
                "query": "What is the batch number or lot number?",
                "required_terms": "no batch|not mentioned|not available|not provided|not provide|not specified|safety data",
                "min_sources": 0,
                "classify": False,
                "criticality": "high",
            },
        ]
    )
    results = harness.run(suite)

    assert len(results) == 5
    for _, row in results.iterrows():
        assert bool(row["has_min_sources"]) is True, f"{row['test_id']} missing sources"
        assert bool(row["category_match"]) is True, f"{row['test_id']} category mismatch"
        assert bool(row["required_terms_match"]) is True, f"{row['test_id']} required terms not found"
        assert bool(row["passed"]) is True, f"{row['test_id']} unexpectedly failed"


def test_compare_to_baseline_flags_regression() -> None:
    current_results = pd.DataFrame(
        [
            {
                "test_id": "TC-001",
                "answer": "Completely different answer.",
                "passed": False,
                "avg_confidence": 45.0,
                "response_time_ms": 6000.0,
                "num_sources": 1,
            }
        ]
    )
    baseline_results = pd.DataFrame(
        [
            {
                "test_id": "TC-001",
                "answer": "Batch number is 12345.",
                "passed": True,
                "avg_confidence": 90.0,
                "response_time_ms": 1200.0,
                "num_sources": 2,
            }
        ]
    )

    fake = FakeRAG(responses={})
    harness = RAGRegressionHarness(fake)
    comparison = harness.compare_to_baseline(current_results, baseline_results)

    assert len(comparison) == 1
    row = comparison.iloc[0]
    assert bool(row["regression_detected"]) is True


def test_summarize_outputs_expected_aggregates() -> None:
    results = pd.DataFrame(
        [
            {"passed": True, "response_time_ms": 1000.0, "avg_confidence": 80.0},
            {"passed": False, "response_time_ms": 2000.0, "avg_confidence": 60.0},
        ]
    )
    summary = RAGRegressionHarness.summarize(results)

    assert summary["total_tests"] == 2
    assert summary["passed_tests"] == 1
    assert summary["failed_tests"] == 1
    assert summary["pass_rate"] == 50.0
    assert summary["avg_response_time_ms"] == 1500.0
    assert summary["avg_confidence"] == 70.0


def test_save_and_load_roundtrip(tmp_path) -> None:
    rows: List[Dict[str, Any]] = [
        {"test_id": "TC-1", "passed": True, "response_time_ms": 123.0, "avg_confidence": 88.0}
    ]
    results = pd.DataFrame(rows)
    out_path = tmp_path / "regression_results.csv"

    RAGRegressionHarness.save_results(results, str(out_path))
    loaded = RAGRegressionHarness.load_results(str(out_path))

    assert loaded.to_dict(orient="records") == results.to_dict(orient="records")


def test_visualize_results_creates_png(tmp_path) -> None:
    results = pd.DataFrame(
        [
            {
                "test_id": "TC-001",
                "passed": True,
                "response_time_ms": 850.0,
                "avg_confidence": 88.0,
            },
            {
                "test_id": "TC-002",
                "passed": False,
                "response_time_ms": 1320.0,
                "avg_confidence": 54.0,
            },
        ]
    )
    out_path = tmp_path / "results_dashboard.png"

    generated = RAGRegressionHarness.visualize_results(results, str(out_path))

    assert generated.endswith("results_dashboard.png")
    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_visualize_comparison_creates_png(tmp_path) -> None:
    comparison = pd.DataFrame(
        [
            {
                "test_id": "TC-001",
                "answer_similarity": 0.92,
                "confidence_delta": -2.0,
                "response_time_delta_ms": 120.0,
                "regression_detected": False,
            },
            {
                "test_id": "TC-002",
                "answer_similarity": 0.41,
                "confidence_delta": -30.0,
                "response_time_delta_ms": 3600.0,
                "regression_detected": True,
            },
        ]
    )
    out_path = tmp_path / "comparison_dashboard.png"

    generated = RAGRegressionHarness.visualize_comparison(comparison, str(out_path))

    assert generated.endswith("comparison_dashboard.png")
    assert out_path.exists()
    assert out_path.stat().st_size > 0

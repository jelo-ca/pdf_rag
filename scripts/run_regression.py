"""Run RAG regression tests and generate CSV/PNG artifacts.

Usage examples:
    python scripts/run_regression.py --pdf docs/Sample1.pdf
    python scripts/run_regression.py --pdf docs/Sample1.pdf --baseline artifacts/baseline_results.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from rag import RAGPipeline, RAGRegressionHarness


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RAG regression suite and visualize results.")
    parser.add_argument(
        "--pdf",
        type=str,
        default="docs/Sample1.pdf",
        help="Path to source PDF for indexing.",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=str,
        default="artifacts",
        help="Directory where CSV/PNG outputs will be written.",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default="",
        help="Optional baseline CSV path for comparison visualization.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="",
        help="Optional GGUF path override. If omitted, RAGPipeline uses env/default resolution.",
    )
    return parser.parse_args()


def build_suite() -> list[dict]:
    """Regression suite for Sample1.pdf (Safety Data Sheet — Pfizer-BioNTech COVID-19 Vaccine).

    Test design rationale:
    - classify=False: The SDS format does not align with the pipeline's pharma
      category taxonomy (CoA, Packaging Spec, etc.), so query-category filtering
      would exclude valid chunks and degrade recall.
    - required_terms use pipe-separated alternatives; any single match counts.
    - TC-005 is a hallucination-resistance check: the SDS contains no batch
      number, so a correct pipeline response must acknowledge the absence.
    """
    return [
        {
            # Source: SDS Section 1 — product identifier block (page 1).
            "test_id": "TC-001",
            "query": "What is the product name and product code?",
            "required_terms": "pfizer|comirnaty|covid|PF00092",
            "min_sources": 1,
            "classify": False,
            "criticality": "high",
        },
        {
            # Source: SDS Section 7 — handling and storage (page 5).
            # "Store as directed by product packaging." is the verbatim text.
            # required_terms deliberately exclude generic substring "condition"
            # to prevent a "no information on conditions" answer from passing.
            "test_id": "TC-002",
            "query": "What are the storage conditions for this product?",
            "required_terms": "store|directed|packaging",
            "min_sources": 1,
            "classify": False,
            "criticality": "high",
        },
        {
            # Source: SDS Section 5 — fire-fighting measures (page 4).
            "test_id": "TC-003",
            "query": "What fire extinguishing media should be used?",
            "required_terms": "CO2|chemical|foam|water",
            "min_sources": 1,
            "classify": False,
            "criticality": "medium",
        },
        {
            # Source: SDS Section 14 — transport information (page 11).
            "test_id": "TC-004",
            "query": "Is this product regulated for transport?",
            "required_terms": "not regulated|not applicable",
            "min_sources": 1,
            "classify": False,
            "criticality": "medium",
        },
        {
            # Hallucination-resistance check: SDS documents do not carry batch
            # or lot numbers.  A correct pipeline response must acknowledge the
            # absence rather than fabricate a value.  min_sources=0 because
            # legitimate retrievals may return zero chunks for absent data.
            "test_id": "TC-005",
            "query": "What is the batch number or lot number?",
            "required_terms": "no batch|not mentioned|not available|not provided|not provide|not specified|safety data",
            "min_sources": 0,
            "classify": False,
            "criticality": "high",
        },
    ]


def main() -> int:
    args = parse_args()

    artifacts_dir = Path(args.artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    pipeline_kwargs = {}
    if args.model_path:
        pipeline_kwargs["model_path"] = args.model_path

    rag = RAGPipeline(**pipeline_kwargs)
    rag.build(args.pdf, classify_docs=True)

    harness = RAGRegressionHarness(rag)
    suite_df = harness.create_test_suite(build_suite())
    results_df = harness.run(suite_df)

    results_csv = artifacts_dir / "regression_results.csv"
    dashboard_png = artifacts_dir / "regression_dashboard.png"

    harness.save_results(results_df, str(results_csv))
    harness.visualize_results(results_df, str(dashboard_png))

    summary = harness.summarize(results_df)
    print("Regression summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")

    print(f"Saved results CSV: {results_csv}")
    print(f"Saved dashboard PNG: {dashboard_png}")

    if args.baseline:
        baseline_df = harness.load_results(args.baseline)
        comparison_df = harness.compare_to_baseline(results_df, baseline_df)
        comparison_csv = artifacts_dir / "baseline_comparison.csv"
        comparison_png = artifacts_dir / "baseline_comparison_dashboard.png"

        comparison_df.to_csv(comparison_csv, index=False)
        harness.visualize_comparison(comparison_df, str(comparison_png))

        print(f"Saved comparison CSV: {comparison_csv}")
        print(f"Saved comparison PNG: {comparison_png}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

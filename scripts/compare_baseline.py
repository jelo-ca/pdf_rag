"""Compare current regression CSV with baseline and generate visualization.

Usage:
    python scripts/compare_baseline.py \
        --current artifacts/regression_results.csv \
        --baseline artifacts/baseline_results.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from rag import RAGRegressionHarness


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare regression run to baseline and visualize drift.")
    parser.add_argument("--current", type=str, required=True, help="Current regression results CSV.")
    parser.add_argument("--baseline", type=str, required=True, help="Baseline regression results CSV.")
    parser.add_argument(
        "--artifacts-dir",
        type=str,
        default="artifacts",
        help="Directory where comparison CSV/PNG outputs are written.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    artifacts_dir = Path(args.artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Pipeline is not used here, but harness expects a pipeline argument.
    harness = RAGRegressionHarness(object())

    current_df = harness.load_results(args.current)
    baseline_df = harness.load_results(args.baseline)

    comparison_df = harness.compare_to_baseline(current_df, baseline_df)

    comparison_csv = artifacts_dir / "baseline_comparison.csv"
    comparison_png = artifacts_dir / "baseline_comparison_dashboard.png"

    comparison_df.to_csv(comparison_csv, index=False)
    harness.visualize_comparison(comparison_df, str(comparison_png))

    regressions = int(comparison_df["regression_detected"].sum())
    print(f"Compared {len(comparison_df)} test(s). Regressions detected: {regressions}")
    print(f"Saved comparison CSV: {comparison_csv}")
    print(f"Saved comparison PNG: {comparison_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Run RAG regression tests across all docs/ PDFs and generate combined CSV/PNG artifacts.

Usage examples:
    python scripts/run_regression.py
    python scripts/run_regression.py --docs-dir docs --artifacts-dir artifacts
    python scripts/run_regression.py --baseline artifacts/baseline_results.csv
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
    parser = argparse.ArgumentParser(
        description="Run RAG regression suite across all PDFs in docs/ and visualize results."
    )
    parser.add_argument(
        "--docs-dir",
        type=str,
        default="docs",
        help="Directory containing source PDFs to index.",
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


# ---------------------------------------------------------------------------
# Per-document test suites
# ---------------------------------------------------------------------------

def suite_test1() -> list[dict]:
    """Tests for test_1.pdf — Pfizer-BioNTech COVID-19 Vaccine Safety Data Sheet.

    Product: Comirnaty / PF00092
    Key facts: lipid nanoparticle formulation, pH 7.4, not classified as hazardous,
               no batch/lot number on SDS, not regulated for transport.
    """
    return [
        {
            # Section 1: Product identifier
            "test_id": "T1-001",
            "query": "What is the product name and product code for the COVID-19 vaccine?",
            "required_terms": "pfizer|comirnaty|covid|PF00092",
            "min_sources": 1,
            "classify": False,
            "criticality": "high",
        },
        {
            # Section 7: Handling and storage
            "test_id": "T1-002",
            "query": "What are the storage conditions for the COVID-19 vaccine product?",
            "required_terms": "store|directed|packaging",
            "min_sources": 1,
            "classify": False,
            "criticality": "high",
        },
        {
            # Section 5: Fire-fighting measures
            "test_id": "T1-003",
            "query": "What fire extinguishing media should be used for the COVID-19 vaccine?",
            "required_terms": "CO2|chemical|foam|water",
            "min_sources": 1,
            "classify": False,
            "criticality": "medium",
        },
        {
            # Section 14: Transport information
            "test_id": "T1-004",
            "query": "Is the COVID-19 vaccine regulated for transport under DOT or IATA?",
            "required_terms": "not regulated|not applicable",
            "min_sources": 1,
            "classify": False,
            "criticality": "medium",
        },
        {
            # Section 9: Physical/chemical properties — formulation type
            "test_id": "T1-005",
            "query": "What is the chemical family or formulation type of the COVID-19 vaccine?",
            "required_terms": "lipid|nanoparticle|LNP|mRNA",
            "min_sources": 1,
            "classify": False,
            "criticality": "medium",
        },
        {
            # Hallucination-resistance: SDS does not contain batch/lot numbers
            "test_id": "T1-006",
            "query": "What is the batch number or lot number of the COVID-19 vaccine?",
            "required_terms": "no batch|not mentioned|not available|not provided|not provide|not specified|safety data",
            "min_sources": 0,
            "classify": False,
            "criticality": "high",
        },
    ]


def suite_test2() -> list[dict]:
    """Tests for test_2.pdf — Paracetamol Solution for Infusion Safety Data Sheet.

    Product: PZ02462 (Perfalgan / paracetamol 10 mg/mL)
    Key facts: analgesic/antipyretic, CAS 103-90-2, MW 151.2,
               molecular formula C8H9NO2, overdose risk (liver damage).
    """
    return [
        {
            # Section 1: Product identifier — trade name and product code
            "test_id": "T2-001",
            "query": "What is the product name and product code for the paracetamol infusion?",
            "required_terms": "paracetamol|perfalgan|PZ02462",
            "min_sources": 1,
            "classify": False,
            "criticality": "high",
        },
        {
            # Section 1: Recommended use
            "test_id": "T2-002",
            "query": "What is the intended use or therapeutic category of the paracetamol product?",
            "required_terms": "analgesic|antipyretic|pain|fever",
            "min_sources": 1,
            "classify": False,
            "criticality": "high",
        },
        {
            # Section 3: CAS number of active ingredient
            "test_id": "T2-003",
            "query": "What is the CAS number of the active ingredient in the paracetamol infusion?",
            "required_terms": "103-90-2",
            "min_sources": 1,
            "classify": False,
            "criticality": "medium",
        },
        {
            # Section 9: Molecular formula
            "test_id": "T2-004",
            "query": "What is the molecular formula of paracetamol?",
            "required_terms": "C8H9NO2",
            "min_sources": 1,
            "classify": False,
            "criticality": "medium",
        },
        {
            # Section 11: Toxicological information — overdose effects
            "test_id": "T2-005",
            "query": "What are the clinical effects of a paracetamol overdose?",
            "required_terms": "liver|hepatic|overdose|toxicity",
            "min_sources": 1,
            "classify": False,
            "criticality": "high",
        },
        {
            # Hallucination-resistance: SDS does not list batch/lot numbers
            "test_id": "T2-006",
            "query": "What is the batch number or lot number of the paracetamol infusion?",
            "required_terms": "no batch|not mentioned|not available|not provided|not provide|not specified|safety data",
            "min_sources": 0,
            "classify": False,
            "criticality": "high",
        },
    ]


def suite_test3() -> list[dict]:
    """Tests for test_3.pdf — Zoledronic Acid Injection Safety Data Sheet.

    Product: PZ01101
    Key facts: bisphosphonate, reproductive toxicity H360FD, signal word "Danger",
               pH 6.2, Pfizer OEL 4 µg/m3, clear colorless solution.
    """
    return [
        {
            # Section 1 + Section 3: Product name and chemical family
            "test_id": "T3-001",
            "query": "What is the product name and chemical family of the zoledronic acid product?",
            "required_terms": "zoledronic|bisphosphonate|PZ01101",
            "min_sources": 1,
            "classify": False,
            "criticality": "high",
        },
        {
            # Section 2: GHS hazard classification — reproductive toxicity
            "test_id": "T3-002",
            "query": "What hazard classification applies to the zoledronic acid product?",
            "required_terms": "H360|reproductive|danger|toxic",
            "min_sources": 1,
            "classify": False,
            "criticality": "high",
        },
        {
            # Section 2: GHS signal word
            "test_id": "T3-003",
            "query": "What is the GHS signal word for zoledronic acid?",
            "required_terms": "danger",
            "min_sources": 1,
            "classify": False,
            "criticality": "medium",
        },
        {
            # Section 9: Physical/chemical properties — pH
            "test_id": "T3-004",
            "query": "What is the pH of the zoledronic acid injection solution?",
            "required_terms": "6.2|6",
            "min_sources": 1,
            "classify": False,
            "criticality": "medium",
        },
        {
            # Section 8: Pfizer Occupational Exposure Limit
            "test_id": "T3-005",
            "query": "What is the Pfizer occupational exposure limit (OEL) for zoledronic acid?",
            "required_terms": "4|OEL|exposure limit|µg/m3|ug/m3",
            "min_sources": 1,
            "classify": False,
            "criticality": "high",
        },
        {
            # Section 14: Transport information
            "test_id": "T3-006",
            "query": "Is zoledronic acid regulated for transport?",
            "required_terms": "not regulated|not applicable|not classified",
            "min_sources": 1,
            "classify": False,
            "criticality": "medium",
        },
    ]


def suite_test4() -> list[dict]:
    """Tests for test_4.pdf — Ciprofloxacin Injection Safety Data Sheet.

    Product: PZ01031
    Key facts: fluoroquinolone antibiotic, pH 3.3-3.9, Pfizer OEL 600 µg/m3,
               aquatic toxicity H401/H411, tendonitis/tendon rupture clinical effect.
    """
    return [
        {
            # Section 1 + Section 3: Product name and chemical family
            "test_id": "T4-001",
            "query": "What is the product name and chemical family of the ciprofloxacin product?",
            "required_terms": "ciprofloxacin|fluoroquinolone|PZ01031",
            "min_sources": 1,
            "classify": False,
            "criticality": "high",
        },
        {
            # Section 1: Recommended use
            "test_id": "T4-002",
            "query": "What is the recommended use or therapeutic category of ciprofloxacin injection?",
            "required_terms": "antibiotic|antibacterial|infection|antimicrobial",
            "min_sources": 1,
            "classify": False,
            "criticality": "high",
        },
        {
            # Section 9: pH
            "test_id": "T4-003",
            "query": "What is the pH range of the ciprofloxacin injection solution?",
            "required_terms": "3.3|3.9|3",
            "min_sources": 1,
            "classify": False,
            "criticality": "medium",
        },
        {
            # Section 8: OEL
            "test_id": "T4-004",
            "query": "What is the Pfizer occupational exposure limit for ciprofloxacin?",
            "required_terms": "600|OEL|exposure limit|µg/m3|ug/m3",
            "min_sources": 1,
            "classify": False,
            "criticality": "high",
        },
        {
            # Section 12: Aquatic toxicity hazard codes
            "test_id": "T4-005",
            "query": "What aquatic environmental hazards are associated with ciprofloxacin?",
            "required_terms": "H401|H411|aquatic|toxic to aquatic",
            "min_sources": 1,
            "classify": False,
            "criticality": "medium",
        },
        {
            # Section 11: Known clinical effects — musculoskeletal
            "test_id": "T4-006",
            "query": "What are the known clinical effects of ciprofloxacin on tendons?",
            "required_terms": "tendon|tendonitis|rupture|musculoskeletal",
            "min_sources": 1,
            "classify": False,
            "criticality": "high",
        },
    ]


def suite_test5() -> list[dict]:
    """Tests for test_5.pdf — Cytiva AKTA ready Flow Kit supplier documents.

    Documents: Certificate of Quality (CoQ), packaging spec, BSE/TSE declaration,
               supplier qualification, chain of custody.
    Key facts: lot numbers 18356721 (Low Flow) / 15102934 (High Flow),
               storage >+5°C, operating +2°C to +40°C, blister changed PVC→PETG,
               spec PKG-SPEC-2023-0847, supplier Cytiva Sweden AB (ISO 9001/13485),
               chain of custody from Eysins Switzerland.
    """
    return [
        {
            # CoQ: storage temperature
            "test_id": "T5-001",
            "query": "What are the recommended storage and operating temperature conditions for the AKTA ready flow kit?",
            "required_terms": "+5|5°C|+2|+40|temperature",
            "min_sources": 1,
            "classify": False,
            "criticality": "high",
        },
        {
            # CoQ: lot numbers for both kits
            "test_id": "T5-002",
            "query": "What are the lot numbers for the AKTA ready High Flow and Low Flow kits?",
            "required_terms": "18356721|15102934|lot",
            "min_sources": 1,
            "classify": False,
            "criticality": "high",
        },
        {
            # BSE/TSE declaration: no animal-origin materials
            "test_id": "T5-003",
            "query": "Does the AKTA ready flow kit contain any materials of animal origin?",
            "required_terms": "no animal|animal origin|BSE|TSE|not derived",
            "min_sources": 1,
            "classify": False,
            "criticality": "high",
        },
        {
            # Packaging spec: blister material change
            "test_id": "T5-004",
            "query": "What change was made to the blister packaging material for the AKTA ready kit?",
            "required_terms": "PVC|PETG|blister|packaging|material change",
            "min_sources": 1,
            "classify": False,
            "criticality": "medium",
        },
        {
            # Supplier qualification: certifications
            "test_id": "T5-005",
            "query": "What quality certifications does the supplier Cytiva hold?",
            "required_terms": "ISO 9001|ISO 13485|Cytiva|quality",
            "min_sources": 1,
            "classify": False,
            "criticality": "medium",
        },
        {
            # Chain of custody: origin location
            "test_id": "T5-006",
            "query": "From which location or country did the AKTA ready kit shipment originate?",
            "required_terms": "Eysins|Switzerland|Sweden|Cytiva",
            "min_sources": 1,
            "classify": False,
            "criticality": "medium",
        },
    ]


def build_combined_suite() -> list[dict]:
    """Combine per-document suites into a single regression suite."""
    return (
        suite_test1()
        + suite_test2()
        + suite_test3()
        + suite_test4()
        + suite_test5()
    )


def main() -> int:
    args = parse_args()

    artifacts_dir = Path(args.artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    docs_dir = Path(args.docs_dir)
    pdf_files = sorted(docs_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"ERROR: No PDF files found in {docs_dir.resolve()}", file=sys.stderr)
        return 1

    print(f"Indexing {len(pdf_files)} PDF(s):")
    for p in pdf_files:
        print(f"  {p}")

    pipeline_kwargs = {}
    if args.model_path:
        pipeline_kwargs["model_path"] = args.model_path

    rag = RAGPipeline(**pipeline_kwargs)
    rag.build_from_multiple_pdfs([str(p) for p in pdf_files], classify_docs=True)

    harness = RAGRegressionHarness(rag)
    suite_df = harness.create_test_suite(build_combined_suite())
    results_df = harness.run(suite_df)

    results_csv = artifacts_dir / "regression_results.csv"
    dashboard_png = artifacts_dir / "regression_dashboard.png"

    harness.save_results(results_df, str(results_csv))
    harness.visualize_results(
        results_df,
        str(dashboard_png),
        title=f"RAG Regression — {len(pdf_files)} Documents ({len(results_df)} Tests)",
    )

    summary = harness.summarize(results_df)
    print("\nRegression summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")

    print(f"\nSaved results CSV : {results_csv}")
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

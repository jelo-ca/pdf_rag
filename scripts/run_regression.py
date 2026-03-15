"""Run RAG regression tests across all docs/ PDFs and generate combined CSV/PNG artifacts.

Usage examples:
    python scripts/run_regression.py
    python scripts/run_regression.py --docs-dir docs --artifacts-dir artifacts
    python scripts/run_regression.py --baseline artifacts/baseline_results.csv
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

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
    parser.add_argument(
        "--index-dir",
        type=str,
        default="",
        help=(
            "Optional parent directory for temporary regression indexes. "
            "PDF and OCR phases rebuild a growing shared corpus in temporary "
            "persisted indexes to simulate how the app accumulates data over time."
        ),
    )
    parser.add_argument(
        "--latency-profile",
        choices=["retrieval_only", "retrieval_plus_classify", "retrieval_plus_expand", "full"],
        default="full",
        help=(
            "Pipeline stages to exercise per test. "
            "'retrieval_only' = no classify, no expand, no generation timing difference; "
            "'retrieval_plus_classify' = classify=True, expand=False; "
            "'retrieval_plus_expand' = classify=False, expand=True; "
            "'full' = classify=True, expand=True (original behaviour)."
        ),
    )
    parser.add_argument(
        "--retrieval-mode",
        choices=["bm25", "vector", "hybrid", "all"],
        default="hybrid",
        help=(
            "Retriever to use. "
            "'bm25' = BM25-only; 'vector' = vector-only; 'hybrid' = RRF fusion (default); "
            "'all' = run each mode separately and concatenate results for comparison."
        ),
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


def suite_test7() -> list[dict]:
    """Tests for test_7.pdf — BioNTech COVID-19 mRNA Vaccine Electronic Protocol (Lot FE3592).

    Document type: Batch release / Electronic Protocol submission (For Release).
    Key facts: Lot FE3592, manufactured 30-Jun-2021, expiry 30-Nov-2021,
               trade name COMIRNATY, STN 125742_0, License No. 2229,
               manufacturer Pharmacia & Upjohn / BioNTech Manufacturing GmbH.
    """
    return [
        {
            "test_id": "T7-001",
            "query": "What is the lot number for this COVID-19 vaccine batch release protocol?",
            "required_terms": "FE3592|lot",
            "min_sources": 1,
            "classify": False,
            "criticality": "high",
        },
        {
            "test_id": "T7-002",
            "query": "What is the trade name of the vaccine in this batch protocol?",
            "required_terms": "COMIRNATY|comirnaty",
            "min_sources": 1,
            "classify": False,
            "criticality": "high",
        },
        {
            "test_id": "T7-003",
            "query": "What is the license number stated in this electronic protocol?",
            "required_terms": "2229|license",
            "min_sources": 1,
            "classify": False,
            "criticality": "medium",
        },
        {
            "test_id": "T7-004",
            "query": "Who is the manufacturer or company named in this batch protocol?",
            "required_terms": "BioNTech|Pharmacia|Upjohn|Kalamazoo",
            "min_sources": 1,
            "classify": False,
            "criticality": "high",
        },
        {
            "test_id": "T7-005",
            "query": "What is the date of manufacture for lot FE3592?",
            "required_terms": "30-Jun-2021|Jun-2021|June 2021|30 Jun|2021",
            "min_sources": 1,
            "classify": False,
            "criticality": "high",
        },
        {
            "test_id": "T7-006",
            "query": "What is the expiration date for this vaccine lot?",
            "required_terms": "30-Nov-2021|Nov-2021|November 2021|30 Nov|2021",
            "min_sources": 1,
            "classify": False,
            "criticality": "high",
        },
    ]


def suite_test8() -> list[dict]:
    """Tests for test_8.pdf — BioNTech COVID-19 mRNA Vaccine Corrected Protocol (Lot FD7220).

    Document type: Corrected Electronic Protocol submission (For Licensing Action).
    Key facts: Lot FD7220, manufactured 23-Jun-2021, expiry 30-Nov-2021,
               trade name COMIRNATY, corrected protocol flag,
               QC tests include Appearance, RNA identity (RT-PCR), Lipid Identity.
    """
    return [
        {
            "test_id": "T8-001",
            "query": "What is the lot number for this corrected COVID-19 vaccine protocol?",
            "required_terms": "FD7220|lot",
            "min_sources": 1,
            "classify": False,
            "criticality": "high",
        },
        {
            "test_id": "T8-002",
            "query": "Is this document a corrected protocol?",
            "required_terms": "corrected|correction|corrected protocol",
            "min_sources": 1,
            "classify": False,
            "criticality": "medium",
        },
        {
            "test_id": "T8-003",
            "query": "What is the trade name of the vaccine in this protocol?",
            "required_terms": "COMIRNATY|comirnaty",
            "min_sources": 1,
            "classify": False,
            "criticality": "high",
        },
        {
            "test_id": "T8-004",
            "query": "What is the date of manufacture for lot FD7220?",
            "required_terms": "23-Jun-2021|Jun-2021|June 2021|23 Jun|2021",
            "min_sources": 1,
            "classify": False,
            "criticality": "high",
        },
        {
            "test_id": "T8-005",
            "query": "What is the expiration date for this vaccine lot?",
            "required_terms": "30-Nov-2021|Nov-2021|November 2021|30 Nov|2021",
            "min_sources": 1,
            "classify": False,
            "criticality": "high",
        },
        {
            "test_id": "T8-006",
            "query": "What quality control tests were performed on the filled vaccine?",
            "required_terms": "appearance|RT-PCR|RNA|lipid|identity",
            "min_sources": 1,
            "classify": False,
            "criticality": "medium",
        },
    ]


def suite_test9() -> list[dict]:
    """Tests for test_9.pdf — BioNTech COVID-19 mRNA Vaccine Protocol (Lot FD7220).

    Document type: Electronic Protocol submission (For Licensing Action).
    Key facts: Lot FD7220, manufactured 23-Jun-2021, expiry 30-Nov-2021,
               STN 125742_0, License No. 2229, COMIRNATY trade name.
    """
    return [
        {
            "test_id": "T9-001",
            "query": "What is the lot number stated in this vaccine protocol?",
            "required_terms": "FD7220|lot",
            "min_sources": 1,
            "classify": False,
            "criticality": "high",
        },
        {
            "test_id": "T9-002",
            "query": "What is the reason for submission of this protocol?",
            "required_terms": "licensing action|release|surveillance",
            "min_sources": 1,
            "classify": False,
            "criticality": "medium",
        },
        {
            "test_id": "T9-003",
            "query": "What is the trade name of the product covered by this protocol?",
            "required_terms": "COMIRNATY|comirnaty",
            "min_sources": 1,
            "classify": False,
            "criticality": "high",
        },
        {
            "test_id": "T9-004",
            "query": "Who is the manufacturer or company named in this protocol?",
            "required_terms": "BioNTech|Pharmacia|Upjohn|Kalamazoo",
            "min_sources": 1,
            "classify": False,
            "criticality": "high",
        },
        {
            "test_id": "T9-005",
            "query": "What is the date of manufacture for this vaccine lot?",
            "required_terms": "23-Jun-2021|Jun-2021|June 2021|23 Jun|2021",
            "min_sources": 1,
            "classify": False,
            "criticality": "high",
        },
        {
            # Hallucination-resistance: batch protocol does not contain a pH value
            "test_id": "T9-006",
            "query": "What is the pH of the vaccine formulation in this batch protocol?",
            "required_terms": "not mentioned|not available|not provided|not specified|not stated|not found|no pH",
            "min_sources": 0,
            "classify": False,
            "criticality": "high",
        },
    ]


# ---------------------------------------------------------------------------
# Image (OCR) test suites — each folder is a scanned version of the
# corresponding numbered PDF (image_test_N ↔ test_N.pdf).
# ---------------------------------------------------------------------------

def suite_image_test1() -> list[dict]:
    """OCR tests for docs/image_test_1/ — scanned COVID-19 Vaccine SDS (image version)."""
    return [
        {
            "test_id": "I1-001",
            "query": "What is the product name and product code for the COVID-19 vaccine?",
            "required_terms": "pfizer|comirnaty|covid|PF00092",
            "min_sources": 1,
            "classify": False,
            "criticality": "high",
        },
        {
            "test_id": "I1-002",
            "query": "What are the storage conditions for the COVID-19 vaccine product?",
            "required_terms": "store|directed|packaging",
            "min_sources": 1,
            "classify": False,
            "criticality": "high",
        },
        {
            "test_id": "I1-003",
            "query": "What fire extinguishing media should be used for the COVID-19 vaccine?",
            "required_terms": "CO2|chemical|foam|water",
            "min_sources": 1,
            "classify": False,
            "criticality": "medium",
        },
        {
            "test_id": "I1-004",
            "query": "What is the chemical family or formulation type of the COVID-19 vaccine?",
            "required_terms": "lipid|nanoparticle|LNP|mRNA",
            "min_sources": 1,
            "classify": False,
            "criticality": "medium",
        },
    ]


def suite_image_test2() -> list[dict]:
    """OCR tests for docs/image_test_2/ — scanned Paracetamol Infusion SDS (image version)."""
    return [
        {
            "test_id": "I2-001",
            "query": "What is the product name and product code for the paracetamol infusion?",
            "required_terms": "paracetamol|perfalgan|PZ02462",
            "min_sources": 1,
            "classify": False,
            "criticality": "high",
        },
        {
            "test_id": "I2-002",
            "query": "What is the CAS number of the active ingredient in the paracetamol infusion?",
            "required_terms": "103-90-2",
            "min_sources": 1,
            "classify": False,
            "criticality": "medium",
        },
        {
            "test_id": "I2-003",
            "query": "What are the clinical effects of a paracetamol overdose?",
            "required_terms": "liver|hepatic|overdose|toxicity",
            "min_sources": 1,
            "classify": False,
            "criticality": "high",
        },
        {
            "test_id": "I2-004",
            "query": "What is the molecular formula of paracetamol?",
            "required_terms": "C8H9NO2",
            "min_sources": 1,
            "classify": False,
            "criticality": "medium",
        },
    ]


def suite_image_test3() -> list[dict]:
    """OCR tests for docs/image_test_3/ — scanned Zoledronic Acid SDS (image version)."""
    return [
        {
            "test_id": "I3-001",
            "query": "What is the product name and chemical family of the zoledronic acid product?",
            "required_terms": "zoledronic|bisphosphonate|PZ01101",
            "min_sources": 1,
            "classify": False,
            "criticality": "high",
        },
        {
            "test_id": "I3-002",
            "query": "What hazard classification applies to the zoledronic acid product?",
            "required_terms": "H360|reproductive|danger|toxic",
            "min_sources": 1,
            "classify": False,
            "criticality": "high",
        },
        {
            "test_id": "I3-003",
            "query": "What is the Pfizer occupational exposure limit (OEL) for zoledronic acid?",
            "required_terms": "4|OEL|exposure limit|µg/m3|ug/m3",
            "min_sources": 1,
            "classify": False,
            "criticality": "high",
        },
        {
            "test_id": "I3-004",
            "query": "What is the pH of the zoledronic acid injection solution?",
            "required_terms": "6.2|6",
            "min_sources": 1,
            "classify": False,
            "criticality": "medium",
        },
    ]


def suite_image_test4() -> list[dict]:
    """OCR tests for docs/image_test_4/ — scanned Ciprofloxacin Injection SDS (image version)."""
    return [
        {
            "test_id": "I4-001",
            "query": "What is the product name and chemical family of the ciprofloxacin product?",
            "required_terms": "ciprofloxacin|fluoroquinolone|PZ01031",
            "min_sources": 1,
            "classify": False,
            "criticality": "high",
        },
        {
            "test_id": "I4-002",
            "query": "What is the pH range of the ciprofloxacin injection solution?",
            "required_terms": "3.3|3.9|3",
            "min_sources": 1,
            "classify": False,
            "criticality": "medium",
        },
        {
            "test_id": "I4-003",
            "query": "What aquatic environmental hazards are associated with ciprofloxacin?",
            "required_terms": "H401|H411|aquatic|toxic to aquatic",
            "min_sources": 1,
            "classify": False,
            "criticality": "medium",
        },
        {
            "test_id": "I4-004",
            "query": "What are the known clinical effects of ciprofloxacin on tendons?",
            "required_terms": "tendon|tendonitis|rupture|musculoskeletal",
            "min_sources": 1,
            "classify": False,
            "criticality": "high",
        },
    ]


def suite_image_test5() -> list[dict]:
    """OCR tests for docs/image_test_5/ — scanned Cytiva AKTA ready Flow Kit docs (image version)."""
    return [
        {
            "test_id": "I5-001",
            "query": "What are the recommended storage and operating temperature conditions for the AKTA ready flow kit?",
            "required_terms": "+5|5°C|+2|+40|temperature",
            "min_sources": 1,
            "classify": False,
            "criticality": "high",
        },
        {
            "test_id": "I5-002",
            "query": "What are the lot numbers for the AKTA ready High Flow and Low Flow kits?",
            "required_terms": "18356721|15102934|lot",
            "min_sources": 1,
            "classify": False,
            "criticality": "high",
        },
        {
            "test_id": "I5-003",
            "query": "Does the AKTA ready flow kit contain any materials of animal origin?",
            "required_terms": "no animal|animal origin|BSE|TSE|not derived",
            "min_sources": 1,
            "classify": False,
            "criticality": "high",
        },
        {
            "test_id": "I5-004",
            "query": "What quality certifications does the supplier Cytiva hold?",
            "required_terms": "ISO 9001|ISO 13485|Cytiva|quality",
            "min_sources": 1,
            "classify": False,
            "criticality": "medium",
        },
    ]


def suite_test10() -> list[dict]:
    """Tests for test_10.pdf — FDA Response Letter: RNA Integrity / CGE Method (BLA 125742, 23 Jul 2021).

    Document type: BLA regulatory response letter.
    Key facts: BLA 125742, IND BB-IND 19736, dated 23 July 2021,
               addressee Marion Gruber PhD (FDA/CBER/OVRR),
               subject: validation of RNA integrity by capillary gel electrophoresis (CGE).
    """
    return [
        {
            "test_id": "T10-001",
            "query": "What BLA number is referenced in this FDA response letter?",
            "required_terms": "125742|BLA",
            "min_sources": 1,
            "classify": False,
            "criticality": "high",
        },
        {
            "test_id": "T10-002",
            "query": "What analytical method is the subject of this FDA information request?",
            "required_terms": "CGE|capillary gel electrophoresis|RNA integrity|RNA",
            "min_sources": 1,
            "classify": False,
            "criticality": "high",
        },
        {
            "test_id": "T10-003",
            "query": "What is the date of this FDA response letter?",
            "required_terms": "23 July 2021|July 2021|23-Jul|2021",
            "min_sources": 1,
            "classify": False,
            "criticality": "medium",
        },
        {
            "test_id": "T10-004",
            "query": "What is the IND number referenced in this regulatory submission?",
            "required_terms": "19736|BB-IND|IND",
            "min_sources": 1,
            "classify": False,
            "criticality": "medium",
        },
        {
            "test_id": "T10-005",
            "query": "Who is the FDA recipient named in this response letter?",
            "required_terms": "Marion Gruber|Gruber|CBER|OVRR|FDA",
            "min_sources": 1,
            "classify": False,
            "criticality": "medium",
        },
        {
            "test_id": "T10-006",
            "query": "What vaccine product does this BLA submission cover?",
            "required_terms": "BNT162|PF-07302048|COVID-19|COMIRNATY|mRNA",
            "min_sources": 1,
            "classify": False,
            "criticality": "high",
        },
    ]


def suite_test11() -> list[dict]:
    """Tests for test_11.pdf — FDA Response Letter: Sterility and Endotoxin Methods (BLA 125742, 30 Jul 2021).

    Document type: BLA regulatory response letter.
    Key facts: BLA 125742, IND BB-IND 19736, dated 30 July 2021,
               subject: sterility and endotoxin test methods,
               facilities: PGS-Puurs and PGS-KZO,
               FDA control number FDA-CBER-2021-5683-1149402.
    """
    return [
        {
            "test_id": "T11-001",
            "query": "What BLA number is referenced in this FDA response?",
            "required_terms": "125742|BLA",
            "min_sources": 1,
            "classify": False,
            "criticality": "high",
        },
        {
            "test_id": "T11-002",
            "query": "What test methods are discussed in this FDA information request response?",
            "required_terms": "sterility|endotoxin|sterile",
            "min_sources": 1,
            "classify": False,
            "criticality": "high",
        },
        {
            "test_id": "T11-003",
            "query": "What manufacturing facilities are referenced in this document?",
            "required_terms": "PGS-Puurs|PGS-KZO|Puurs|Kalamazoo",
            "min_sources": 1,
            "classify": False,
            "criticality": "high",
        },
        {
            "test_id": "T11-004",
            "query": "What is the date of this regulatory response letter?",
            "required_terms": "30 July 2021|July 2021|30-Jul|2021",
            "min_sources": 1,
            "classify": False,
            "criticality": "medium",
        },
        {
            "test_id": "T11-005",
            "query": "What is the FDA control number for this submission?",
            "required_terms": "FDA-CBER-2021-5683|1149402",
            "min_sources": 1,
            "classify": False,
            "criticality": "medium",
        },
        {
            "test_id": "T11-006",
            "query": "What vaccine product does this FDA response cover?",
            "required_terms": "BNT162|PF-07302048|COVID-19|mRNA",
            "min_sources": 1,
            "classify": False,
            "criticality": "high",
        },
    ]


def suite_test12() -> list[dict]:
    """Tests for test_12.pdf — Technical Response: Sterility and Endotoxin Verification (BLA 125742/0, 16 Jul 2021).

    Document type: Technical regulatory submission (FDA query response).
    Key facts: BLA 125742/0, dated 16 July 2021, approved 29 July 2021,
               LAL (endotoxin) and sterility method verification,
               PPC (positive product control) percent recoveries reported,
               facilities: PGS-Puurs, PGS-KZO.
    """
    return [
        {
            "test_id": "T12-001",
            "query": "What analytical methods are verified in this FDA technical response?",
            "required_terms": "sterility|endotoxin|LAL|PPC",
            "min_sources": 1,
            "classify": False,
            "criticality": "high",
        },
        {
            "test_id": "T12-002",
            "query": "What does PPC stand for in this endotoxin testing document?",
            "required_terms": "positive product control|PPC",
            "min_sources": 1,
            "classify": False,
            "criticality": "medium",
        },
        {
            "test_id": "T12-003",
            "query": "Which testing facilities are mentioned in this sterility and endotoxin verification?",
            "required_terms": "PGS-Puurs|PGS-KZO|Puurs|Kalamazoo",
            "min_sources": 1,
            "classify": False,
            "criticality": "high",
        },
        {
            "test_id": "T12-004",
            "query": "What is the BLA number associated with this technical document?",
            "required_terms": "125742|BLA",
            "min_sources": 1,
            "classify": False,
            "criticality": "high",
        },
        {
            "test_id": "T12-005",
            "query": "What method is used for endotoxin testing in this document?",
            "required_terms": "LAL|limulus|endotoxin",
            "min_sources": 1,
            "classify": False,
            "criticality": "high",
        },
        {
            "test_id": "T12-006",
            "query": "When was this technical document approved or submitted?",
            "required_terms": "2021|July 2021|29 July|29-Jul",
            "min_sources": 1,
            "classify": False,
            "criticality": "medium",
        },
    ]


def suite_test13() -> list[dict]:
    """Tests for test_13.pdf — FDA Response Letter: Manufacturing and Equipment (BLA 125742, 30 Jul 2021).

    Document type: BLA regulatory response letter.
    Key facts: BLA 125742, IND BB-IND 19736, dated 30 July 2021,
               subject: manufacturing and equipment queries,
               FDA contact: Laura Gottschalk PhD (CBER/OVRR),
               original BLA submission: 18 May 2021.
    """
    return [
        {
            "test_id": "T13-001",
            "query": "What is the subject of this FDA information request response?",
            "required_terms": "manufacturing|equipment",
            "min_sources": 1,
            "classify": False,
            "criticality": "high",
        },
        {
            "test_id": "T13-002",
            "query": "What BLA number is cited in this regulatory submission?",
            "required_terms": "125742|BLA",
            "min_sources": 1,
            "classify": False,
            "criticality": "high",
        },
        {
            "test_id": "T13-003",
            "query": "What is the date of this FDA manufacturing response letter?",
            "required_terms": "30 July 2021|July 2021|30-Jul|2021",
            "min_sources": 1,
            "classify": False,
            "criticality": "medium",
        },
        {
            "test_id": "T13-004",
            "query": "Who is the FDA contact person referenced in this letter?",
            "required_terms": "Laura Gottschalk|Gottschalk|CBER|OVRR",
            "min_sources": 1,
            "classify": False,
            "criticality": "medium",
        },
        {
            "test_id": "T13-005",
            "query": "Who is the applicant or sender named in this BLA response?",
            "required_terms": "BioNTech|Pfizer|Pharmacia|Upjohn",
            "min_sources": 1,
            "classify": False,
            "criticality": "high",
        },
        {
            "test_id": "T13-006",
            "query": "What is the original BLA submission date mentioned in this letter?",
            "required_terms": "18 May 2021|May 2021|18-May|2021",
            "min_sources": 1,
            "classify": False,
            "criticality": "medium",
        },
    ]


def suite_test14() -> list[dict]:
    """Tests for test_14.pdf — Technical Response: Manufacturing and Equipment (BLA 125742/0, 26 Jul 2021).

    Document type: Technical regulatory submission (FDA query response).
    Key facts: BLA 125742/0, dated 26 July 2021, approved 30 July 2021,
               16 manufacturing/equipment queries addressed,
               bioburden and endotoxin action limits, hold times,
               validated shipping with temperature monitoring,
               facilities: PGS-Puurs, PGS-KZO.
    """
    return [
        {
            "test_id": "T14-001",
            "query": "What testing parameters are addressed in this manufacturing technical response?",
            "required_terms": "bioburden|endotoxin|hold time|sterility",
            "min_sources": 1,
            "classify": False,
            "criticality": "high",
        },
        {
            "test_id": "T14-002",
            "query": "How many FDA queries are addressed in this technical response document?",
            "required_terms": "16|sixteen",
            "min_sources": 1,
            "classify": False,
            "criticality": "medium",
        },
        {
            "test_id": "T14-003",
            "query": "What in-process controls are described in this manufacturing document?",
            "required_terms": "bioburden|endotoxin|action limit|sampling",
            "min_sources": 1,
            "classify": False,
            "criticality": "high",
        },
        {
            "test_id": "T14-004",
            "query": "What validated process is described for product shipping?",
            "required_terms": "shipping|temperature|validated|cold chain",
            "min_sources": 1,
            "classify": False,
            "criticality": "high",
        },
        {
            "test_id": "T14-005",
            "query": "What BLA number does this manufacturing technical document relate to?",
            "required_terms": "125742|BLA",
            "min_sources": 1,
            "classify": False,
            "criticality": "high",
        },
        {
            "test_id": "T14-006",
            "query": "When was this manufacturing technical document approved?",
            "required_terms": "30 July 2021|July 2021|30-Jul|2021",
            "min_sources": 1,
            "classify": False,
            "criticality": "medium",
        },
    ]


def build_combined_suite() -> list[dict]:
    """Combine all PDF per-document suites into a single regression suite."""
    return (
        suite_test1()
        + suite_test2()
        + suite_test3()
        + suite_test4()
        + suite_test5()
        + suite_test7()
        + suite_test8()
        + suite_test9()
        + suite_test10()
        + suite_test11()
        + suite_test12()
        + suite_test13()
        + suite_test14()
    )


def build_pdf_suite_map() -> dict[str, list[dict]]:
    """Return per-file PDF regression suites keyed by filename."""
    return {
        "test_1.pdf": suite_test1(),
        "test_2.pdf": suite_test2(),
        "test_3.pdf": suite_test3(),
        "test_4.pdf": suite_test4(),
        "test_5.pdf": suite_test5(),
        "test_7.pdf": suite_test7(),
        "test_8.pdf": suite_test8(),
        "test_9.pdf": suite_test9(),
        "test_10.pdf": suite_test10(),
        "test_11.pdf": suite_test11(),
        "test_12.pdf": suite_test12(),
        "test_13.pdf": suite_test13(),
        "test_14.pdf": suite_test14(),
    }


def build_image_suite() -> list[dict]:
    """Combine all image (OCR) per-folder suites into a single regression suite."""
    return (
        suite_image_test1()
        + suite_image_test2()
        + suite_image_test3()
        + suite_image_test4()
        + suite_image_test5()
    )


def build_image_suite_map() -> dict[str, list[dict]]:
    """Return per-folder OCR regression suites keyed by folder name."""
    return {
        "image_test_1": suite_image_test1(),
        "image_test_2": suite_image_test2(),
        "image_test_3": suite_image_test3(),
        "image_test_4": suite_image_test4(),
        "image_test_5": suite_image_test5(),
    }


# ---------------------------------------------------------------------------
# Corpus growth plans
# ---------------------------------------------------------------------------
#
# A "corpus plan" is an ordered list of dicts, one per document.  Documents
# are processed strictly in list order; each entry defines the document to
# add at that step and the callable that returns its test cases.
#
# PDF_CORPUS_PLAN  — 13 digital PDFs indexed one by one.
# IMAGE_CORPUS_PLAN — 5 scanned image folders indexed one by one.
#
# The order inside each plan matters: it determines which milestone index a
# document receives, and therefore where it appears on the x-axis of the
# growth-trend charts.  The order here matches build_combined_suite() and
# build_image_suite() so that the test_ids are stable across all outputs.
# ---------------------------------------------------------------------------

PDF_CORPUS_PLAN: list[dict] = [
    # Each entry: the document added at this step + its suite callable.
    # "suite_fn" is called (not pre-called) so that each run gets a fresh
    # copy of the test case list with no shared mutable state.
    {"doc_name": "test_1.pdf",  "suite_fn": suite_test1},
    {"doc_name": "test_2.pdf",  "suite_fn": suite_test2},
    {"doc_name": "test_3.pdf",  "suite_fn": suite_test3},
    {"doc_name": "test_4.pdf",  "suite_fn": suite_test4},
    {"doc_name": "test_5.pdf",  "suite_fn": suite_test5},
    {"doc_name": "test_7.pdf",  "suite_fn": suite_test7},
    {"doc_name": "test_8.pdf",  "suite_fn": suite_test8},
    {"doc_name": "test_9.pdf",  "suite_fn": suite_test9},
    {"doc_name": "test_10.pdf", "suite_fn": suite_test10},
    {"doc_name": "test_11.pdf", "suite_fn": suite_test11},
    {"doc_name": "test_12.pdf", "suite_fn": suite_test12},
    {"doc_name": "test_13.pdf", "suite_fn": suite_test13},
    {"doc_name": "test_14.pdf", "suite_fn": suite_test14},
]

IMAGE_CORPUS_PLAN: list[dict] = [
    {"doc_name": "image_test_1", "suite_fn": suite_image_test1},
    {"doc_name": "image_test_2", "suite_fn": suite_image_test2},
    {"doc_name": "image_test_3", "suite_fn": suite_image_test3},
    {"doc_name": "image_test_4", "suite_fn": suite_image_test4},
    {"doc_name": "image_test_5", "suite_fn": suite_image_test5},
]


# ---------------------------------------------------------------------------
# Cumulative regression runner
# ---------------------------------------------------------------------------

def run_cumulative_regression(
    plan: list[dict],
    docs_dir: Path,
    pipeline_kwargs: dict,
    phase: str,
    global_milestone_offset: int,
    index_root: "Path | None",
    latency_profile: str = "full",
    retrieval_modes: "list[str] | None" = None,
) -> object:
    """Execute one full cumulative corpus-growth regression phase.

    This function is the core orchestration loop.  It processes the entries in
    ``plan`` one at a time, growing the corpus by one document per iteration.
    At each step it:

      1. **Appends** the new document path to the cumulative list.
      2. **Rebuilds** the full index from *all* accumulated documents so far.
         This simulates how a real deployment accumulates data over time: the
         pipeline sees the same growing context that a production system would.
      3. **Assembles** the test suite for *all* documents currently in the index
         (not just the newly-added one).  Running tests for every indexed document
         at every milestone is what separates proper regression from a simple
         smoke test: you can detect when adding document N causes retrieval for
         document M to degrade — cross-contamination.
      4. **Executes** the suite via ``harness.run_at_milestone()``, which records
         per-test metrics and stamps each row with the corpus state at that moment.
      5. **Prints** a live progress summary so long runs produce visible output.

    The function returns the concatenated milestone DataFrame covering all steps.
    Call ``RAGRegressionHarness.summarize_milestones()`` on it for aggregates, and
    ``RAGRegressionHarness.visualize_growth_trends()`` for the 6-panel chart.

    Parameters
    ----------
    plan:
        Ordered list of ``{"doc_name": str, "suite_fn": callable}`` dicts.
        Documents are processed in this order; each entry adds one document.
    docs_dir:
        Root directory that contains the documents referenced in ``plan``.
    pipeline_kwargs:
        Keyword arguments forwarded verbatim to ``RAGPipeline()``.  Pass
        ``{"model_path": "..."}`` when using a non-default LLM.
    phase:
        ``"pdf"`` or ``"images"`` — controls which pipeline build method is used
        and which column value appears in the ``phase`` column of the output.
    global_milestone_offset:
        Added to the 1-based step index to produce a globally unique
        ``milestone_index`` across both phases.  Pass 0 for the first phase
        and ``len(PDF_CORPUS_PLAN)`` for the second phase so that the combined
        DataFrame has a contiguous, non-repeating milestone sequence.
    index_root:
        Optional parent directory for the temporary persisted index.  When
        ``None`` the OS default temp directory is used.

    Returns
    -------
    pd.DataFrame
        One row per (test_case × milestone).  Columns include all standard
        ``run()`` columns plus the milestone metadata columns added by
        ``run_at_milestone``.
    """
    import pandas as pd  # local import: keep top-level imports minimal

    # Lists that grow across iterations; each loop pass adds exactly one entry.
    cumulative_paths: list[str] = []   # absolute paths of all indexed documents so far
    all_applicable_tests: list[dict] = []  # test cases for all indexed documents so far
    all_milestone_results: list[pd.DataFrame] = []  # one DataFrame per milestone

    # Create a single temporary index directory that persists across the loop so
    # that each rebuild writes over the previous index rather than creating a new
    # temp folder, keeping disk usage bounded.
    with tempfile.TemporaryDirectory(
        dir=str(index_root) if index_root else None
    ) as temp_index_root:
        index_dir_path = Path(temp_index_root) / f"{phase}_cumulative"

        for step_index, entry in enumerate(plan, start=1):
            # ----------------------------------------------------------------
            # Step metadata
            # ----------------------------------------------------------------
            doc_name = entry["doc_name"]
            # milestone_index is globally unique across both phases so that the
            # PDF and image DataFrames can be concatenated without collision.
            milestone_index = global_milestone_offset + step_index
            doc_path = str(docs_dir / doc_name)

            # Resolve the test cases for the document being added at this step.
            # calling suite_fn() each time ensures no shared list references.
            new_tests: list[dict] = entry["suite_fn"]()
            # The set of test_ids for *this* document only.  Passed to
            # run_at_milestone so it can mark is_new_test correctly.
            new_test_ids: set[str] = {t["test_id"] for t in new_tests}

            # ----------------------------------------------------------------
            # 1. Expand cumulative corpus
            # ----------------------------------------------------------------
            # Appending the path before the build call ensures the index always
            # contains the current document and all prior ones.
            cumulative_paths.append(doc_path)

            # ----------------------------------------------------------------
            # 2. Rebuild index with all accumulated documents
            # ----------------------------------------------------------------
            # A fresh RAGPipeline is created per milestone so there is no
            # stale state from the previous build (caches, cached embeddings,
            # or in-memory structures that don't get flushed on rebuild).
            rag = RAGPipeline(**{**pipeline_kwargs, "persist_dir": str(index_dir_path)})

            print(
                f"\n[Milestone {milestone_index:2d}] Adding '{doc_name}' "
                f"(corpus = {len(cumulative_paths)} {phase} doc(s))"
            )

            try:
                if phase == "pdf":
                    # build_from_multiple_pdfs: parses text from digital PDFs,
                    # falls back to Tesseract OCR for scanned pages, chunks with
                    # SentenceSplitter, embeds with HuggingFace, builds FAISS+BM25.
                    rag.build_from_multiple_pdfs(
                        cumulative_paths,
                        classify_docs=True,
                    )
                else:
                    # build_from_multiple_image_folders: runs Tesseract at 200 DPI
                    # on every PNG in each folder, then the same indexing pipeline.
                    rag.build_from_multiple_image_folders(
                        cumulative_paths,
                        classify_docs=True,
                    )
            except RuntimeError as exc:
                # Tesseract not installed, GPU OOM, corrupt PDF, etc.
                print(f"  ERROR building index at milestone {milestone_index}: {exc}", file=sys.stderr)
                print("  Stopping phase early — subsequent milestones skipped.", file=sys.stderr)
                break

            # ----------------------------------------------------------------
            # 3. Expand the cumulative test suite
            # ----------------------------------------------------------------
            # all_applicable_tests grows by exactly one document's worth of tests
            # per iteration.  By milestone N it covers all N documents' tests,
            # which is exactly the right scope: the retriever should be able to
            # answer questions about any document currently in the index.
            all_applicable_tests.extend(new_tests)

            # Build the test list respecting the requested latency profile and
            # retrieval modes.  When retrieval_modes has more than one entry every
            # test is replicated once per mode so the results CSV contains a
            # side-by-side comparison across modes.
            _do_classify = latency_profile in ("retrieval_plus_classify", "full")
            _do_expand   = latency_profile in ("retrieval_plus_expand",   "full")
            _modes        = retrieval_modes if retrieval_modes else ["hybrid"]
            full_flow_tests = [
                {
                    **t,
                    "classify":       _do_classify,
                    "expand":         _do_expand,
                    "retrieval_mode": mode,
                    "num_expansions": max(int(t.get("num_expansions", 3)), 3),
                }
                for mode in _modes
                for t in all_applicable_tests
            ]
            suite_df = RAGRegressionHarness.create_test_suite(full_flow_tests)

            # ----------------------------------------------------------------
            # 4. Run all applicable tests and stamp milestone metadata
            # ----------------------------------------------------------------
            harness = RAGRegressionHarness(rag)
            milestone_results = harness.run_at_milestone(
                suite_df,
                milestone_index=milestone_index,
                doc_count=len(cumulative_paths),
                doc_just_added=doc_name,
                new_test_ids=new_test_ids,
            )

            # ----------------------------------------------------------------
            # 5. Live progress summary
            # ----------------------------------------------------------------
            # Print a one-line summary immediately so long regressions (which
            # can run for hours on a real model) show continuous output.
            n_pass = int(milestone_results["passed"].sum())
            n_total = len(milestone_results)
            n_new = int(milestone_results["is_new_test"].sum())
            n_existing = n_total - n_new
            existing_pass = int(
                milestone_results.loc[~milestone_results["is_new_test"], "passed"].sum()
            ) if n_existing > 0 else 0

            print(
                f"  Tests: {n_total} total "
                f"({n_new} new, {n_existing} existing)  "
                f"Pass: {n_pass}/{n_total}"
                + (f"  Existing pass: {existing_pass}/{n_existing}" if n_existing > 0 else "")
            )

            # Collect for concatenation after the loop.
            all_milestone_results.append(milestone_results)

    import pandas as pd

    if not all_milestone_results:
        # Return an empty DataFrame so the caller can handle an empty result
        # without special-casing a None return value.
        return pd.DataFrame()

    combined = pd.concat(all_milestone_results, ignore_index=True)
    # Tag every row with the phase label so that PDF and image results can be
    # distinguished after they are combined into a single artifacts CSV.
    combined["phase"] = phase
    return combined


def main() -> int:
    """Orchestrate the two-phase cumulative corpus-growth regression.

    Phase 1 — PDF: indexes test_1.pdf through test_14.pdf one by one.
    Phase 2 — Images: indexes image_test_1 through image_test_5 one by one.

    At each step ALL tests for documents currently in the index are re-run,
    not just the tests for the newly-added document.  This is what makes the
    regression "cumulative": you can observe how answer quality, retrieval
    confidence, and latency evolve as the corpus grows, and detect cross-
    contamination (a new document degrading retrieval for an older one).

    Output artifacts
    ----------------
    artifacts/regression_results.csv
        Raw per-test-per-milestone results for both phases combined.
        One row per (test_case, milestone_index).  Suitable for custom
        analysis in pandas, Excel, or any BI tool.

    artifacts/regression_milestones_pdf.csv
    artifacts/regression_milestones_images.csv
        Per-milestone aggregate summary (pass rate, confidence, latency)
        for each phase.  One row per milestone — ideal for quick review.

    artifacts/regression_growth_pdf.png
    artifacts/regression_growth_images.png
        Six-panel growth-trend charts for each phase.

    artifacts/regression_dashboard_pdf.png
    artifacts/regression_dashboard_images.png
        Single-run summary dashboard for the final (all-docs) milestone.

    artifacts/baseline_comparison.csv  (optional)
    artifacts/baseline_comparison_dashboard.png  (optional)
        Diff against a previously-saved baseline CSV.
    """
    args = parse_args()

    artifacts_dir = Path(args.artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    docs_dir = Path(args.docs_dir)
    pipeline_kwargs: dict = {}
    if args.model_path:
        pipeline_kwargs["model_path"] = args.model_path

    index_root = Path(args.index_dir) if args.index_dir else None

    exit_code = 0

    # ------------------------------------------------------------------ #
    # Phase 1: PDF corpus growth                                          #
    # ------------------------------------------------------------------ #
    # Verify that at least one expected PDF exists before starting.
    first_pdf = docs_dir / "test_1.pdf"
    if not first_pdf.exists():
        print(f"ERROR: expected PDF not found: {first_pdf}", file=sys.stderr)
        return 1

    print(f"=== Phase 1: PDF regression ({len(PDF_CORPUS_PLAN)} documents) ===")

    _retrieval_modes = (
        ["bm25", "vector", "hybrid"] if args.retrieval_mode == "all" else [args.retrieval_mode]
    )
    pdf_milestone_df = run_cumulative_regression(
        plan=PDF_CORPUS_PLAN,
        docs_dir=docs_dir,
        pipeline_kwargs=pipeline_kwargs,
        phase="pdf",
        # Milestone indices for Phase 1 start at 1 (offset = 0).
        global_milestone_offset=0,
        index_root=index_root,
        latency_profile=args.latency_profile,
        retrieval_modes=_retrieval_modes,
    )

    if not pdf_milestone_df.empty:
        import pandas as pd

        # Save the raw per-test-per-milestone results.
        results_csv = artifacts_dir / "regression_results.csv"
        RAGRegressionHarness.save_results(pdf_milestone_df, str(results_csv))
        print(f"\nSaved PDF milestone results  : {results_csv}")

        # Aggregate to one row per milestone for easy review.
        pdf_summary_df = RAGRegressionHarness.summarize_milestones(pdf_milestone_df)
        pdf_milestones_csv = artifacts_dir / "regression_milestones_pdf.csv"
        RAGRegressionHarness.save_results(pdf_summary_df, str(pdf_milestones_csv))
        print(f"Saved PDF milestone summary   : {pdf_milestones_csv}")

        # Growth-trend chart: 6 panels covering pass rate, confidence, latency,
        # heatmap, latency distribution, and source-count trend.
        growth_png = artifacts_dir / "regression_growth_pdf.png"
        RAGRegressionHarness.visualize_growth_trends(
            pdf_milestone_df,
            str(growth_png),
            title=f"PDF Corpus Growth Regression — {len(PDF_CORPUS_PLAN)} documents",
        )
        print(f"Saved PDF growth-trend chart  : {growth_png}")

        # Single-run dashboard for the final milestone (all PDFs indexed).
        final_milestone = int(pdf_milestone_df["milestone_index"].max())
        final_pdf_results = pdf_milestone_df[
            pdf_milestone_df["milestone_index"] == final_milestone
        ].copy()
        dashboard_png = artifacts_dir / "regression_dashboard_pdf.png"
        RAGRegressionHarness.visualize_results(
            final_pdf_results,
            str(dashboard_png),
            title=f"PDF Suite Dashboard — milestone {final_milestone} ({len(final_pdf_results)} tests)",
        )
        print(f"Saved PDF final dashboard     : {dashboard_png}")

        # Headline summary for the terminal.
        pdf_summary = RAGRegressionHarness.summarize(final_pdf_results)
        print("\nPDF phase summary (final milestone):")
        for key, value in pdf_summary.items():
            print(f"  {key}: {value}")
        if pdf_summary.get("failed_tests", 0) > 0:
            print(
                f"ERROR: {pdf_summary['failed_tests']} PDF test(s) failed at final milestone.",
                file=sys.stderr,
            )
            exit_code = 1
    else:
        print("WARNING: PDF phase produced no results.", file=sys.stderr)

    # ------------------------------------------------------------------ #
    # Phase 2: OCR / image-folder corpus growth                           #
    # ------------------------------------------------------------------ #
    # Image folders use a fresh pipeline instance (separate FAISS index)
    # because the pipeline does not support mixing PDF and image sources in
    # one build call.  Milestone indices continue from where Phase 1 left off
    # so the combined CSV has a contiguous, non-repeating sequence.
    first_image_folder = docs_dir / "image_test_1"
    if not first_image_folder.exists():
        print(f"\nNo image_test_1 folder found in {docs_dir.resolve()} — skipping OCR phase.")
    else:
        print(f"\n=== Phase 2: Image/OCR regression ({len(IMAGE_CORPUS_PLAN)} folders) ===")

        image_milestone_df = run_cumulative_regression(
            plan=IMAGE_CORPUS_PLAN,
            docs_dir=docs_dir,
            pipeline_kwargs=pipeline_kwargs,
            phase="images",
            # Phase 2 milestones continue numbering after Phase 1 ends.
            global_milestone_offset=len(PDF_CORPUS_PLAN),
            index_root=index_root,
            latency_profile=args.latency_profile,
            retrieval_modes=_retrieval_modes,
        )

        if not image_milestone_df.empty:
            import pandas as pd

            image_csv = artifacts_dir / "image_regression_results.csv"
            RAGRegressionHarness.save_results(image_milestone_df, str(image_csv))
            print(f"\nSaved image milestone results : {image_csv}")

            img_summary_df = RAGRegressionHarness.summarize_milestones(image_milestone_df)
            img_milestones_csv = artifacts_dir / "regression_milestones_images.csv"
            RAGRegressionHarness.save_results(img_summary_df, str(img_milestones_csv))
            print(f"Saved image milestone summary : {img_milestones_csv}")

            img_growth_png = artifacts_dir / "regression_growth_images.png"
            RAGRegressionHarness.visualize_growth_trends(
                image_milestone_df,
                str(img_growth_png),
                title=f"Image/OCR Corpus Growth Regression — {len(IMAGE_CORPUS_PLAN)} folders",
            )
            print(f"Saved image growth-trend chart: {img_growth_png}")

            final_img_milestone = int(image_milestone_df["milestone_index"].max())
            final_img_results = image_milestone_df[
                image_milestone_df["milestone_index"] == final_img_milestone
            ].copy()
            img_dashboard_png = artifacts_dir / "regression_dashboard_images.png"
            RAGRegressionHarness.visualize_results(
                final_img_results,
                str(img_dashboard_png),
                title=f"Image Suite Dashboard — milestone {final_img_milestone} ({len(final_img_results)} tests)",
            )
            print(f"Saved image final dashboard   : {img_dashboard_png}")

            img_summary = RAGRegressionHarness.summarize(final_img_results)
            print("\nImage phase summary (final milestone):")
            for key, value in img_summary.items():
                print(f"  {key}: {value}")
            if img_summary.get("failed_tests", 0) > 0:
                print(
                    f"ERROR: {img_summary['failed_tests']} OCR test(s) failed at final milestone.",
                    file=sys.stderr,
                )
                exit_code = 1

    # ------------------------------------------------------------------ #
    # Optional: baseline comparison against a previously-saved CSV        #
    # ------------------------------------------------------------------ #
    # Compares the final-milestone PDF results against a baseline snapshot
    # to detect whether code changes (not corpus growth) caused regressions.
    # Pass --baseline artifacts/baseline_results.csv to enable this step.
    if args.baseline and Path(args.baseline).exists() and not pdf_milestone_df.empty:
        import pandas as pd

        baseline_df = RAGRegressionHarness.load_results(args.baseline)
        # Use the final-milestone PDF results for the comparison so both sides
        # reflect the same corpus state (all PDFs indexed).
        final_milestone = int(pdf_milestone_df["milestone_index"].max())
        final_pdf_results = pdf_milestone_df[
            pdf_milestone_df["milestone_index"] == final_milestone
        ].copy()

        # Create a temporary harness just for the comparison (no pipeline needed).
        _tmp_harness = RAGRegressionHarness(rag_pipeline=None)  # type: ignore[arg-type]
        comparison_df = _tmp_harness.compare_to_baseline(final_pdf_results, baseline_df)

        comparison_csv = artifacts_dir / "baseline_comparison.csv"
        comparison_png = artifacts_dir / "baseline_comparison_dashboard.png"
        comparison_df.to_csv(comparison_csv, index=False)
        RAGRegressionHarness.visualize_comparison(comparison_df, str(comparison_png))

        regressions = int(comparison_df["regression_detected"].sum())
        print(f"\nSaved baseline comparison CSV : {comparison_csv}")
        print(f"Saved baseline comparison PNG : {comparison_png}")
        if regressions > 0:
            print(
                f"ERROR: {regressions} regression(s) detected vs baseline.",
                file=sys.stderr,
            )
            exit_code = 1

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

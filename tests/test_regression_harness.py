"""Tests for pandas-based regression harness.

The FakeRAG stub is seeded with deterministic responses for all documents in
docs/ so that `test_run_generates_pass_fail_metrics` validates the complete
multi-document test suite (PDF and image/OCR) without hitting the real pipeline.

FakeRAG mirrors the RAGPipeline public interface:
  - build(pdf_path, classify_docs=False)
  - build_from_multiple_pdfs(pdf_paths, classify_docs=False)
  - build_from_images(folder_path, classify_docs=False)
  - query_with_sources(question, ...)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import pandas as pd
import pytest

from rag.regression import RAGRegressionHarness

# All tests in this file exercise the harness mechanics via a deterministic
# FakeRAG stub.  They are marked ``unit`` to distinguish them from
# ``integration`` tests that require a real model and indexed documents.
pytestmark = pytest.mark.unit


@dataclass
class FakeRAG:
    """Deterministic stub that mirrors the RAGPipeline public interface.

    ``responses`` maps query text → the dict returned by query_with_sources.
    ``build``, ``build_from_multiple_pdfs``, and ``build_from_images`` are
    no-ops so that tests can exercise the full harness flow without a real
    model or index.
    """

    responses: Dict[str, Dict[str, Any]]
    _built: bool = field(default=False, init=False, repr=False)

    # ------------------------------------------------------------------
    # Pipeline build methods (no-ops in the fake)
    # ------------------------------------------------------------------

    def build(self, pdf_path: str, classify_docs: bool = False) -> None:  # noqa: ARG002
        self._built = True

    def build_from_multiple_pdfs(
        self,
        pdf_paths: List[str],
        classify_docs: bool = False,
        progress_callback: Any = None,
    ) -> None:  # noqa: ARG002
        self._built = True

    def build_from_images(self, folder_path: str, classify_docs: bool = False) -> None:  # noqa: ARG002
        self._built = True

    # ------------------------------------------------------------------
    # Query method
    # ------------------------------------------------------------------

    def query_with_sources(
        self,
        question: str,
        classify: bool = False,
        expand: bool = False,
        num_expansions: int = 3,
    ) -> Dict[str, Any]:
        _ = classify, expand, num_expansions
        return self.responses[question]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _src(file: str, page: int, score: float = 90.0) -> Dict[str, Any]:
    return {"score": score, "file": file, "page": page}


# ---------------------------------------------------------------------------
# Shared FakeRAG responses for the full multi-document suite
# ---------------------------------------------------------------------------

MULTI_DOC_RESPONSES: Dict[str, Dict[str, Any]] = {
    # ---- test_1.pdf  (COVID-19 Vaccine SDS) --------------------------------
    "What is the product name and product code for the COVID-19 vaccine?": {
        "answer": "The product name is Pfizer-BioNTech COVID-19 Vaccine (Comirnaty) with product code PF00092.",
        "query_category": None,
        "sources": [_src("test_1.pdf", 1, 95.0)],
    },
    "What are the storage conditions for the COVID-19 vaccine product?": {
        "answer": "Store as directed by product packaging.",
        "query_category": None,
        "sources": [_src("test_1.pdf", 5, 88.0)],
    },
    "What fire extinguishing media should be used for the COVID-19 vaccine?": {
        "answer": "Use dry chemical, CO2, alcohol-resistant foam or water spray.",
        "query_category": None,
        "sources": [_src("test_1.pdf", 4, 85.0)],
    },
    "Is the COVID-19 vaccine regulated for transport under DOT or IATA?": {
        "answer": "The product is not regulated for transport under USDOT, EUADR, IATA, or IMDG regulations.",
        "query_category": None,
        "sources": [_src("test_1.pdf", 11, 90.0)],
    },
    "What is the chemical family or formulation type of the COVID-19 vaccine?": {
        "answer": "The COVID-19 vaccine uses a lipid nanoparticle (LNP) formulation to deliver mRNA.",
        "query_category": None,
        "sources": [_src("test_1.pdf", 3, 88.0)],
    },
    "What is the batch number or lot number of the COVID-19 vaccine?": {
        # Hallucination-resistance: SDS contains no batch/lot number.
        "answer": "There is no batch number or lot number mentioned in this Safety Data Sheet.",
        "query_category": None,
        "sources": [],
    },

    # ---- test_2.pdf  (Paracetamol Infusion SDS) ----------------------------
    "What is the product name and product code for the paracetamol infusion?": {
        "answer": "The product name is Paracetamol Solution for Infusion (Perfalgan) with product code PZ02462.",
        "query_category": None,
        "sources": [_src("test_2.pdf", 1, 93.0)],
    },
    "What is the intended use or therapeutic category of the paracetamol product?": {
        "answer": "Paracetamol is used as an analgesic and antipyretic for treatment of pain and fever.",
        "query_category": None,
        "sources": [_src("test_2.pdf", 1, 90.0)],
    },
    "What is the CAS number of the active ingredient in the paracetamol infusion?": {
        "answer": "The CAS number of paracetamol (acetaminophen) is 103-90-2.",
        "query_category": None,
        "sources": [_src("test_2.pdf", 3, 88.0)],
    },
    "What is the molecular formula of paracetamol?": {
        "answer": "The molecular formula of paracetamol is C8H9NO2 with a molecular weight of 151.2.",
        "query_category": None,
        "sources": [_src("test_2.pdf", 9, 87.0)],
    },
    "What are the clinical effects of a paracetamol overdose?": {
        "answer": "Overdose can cause severe hepatic (liver) toxicity and liver failure.",
        "query_category": None,
        "sources": [_src("test_2.pdf", 11, 91.0)],
    },
    "What is the batch number or lot number of the paracetamol infusion?": {
        # Hallucination-resistance: SDS contains no batch/lot number.
        "answer": "The Safety Data Sheet does not provide a batch number or lot number.",
        "query_category": None,
        "sources": [],
    },

    # ---- test_3.pdf  (Zoledronic Acid SDS) ---------------------------------
    "What is the product name and chemical family of the zoledronic acid product?": {
        "answer": "The product is Zoledronic Acid Injection (PZ01101), a bisphosphonate compound.",
        "query_category": None,
        "sources": [_src("test_3.pdf", 1, 94.0)],
    },
    "What hazard classification applies to the zoledronic acid product?": {
        "answer": "Zoledronic acid is classified as reproductive toxic (H360FD — Danger).",
        "query_category": None,
        "sources": [_src("test_3.pdf", 2, 92.0)],
    },
    "What is the GHS signal word for zoledronic acid?": {
        "answer": "The GHS signal word for zoledronic acid is Danger.",
        "query_category": None,
        "sources": [_src("test_3.pdf", 2, 90.0)],
    },
    "What is the pH of the zoledronic acid injection solution?": {
        "answer": "The pH of the zoledronic acid injection solution is 6.2.",
        "query_category": None,
        "sources": [_src("test_3.pdf", 9, 88.0)],
    },
    "What is the Pfizer occupational exposure limit (OEL) for zoledronic acid?": {
        "answer": "The Pfizer OEL for zoledronic acid is 4 µg/m3 as an inhalable dust.",
        "query_category": None,
        "sources": [_src("test_3.pdf", 8, 89.0)],
    },
    "Is zoledronic acid regulated for transport?": {
        "answer": "Zoledronic acid is not regulated for transport under DOT, IATA, or IMDG.",
        "query_category": None,
        "sources": [_src("test_3.pdf", 14, 87.0)],
    },

    # ---- test_4.pdf  (Ciprofloxacin Injection SDS) -------------------------
    "What is the product name and chemical family of the ciprofloxacin product?": {
        "answer": "The product is Ciprofloxacin Injection (PZ01031), a fluoroquinolone antibiotic.",
        "query_category": None,
        "sources": [_src("test_4.pdf", 1, 93.0)],
    },
    "What is the recommended use or therapeutic category of ciprofloxacin injection?": {
        "answer": "Ciprofloxacin is a broad-spectrum antibiotic used to treat bacterial infections.",
        "query_category": None,
        "sources": [_src("test_4.pdf", 1, 91.0)],
    },
    "What is the pH range of the ciprofloxacin injection solution?": {
        "answer": "The pH of the ciprofloxacin injection solution ranges from 3.3 to 3.9.",
        "query_category": None,
        "sources": [_src("test_4.pdf", 9, 88.0)],
    },
    "What is the Pfizer occupational exposure limit for ciprofloxacin?": {
        "answer": "The Pfizer OEL for ciprofloxacin is 600 µg/m3.",
        "query_category": None,
        "sources": [_src("test_4.pdf", 8, 90.0)],
    },
    "What aquatic environmental hazards are associated with ciprofloxacin?": {
        "answer": "Ciprofloxacin is toxic to aquatic life (H401) and with long lasting effects (H411).",
        "query_category": None,
        "sources": [_src("test_4.pdf", 12, 87.0)],
    },
    "What are the known clinical effects of ciprofloxacin on tendons?": {
        "answer": "Ciprofloxacin may cause tendonitis and tendon rupture, particularly the Achilles tendon.",
        "query_category": None,
        "sources": [_src("test_4.pdf", 11, 89.0)],
    },

    # ---- test_5.pdf  (Cytiva AKTA ready Flow Kit documents) ----------------
    "What are the recommended storage and operating temperature conditions for the AKTA ready flow kit?": {
        "answer": "Store at temperatures greater than +5°C. The kit can be used at temperatures between +2°C and +40°C.",
        "query_category": None,
        "sources": [_src("test_5.pdf", 1, 92.0)],
    },
    "What are the lot numbers for the AKTA ready High Flow and Low Flow kits?": {
        "answer": "The lot number for the High Flow kit is 18356721 and for the Low Flow kit is 15102934.",
        "query_category": None,
        "sources": [_src("test_5.pdf", 1, 94.0)],
    },
    "Does the AKTA ready flow kit contain any materials of animal origin?": {
        "answer": "No, the AKTA ready flow kit does not contain materials of animal origin. The BSE/TSE declaration confirms this.",
        "query_category": None,
        "sources": [_src("test_5.pdf", 2, 90.0)],
    },
    "What change was made to the blister packaging material for the AKTA ready kit?": {
        "answer": "The blister packaging material was changed from PVC to PETG as documented in the packaging spec.",
        "query_category": None,
        "sources": [_src("test_5.pdf", 3, 88.0)],
    },
    "What quality certifications does the supplier Cytiva hold?": {
        "answer": "Cytiva Sweden AB holds ISO 9001 and ISO 13485 quality management system certifications.",
        "query_category": None,
        "sources": [_src("test_5.pdf", 4, 89.0)],
    },
    "From which location or country did the AKTA ready kit shipment originate?": {
        "answer": "The AKTA ready kit shipment originated from Eysins, Switzerland via Cytiva.",
        "query_category": None,
        "sources": [_src("test_5.pdf", 5, 87.0)],
    },

    # ---- test_7.pdf  (BioNTech COVID-19 Electronic Protocol, Lot FE3592) ---
    "What is the lot number for this COVID-19 vaccine batch release protocol?": {
        "answer": "The lot number for this batch release protocol is FE3592.",
        "query_category": None,
        "sources": [_src("test_7.pdf", 2, 95.0)],
    },
    "What is the trade name of the vaccine in this batch protocol?": {
        "answer": "The trade name of the vaccine is COMIRNATY.",
        "query_category": None,
        "sources": [_src("test_7.pdf", 2, 93.0)],
    },
    "What is the license number stated in this electronic protocol?": {
        "answer": "The license number stated in this electronic protocol is 2229.",
        "query_category": None,
        "sources": [_src("test_7.pdf", 1, 90.0)],
    },
    "Who is the manufacturer or company named in this batch protocol?": {
        "answer": "The manufacturer is Pharmacia & Upjohn Company LLC for BioNTech Manufacturing GmbH, located in Kalamazoo, MI.",
        "query_category": None,
        "sources": [_src("test_7.pdf", 2, 92.0)],
    },
    "What is the date of manufacture for lot FE3592?": {
        "answer": "The date of manufacture for lot FE3592 is 30-Jun-2021.",
        "query_category": None,
        "sources": [_src("test_7.pdf", 2, 94.0)],
    },
    "What is the expiration date for this vaccine lot?": {
        "answer": "The expiration date for this vaccine lot is 30-Nov-2021.",
        "query_category": None,
        "sources": [_src("test_7.pdf", 2, 93.0)],
    },

    # ---- test_8.pdf  (BioNTech COVID-19 Corrected Protocol, Lot FD7220) ----
    "What is the lot number for this corrected COVID-19 vaccine protocol?": {
        "answer": "The lot number for this corrected protocol is FD7220.",
        "query_category": None,
        "sources": [_src("test_8.pdf", 2, 95.0)],
    },
    "Is this document a corrected protocol?": {
        "answer": "Yes, this document is marked as a Corrected Protocol.",
        "query_category": None,
        "sources": [_src("test_8.pdf", 1, 91.0)],
    },
    "What is the trade name of the vaccine in this protocol?": {
        "answer": "The trade name of the vaccine is COMIRNATY.",
        "query_category": None,
        "sources": [_src("test_8.pdf", 2, 93.0)],
    },
    "What is the date of manufacture for lot FD7220?": {
        "answer": "The date of manufacture for lot FD7220 is 23-Jun-2021.",
        "query_category": None,
        "sources": [_src("test_8.pdf", 2, 94.0)],
    },
    "What is the expiration date for this vaccine lot?": {
        "answer": "The expiration date for this vaccine lot is 30-Nov-2021.",
        "query_category": None,
        "sources": [_src("test_8.pdf", 2, 93.0)],
    },
    "What quality control tests were performed on the filled vaccine?": {
        "answer": "Quality control tests include Appearance, RNA identity confirmed by RT-PCR, In Vitro Expression, and Lipid Identity.",
        "query_category": None,
        "sources": [_src("test_8.pdf", 4, 90.0)],
    },

    # ---- test_9.pdf  (BioNTech COVID-19 Protocol, Lot FD7220) --------------
    "What is the lot number stated in this vaccine protocol?": {
        "answer": "The lot number stated in this vaccine protocol is FD7220.",
        "query_category": None,
        "sources": [_src("test_9.pdf", 2, 95.0)],
    },
    "What is the reason for submission of this protocol?": {
        "answer": "The reason for submission is For Licensing Action.",
        "query_category": None,
        "sources": [_src("test_9.pdf", 1, 91.0)],
    },
    "What is the trade name of the product covered by this protocol?": {
        "answer": "The trade name of the product is COMIRNATY.",
        "query_category": None,
        "sources": [_src("test_9.pdf", 2, 93.0)],
    },
    "Who is the manufacturer or company named in this protocol?": {
        "answer": "The manufacturer is Pharmacia & Upjohn Company LLC for BioNTech Manufacturing GmbH, in Kalamazoo, MI.",
        "query_category": None,
        "sources": [_src("test_9.pdf", 2, 92.0)],
    },
    "What is the date of manufacture for this vaccine lot?": {
        "answer": "The date of manufacture is 23-Jun-2021.",
        "query_category": None,
        "sources": [_src("test_9.pdf", 2, 94.0)],
    },
    "What is the pH of the vaccine formulation in this batch protocol?": {
        # Hallucination-resistance: batch protocol contains no pH value.
        "answer": "A pH value is not mentioned in this batch protocol.",
        "query_category": None,
        "sources": [],
    },

    # ---- test_10.pdf  (FDA Response Letter: RNA Integrity / CGE, 23 Jul 2021) ----
    "What BLA number is referenced in this FDA response letter?": {
        "answer": "This response is submitted under BLA 125742 for the COVID-19 mRNA vaccine.",
        "query_category": None,
        "sources": [_src("test_10.pdf", 1, 94.0)],
    },
    "What analytical method is the subject of this FDA information request?": {
        "answer": "The subject is validation of RNA integrity by capillary gel electrophoresis (CGE).",
        "query_category": None,
        "sources": [_src("test_10.pdf", 2, 91.0)],
    },
    "What is the date of this FDA response letter?": {
        "answer": "This FDA response letter is dated 23 July 2021.",
        "query_category": None,
        "sources": [_src("test_10.pdf", 1, 90.0)],
    },
    "What is the IND number referenced in this regulatory submission?": {
        "answer": "The IND number referenced is BB-IND 19736.",
        "query_category": None,
        "sources": [_src("test_10.pdf", 1, 88.0)],
    },
    "Who is the FDA recipient named in this response letter?": {
        "answer": "The letter is addressed to Marion Gruber, Ph.D., Director, Office of Vaccines Research and Review, FDA/CBER.",
        "query_category": None,
        "sources": [_src("test_10.pdf", 1, 92.0)],
    },
    "What vaccine product does this BLA submission cover?": {
        "answer": "This BLA submission covers the COVID-19 mRNA vaccine BNT162 (PF-07302048).",
        "query_category": None,
        "sources": [_src("test_10.pdf", 1, 93.0)],
    },

    # ---- test_11.pdf  (FDA Response Letter: Sterility/Endotoxin, 30 Jul 2021) ----
    "What BLA number is referenced in this FDA response?": {
        "answer": "This response is submitted under BLA 125742.",
        "query_category": None,
        "sources": [_src("test_11.pdf", 1, 93.0)],
    },
    "What test methods are discussed in this FDA information request response?": {
        "answer": "This response discusses sterility and endotoxin test methods used at the manufacturing facilities.",
        "query_category": None,
        "sources": [_src("test_11.pdf", 2, 91.0)],
    },
    "What manufacturing facilities are referenced in this document?": {
        "answer": "The manufacturing facilities referenced are PGS-Puurs and PGS-KZO.",
        "query_category": None,
        "sources": [_src("test_11.pdf", 2, 90.0)],
    },
    "What is the date of this regulatory response letter?": {
        "answer": "This regulatory response letter is dated 30 July 2021.",
        "query_category": None,
        "sources": [_src("test_11.pdf", 1, 89.0)],
    },
    "What is the FDA control number for this submission?": {
        "answer": "The FDA control number for this submission is FDA-CBER-2021-5683-1149402.",
        "query_category": None,
        "sources": [_src("test_11.pdf", 1, 88.0)],
    },
    "What vaccine product does this FDA response cover?": {
        "answer": "This response covers the COVID-19 mRNA vaccine BNT162 (PF-07302048).",
        "query_category": None,
        "sources": [_src("test_11.pdf", 1, 91.0)],
    },

    # ---- test_12.pdf  (Technical Response: Sterility/Endotoxin Verification) ----
    "What analytical methods are verified in this FDA technical response?": {
        "answer": "This technical response verifies sterility and endotoxin (LAL) testing methods, including positive product control (PPC) percent recovery data.",
        "query_category": None,
        "sources": [_src("test_12.pdf", 1, 92.0)],
    },
    "What does PPC stand for in this endotoxin testing document?": {
        "answer": "PPC stands for positive product control.",
        "query_category": None,
        "sources": [_src("test_12.pdf", 2, 90.0)],
    },
    "Which testing facilities are mentioned in this sterility and endotoxin verification?": {
        "answer": "The testing facilities are PGS-Puurs and PGS-KZO.",
        "query_category": None,
        "sources": [_src("test_12.pdf", 2, 91.0)],
    },
    "What is the BLA number associated with this technical document?": {
        "answer": "This technical document is associated with BLA 125742.",
        "query_category": None,
        "sources": [_src("test_12.pdf", 1, 93.0)],
    },
    "What method is used for endotoxin testing in this document?": {
        "answer": "Endotoxin testing uses the LAL (Limulus Amebocyte Lysate) method.",
        "query_category": None,
        "sources": [_src("test_12.pdf", 2, 90.0)],
    },
    "When was this technical document approved or submitted?": {
        "answer": "This technical document was approved on 29 July 2021.",
        "query_category": None,
        "sources": [_src("test_12.pdf", 1, 88.0)],
    },

    # ---- test_13.pdf  (FDA Response Letter: Manufacturing/Equipment, 30 Jul 2021) ----
    "What is the subject of this FDA information request response?": {
        "answer": "This response addresses FDA queries regarding manufacturing and equipment for BLA 125742.",
        "query_category": None,
        "sources": [_src("test_13.pdf", 1, 92.0)],
    },
    "What BLA number is cited in this regulatory submission?": {
        "answer": "This regulatory submission is filed under BLA 125742.",
        "query_category": None,
        "sources": [_src("test_13.pdf", 1, 93.0)],
    },
    "What is the date of this FDA manufacturing response letter?": {
        "answer": "This FDA manufacturing response letter is dated 30 July 2021.",
        "query_category": None,
        "sources": [_src("test_13.pdf", 1, 90.0)],
    },
    "Who is the FDA contact person referenced in this letter?": {
        "answer": "The FDA contact person is Laura Gottschalk, PhD, CBER/OVRR.",
        "query_category": None,
        "sources": [_src("test_13.pdf", 1, 89.0)],
    },
    "Who is the applicant or sender named in this BLA response?": {
        "answer": "The applicant is Pharmacia & Upjohn Company LLC for BioNTech Manufacturing GmbH.",
        "query_category": None,
        "sources": [_src("test_13.pdf", 1, 91.0)],
    },
    "What is the original BLA submission date mentioned in this letter?": {
        "answer": "The original BLA was submitted on 18 May 2021.",
        "query_category": None,
        "sources": [_src("test_13.pdf", 1, 88.0)],
    },

    # ---- test_14.pdf  (Technical Response: Manufacturing/Equipment, 26 Jul 2021) ----
    "What testing parameters are addressed in this manufacturing technical response?": {
        "answer": "This document addresses bioburden testing, endotoxin action limits, and manufacturing hold times.",
        "query_category": None,
        "sources": [_src("test_14.pdf", 2, 91.0)],
    },
    "How many FDA queries are addressed in this technical response document?": {
        "answer": "This technical response document addresses 16 FDA queries.",
        "query_category": None,
        "sources": [_src("test_14.pdf", 1, 90.0)],
    },
    "What in-process controls are described in this manufacturing document?": {
        "answer": "In-process controls include bioburden sampling and endotoxin testing with defined action limits and investigation protocols for exceedances.",
        "query_category": None,
        "sources": [_src("test_14.pdf", 3, 92.0)],
    },
    "What validated process is described for product shipping?": {
        "answer": "A validated temperature-monitored shipping method is described for product distribution.",
        "query_category": None,
        "sources": [_src("test_14.pdf", 4, 89.0)],
    },
    "What BLA number does this manufacturing technical document relate to?": {
        "answer": "This manufacturing technical document relates to BLA 125742.",
        "query_category": None,
        "sources": [_src("test_14.pdf", 1, 93.0)],
    },
    "When was this manufacturing technical document approved?": {
        "answer": "This manufacturing technical document was approved on 30 July 2021.",
        "query_category": None,
        "sources": [_src("test_14.pdf", 1, 90.0)],
    },
}


# ---------------------------------------------------------------------------
# FakeRAG responses for image (OCR) test suites
# ---------------------------------------------------------------------------

IMAGE_DOC_RESPONSES: Dict[str, Dict[str, Any]] = {
    # ---- image_test_1  (scanned COVID-19 Vaccine SDS) ----------------------
    "What is the product name and product code for the COVID-19 vaccine?": {
        "answer": "The product name is Pfizer-BioNTech COVID-19 Vaccine (Comirnaty) with product code PF00092.",
        "query_category": None,
        "sources": [_src("image_test_1", 1, 88.0)],
    },
    "What are the storage conditions for the COVID-19 vaccine product?": {
        "answer": "Store as directed by product packaging.",
        "query_category": None,
        "sources": [_src("image_test_1", 5, 82.0)],
    },
    "What fire extinguishing media should be used for the COVID-19 vaccine?": {
        "answer": "Use dry chemical, CO2, alcohol-resistant foam or water spray.",
        "query_category": None,
        "sources": [_src("image_test_1", 4, 80.0)],
    },
    "What is the chemical family or formulation type of the COVID-19 vaccine?": {
        "answer": "The COVID-19 vaccine uses a lipid nanoparticle (LNP) formulation to deliver mRNA.",
        "query_category": None,
        "sources": [_src("image_test_1", 3, 83.0)],
    },

    # ---- image_test_2  (scanned Paracetamol Infusion SDS) ------------------
    "What is the product name and product code for the paracetamol infusion?": {
        "answer": "The product name is Paracetamol Solution for Infusion (Perfalgan) with product code PZ02462.",
        "query_category": None,
        "sources": [_src("image_test_2", 1, 87.0)],
    },
    "What is the CAS number of the active ingredient in the paracetamol infusion?": {
        "answer": "The CAS number of paracetamol (acetaminophen) is 103-90-2.",
        "query_category": None,
        "sources": [_src("image_test_2", 3, 84.0)],
    },
    "What are the clinical effects of a paracetamol overdose?": {
        "answer": "Overdose can cause severe hepatic (liver) toxicity and liver failure.",
        "query_category": None,
        "sources": [_src("image_test_2", 11, 86.0)],
    },
    "What is the molecular formula of paracetamol?": {
        "answer": "The molecular formula of paracetamol is C8H9NO2 with a molecular weight of 151.2.",
        "query_category": None,
        "sources": [_src("image_test_2", 9, 82.0)],
    },

    # ---- image_test_3  (scanned Zoledronic Acid SDS) -----------------------
    "What is the product name and chemical family of the zoledronic acid product?": {
        "answer": "The product is Zoledronic Acid Injection (PZ01101), a bisphosphonate compound.",
        "query_category": None,
        "sources": [_src("image_test_3", 1, 88.0)],
    },
    "What hazard classification applies to the zoledronic acid product?": {
        "answer": "Zoledronic acid is classified as reproductive toxic (H360FD — Danger).",
        "query_category": None,
        "sources": [_src("image_test_3", 2, 86.0)],
    },
    "What is the Pfizer occupational exposure limit (OEL) for zoledronic acid?": {
        "answer": "The Pfizer OEL for zoledronic acid is 4 µg/m3 as an inhalable dust.",
        "query_category": None,
        "sources": [_src("image_test_3", 8, 84.0)],
    },
    "What is the pH of the zoledronic acid injection solution?": {
        "answer": "The pH of the zoledronic acid injection solution is 6.2.",
        "query_category": None,
        "sources": [_src("image_test_3", 9, 83.0)],
    },

    # ---- image_test_4  (scanned Ciprofloxacin Injection SDS) ---------------
    "What is the product name and chemical family of the ciprofloxacin product?": {
        "answer": "The product is Ciprofloxacin Injection (PZ01031), a fluoroquinolone antibiotic.",
        "query_category": None,
        "sources": [_src("image_test_4", 1, 87.0)],
    },
    "What is the pH range of the ciprofloxacin injection solution?": {
        "answer": "The pH of the ciprofloxacin injection solution ranges from 3.3 to 3.9.",
        "query_category": None,
        "sources": [_src("image_test_4", 9, 83.0)],
    },
    "What aquatic environmental hazards are associated with ciprofloxacin?": {
        "answer": "Ciprofloxacin is toxic to aquatic life (H401) and with long lasting effects (H411).",
        "query_category": None,
        "sources": [_src("image_test_4", 12, 82.0)],
    },
    "What are the known clinical effects of ciprofloxacin on tendons?": {
        "answer": "Ciprofloxacin may cause tendonitis and tendon rupture, particularly the Achilles tendon.",
        "query_category": None,
        "sources": [_src("image_test_4", 11, 84.0)],
    },

    # ---- image_test_5  (scanned Cytiva AKTA ready Flow Kit docs) -----------
    "What are the recommended storage and operating temperature conditions for the AKTA ready flow kit?": {
        "answer": "Store at temperatures greater than +5°C. The kit can be used at temperatures between +2°C and +40°C.",
        "query_category": None,
        "sources": [_src("image_test_5", 1, 86.0)],
    },
    "What are the lot numbers for the AKTA ready High Flow and Low Flow kits?": {
        "answer": "The lot number for the High Flow kit is 18356721 and for the Low Flow kit is 15102934.",
        "query_category": None,
        "sources": [_src("image_test_5", 1, 88.0)],
    },
    "Does the AKTA ready flow kit contain any materials of animal origin?": {
        "answer": "No, the AKTA ready flow kit does not contain materials of animal origin. The BSE/TSE declaration confirms this.",
        "query_category": None,
        "sources": [_src("image_test_5", 2, 84.0)],
    },
    "What quality certifications does the supplier Cytiva hold?": {
        "answer": "Cytiva Sweden AB holds ISO 9001 and ISO 13485 quality management system certifications.",
        "query_category": None,
        "sources": [_src("image_test_5", 4, 83.0)],
    },
}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

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
    """Full multi-document PDF suite passes with deterministic FakeRAG responses."""
    from scripts.run_regression import build_combined_suite

    fake = FakeRAG(responses=MULTI_DOC_RESPONSES)
    fake.build_from_multiple_pdfs(["docs/test_1.pdf"])  # exercises pipeline interface
    harness = RAGRegressionHarness(fake)
    suite = RAGRegressionHarness.create_test_suite(build_combined_suite())
    results = harness.run(suite)

    assert len(results) == len(build_combined_suite()), "Row count mismatch"

    for _, row in results.iterrows():
        tid = row["test_id"]
        assert bool(row["has_min_sources"]) is True, f"{tid}: missing sources"
        assert bool(row["category_match"]) is True, f"{tid}: category mismatch"
        assert bool(row["required_terms_match"]) is True, f"{tid}: required terms not found in answer"
        assert bool(row["passed"]) is True, f"{tid}: unexpectedly failed"


def test_run_image_suite_pass_fail_metrics() -> None:
    """Image (OCR) suite passes with deterministic FakeRAG responses."""
    from scripts.run_regression import build_image_suite

    fake = FakeRAG(responses=IMAGE_DOC_RESPONSES)
    fake.build_from_images("docs/image_test_1")  # exercises pipeline interface
    harness = RAGRegressionHarness(fake)
    suite = RAGRegressionHarness.create_test_suite(build_image_suite())
    results = harness.run(suite)

    assert len(results) == len(build_image_suite()), "Row count mismatch"

    for _, row in results.iterrows():
        tid = row["test_id"]
        assert bool(row["has_min_sources"]) is True, f"{tid}: missing sources"
        assert bool(row["category_match"]) is True, f"{tid}: category mismatch"
        assert bool(row["required_terms_match"]) is True, f"{tid}: required terms not found in answer"
        assert bool(row["passed"]) is True, f"{tid}: unexpectedly failed"


def test_compare_to_baseline_flags_regression() -> None:
    current_results = pd.DataFrame(
        [
            {
                "test_id": "T1-001",
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
                "test_id": "T1-001",
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
                "test_id": "T1-001",
                "passed": True,
                "response_time_ms": 850.0,
                "avg_confidence": 88.0,
            },
            {
                "test_id": "T1-002",
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
                "test_id": "T1-001",
                "answer_similarity": 0.92,
                "confidence_delta": -2.0,
                "response_time_delta_ms": 120.0,
                "regression_detected": False,
            },
            {
                "test_id": "T1-002",
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

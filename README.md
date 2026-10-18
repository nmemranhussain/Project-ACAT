# Project ACAT: Acute Care Agentic Triage

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Clinical Architecture: OMOP CDM](https://img.shields.io/badge/EHR%20Standard-OMOP%20CDM%20v5.3-red.svg)](https://www.ohdsi.org/)

**Project ACAT** is an end-to-end Clinical Decision Support System (CDSS) designed for emergency department triage and 30-day cardiovascular deterioration stratification. It integrates the **OMOP Common Data Model (CMS Synthetic Public Schema)**, a cost-sensitive **LightGBM** classifier with observation-level **TreeSHAP** attributions, a semantic vector store indexing the **2022 AHA/ACC/HFSA Heart Failure Guidelines**, and a 4-agent consensus engine driven by **Gemini 2.5 Flash**.

---

## 1. System Architecture

```mermaid
flowchart TD
    A[Unstructured Clinical Narrative] --> B[Agent 1: Vitals Normalizer<br/>Gemini 2.5 Flash + Pydantic]
    B --> C[PatientVitals Schema<br/>LVEF, Cr, SBP, Age, Comorbidities]
    C --> D[Agent 2: Risk Analyst<br/>LightGBM + TreeSHAP Engine]
    D --> E{Predicted Risk >= 0.7269?}
    E -- Yes: Critical --> F[Agent 3: Guideline Specialist<br/>ChromaDB RAG - 2022 AHA Guidelines]
    E -- No: Standard --> G[Standard Monitoring Protocol]
    F --> H[Agent 4: Clinical Auditor<br/>Gemini 2.5 Flash]
    G --> H
    H --> I[Executive Bedside Brief &<br/>Drug Contraindication Check]
    I --> J[Point-of-Care Gradio Dashboard]
```

"""
Clinical Multi-Agent Orchestrator Module for Project ACAT.
Powered by Gemini 2.5 Flash, LightGBM (with TreeSHAP), and ChromaDB RAG.
"""

import json
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from src.model_engine import CardiacRiskEngine
from src.rag_engine import ClinicalGuidelineRAG

# 1. Pydantic Extraction Schema
class PatientVitals(BaseModel):
    age: int = Field(description="Patient age in years")
    is_male: int = Field(description="1 if male, 0 if female")
    length_of_stay: int = Field(default=3, description="Estimated or current inpatient length of stay in days")
    ejection_fraction: float = Field(description="Left Ventricular Ejection Fraction (LVEF %)")
    serum_creatinine: float = Field(description="Serum creatinine level in mg/dL")
    systolic_bp: int = Field(description="Systolic blood pressure in mmHg")
    has_diabetes: int = Field(description="1 if diagnosed with Type 2 Diabetes, else 0")
    prior_comorbidity_count: int = Field(default=2, description="Count of pre-existing chronic conditions")
    symptoms_summary: str = Field(description="Summary of clinical symptoms and NYHA class")

class MultiAgentClinicalOrchestrator:
    def __init__(self, project_id: str):
        self.project_id = project_id
        self.ai_client = genai.Client(vertexai=True, project=project_id, location="us-central1")
        self.risk_engine = CardiacRiskEngine(model_path="src/model.joblib", metadata_path="src/model_metadata.json")
        self.rag_engine = ClinicalGuidelineRAG(project_id=project_id, persist_dir="data/chroma_hf_rag")

    # Agent 1: Vitals Normalizer (Gemini Flash + Pydantic Schema)
    def agent_vitals_normalizer(self, physician_note: str) -> PatientVitals:
        sys_instruction = (
            "You are an expert clinical data extraction officer. "
            "Extract structured medical parameters precisely from raw physician notes. "
            "If any parameter is not explicitly stated, infer a clinically safe standard value based on context."
        )
        response = self.ai_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=f"Extract clinical variables from this note:\n{physician_note}",
            config=types.GenerateContentConfig(
                system_instruction=sys_instruction,
                response_mime_type="application/json",
                response_schema=PatientVitals,
                temperature=0.0
            )
        )
        return PatientVitals.model_validate_json(response.text)

    # Agent 2: Risk Analyst (Native LightGBM + Observation TreeSHAP)
    def agent_risk_analyst(self, vitals: PatientVitals) -> dict:
        patient_dict = {
            'age': vitals.age,
            'is_male': vitals.is_male,
            'length_of_stay': vitals.length_of_stay,
            'ejection_fraction': vitals.ejection_fraction,
            'serum_creatinine': vitals.serum_creatinine,
            'systolic_bp': vitals.systolic_bp,
            'has_diabetes': vitals.has_diabetes,
            'prior_comorbidity_count': vitals.prior_comorbidity_count
        }
        return self.risk_engine.predict_risk(patient_dict)

    # Agent 3: Guideline Specialist (Semantic Search on ChromaDB)
    def agent_guideline_specialist(self, vitals: PatientVitals, risk_assessment: dict) -> list:
        if not risk_assessment["is_critical"]:
            return []
        return self.rag_engine.retrieve_guidelines(
            ef=vitals.ejection_fraction,
            creatinine=vitals.serum_creatinine,
            sbp=vitals.systolic_bp,
            k=3
        )

    # Agent 4: Clinical Auditor & Bedside Brief Synthesizer
    def agent_clinical_auditor(self, vitals: PatientVitals, risk: dict, guidelines: list) -> str:
        prompt = f"""
        You are an expert ICU Clinical Audit Director. Reconcile this cardiac evaluation into an Executive Bedside Brief:
        
        1. Extracted Patient Vitals & Clinical Picture:
        {vitals.model_dump_json(indent=2)}

        2. Machine Learning Predictive Risk Stratification (LightGBM + TreeSHAP):
        {json.dumps(risk, indent=2)}

        3. Evidence-Based 2022 AHA/ACC/HFSA Heart Failure Guidelines Retrieved:
        {json.dumps(guidelines, indent=2)}

        Clinical Reporting Guidelines:
        - Provide an executive clinical summary of the patient's cardiorenal risk profile.
        - Interpret the model's predicted probability against the calibrated decision threshold ({risk['decision_threshold']}).
        - Highlight the top TreeSHAP risk drivers and explain their biological significance.
        - Check drug contraindications:
            * If SBP < 100 mmHg, explicitly flag hypotension risk with ARNi / vasodilators.
            * If Serum Creatinine > 2.0 mg/dL, evaluate MRA hyperkalemia and renal threshold requirements.
            * Highlight SGLT2 inhibitor recommendations.
        """
        response = self.ai_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.2)
        )
        return response.text

    # End-to-End Orchestrator Pipeline
    def run_pipeline(self, physician_note: str) -> dict:
        vitals = self.agent_vitals_normalizer(physician_note)
        risk = self.agent_risk_analyst(vitals)
        guidelines = self.agent_guideline_specialist(vitals, risk)
        brief = self.agent_clinical_auditor(vitals, risk, guidelines)
        return {
            "vitals": vitals.model_dump(),
            "risk": risk,
            "guidelines": guidelines,
            "brief": brief
        }

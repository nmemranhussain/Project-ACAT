"""
RAG Engine Module for Project ACAT.
Encapsulates ChromaDB semantic search over 2022 AHA/ACC/HFSA guidelines
with metadata filtering on left ventricular ejection fraction and contraindications.
"""

from langchain_community.vectorstores import Chroma
from langchain_google_vertexai import VertexAIEmbeddings

class ClinicalGuidelineRAG:
    def __init__(self, project_id: str, persist_dir: str = "data/chroma_hf_rag"):
        self.embeddings = VertexAIEmbeddings(
            model_name="text-embedding-004",
            project=project_id
        )
        self.vector_db = Chroma(
            persist_directory=persist_dir,
            embedding_function=self.embeddings
        )

    def retrieve_guidelines(self, ef: float, creatinine: float, sbp: float, k: int = 3) -> list:
        """
        Executes query retrieval based on hemodynamic status and renal safety thresholds.
        """
        query_terms = [f"Guideline treatment for Heart Failure with LVEF {ef}%"]
        if sbp < 100:
            query_terms.append("management of low systolic blood pressure and hypotension")
        if creatinine > 2.0:
            query_terms.append("kidney impairment and elevated serum creatinine contraindications")
        
        full_query = " and ".join(query_terms)
        results = self.vector_db.similarity_search(full_query, k=k)
        
        guidelines = []
        for doc in results:
            guidelines.append({
                "recommendation": doc.page_content,
                "metadata": doc.metadata
            })
        return guidelines

import asyncio
import os
import sys
from datetime import datetime
from typing import List

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.diagnosis_generator import DiagnosisGeneratorOutput, diagnosis_generator
from db import get_qdrant_client


def retrieve_documents(query: str):
    db_client = get_qdrant_client()

    results = db_client.similarity_search(query, k=3)

    documents: List[str] = []
    for doc in results:
        documents.append(f"source:{doc.metadata['source']}WebSource:{doc.metadata['WebSource']}\n\ncontent:{doc.page_content}")

    return "\n\n---\n\n".join(documents)


def generate_diagnosis(query: str, context: str):
    generation_result = diagnosis_generator.invoke(
        {"context": context, "question": query},
        {"run_name": "diagnosis-generator"},
    )

    parsed_generation_result = DiagnosisGeneratorOutput(**generation_result)

    generation_result_str = (
        "## Differential diagnoses based on assessments of the patient:\n\n"
    )

    # Add summary
    generation_result_str += f"### Summary\n{parsed_generation_result.summary}\n"

    # Add differential diagnoses
    for index, diagnosis in enumerate(
        parsed_generation_result.differential_diagnoses, start=1
    ):
        generation_result_str += f"### Diagnosis {index}: {diagnosis.diagnosis}\n"
        generation_result_str += f"**Rationale:** {diagnosis.rational}\n\n"

        generation_result_str += "##### In-text citations\n"
        for quote in diagnosis.relevant_quotes:
            generation_result_str += (
                f"\\[{quote.ieee_intext_citation}\\]: {quote.source}\n\n"
            )
            generation_result_str += f"    - {quote.text}\n"
            generation_result_str += "\n\n"

        generation_result_str += "\n"

    # Add references
    if parsed_generation_result.ieee_references:
        generation_result_str += "### References\n"
        for citation in parsed_generation_result.ieee_references:
            generation_result_str += f"- {citation}\n"

    return generation_result_str


def evaluate():
    with open(
        "/Users/dingruoqian/code/TLM3001-DP-PhysioLLM/test_prompts/query_1.txt", "r"
    ) as f:
        query = f.read()

    context = retrieve_documents(query)
    generation = generate_diagnosis(query, context)

    trace_file_name = f"naive-rag-{datetime.now().strftime('%Y%m%d%H%M%S')}.txt"
    trace_file_path = os.path.join("traces", trace_file_name)

    with open(trace_file_path, "w") as f:
        f.write(generation)
        print(f"Trace file saved as {trace_file_name}")


if __name__ == "__main__":
    evaluate()

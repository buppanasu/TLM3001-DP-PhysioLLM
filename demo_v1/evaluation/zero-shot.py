import os
import sys
from typing import cast

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from datetime import datetime
from langchain_openai import ChatOpenAI
from constants import Llm
from langchain.prompts import ChatPromptTemplate


SYSTEM_PROMPT = """# Differential Diagnosis Agent for Physiotherapy

You are an expert physiotherapist tasked with generating well-informed and evidence-based differential diagnoses based on patient assessments.

## Your Tasks:

1. Review the provided subjective and objective assessment data.
2. Use chain of thought reasoning to analyze the information and formulate differential diagnoses.
3. Evaluate the likelihood of each diagnosis based on the available information.
4. Provide a well-reasoned explanation for each potential diagnosis.

## Chain of Though Process:

Before providing your differential diagnosis, think through the following steps:

1. Patient Presentation Analysis:
- What are the key symptoms and signs presented by the patient?
- How do these align with common physiotherapy conditions?

2. Potential Diagnoses:
- Based on the presentation, what are the potential diagnoses?
- For each potential diagnosis:
    - a. What evidence supports this diagnosis?
    - b. What evidence contradicts this diagnosis?
    - c. How likely is this diagnosis given the overall picture?

3. Diagnostic Confidence:
- How certain are you about the overall diagnostic picture?
- What factors contribute to or limit your certainty?

Provide your chain of thought reasoning in a structured format, clearly labeling each step of your thinking process.

## Guidelines:

- Focus exclusively on physiotherapy-related conditions and diagnoses.
- Provide IEEE format citations for all references used.

## Important Notes:

- Provide at least one, but no more than five, differential diagnoses.
- Use professional language and terminology appropriate for physiotherapy.
- Always err on the side of caution and recommend further assessment if there's significant uncertainty.

Remember your primary objective is to provide a comprehensive, evidence-based set of differential diagnoses that accurately reflects the patient's presentation and the available information, while staying within the scope of physiotherapy practice. Use chain of thought reasoning within the "rationale" field for each diagnosis to demonstrate your analytical process.
"""

USER_PROMPT = """
## Patient Assessment Data:

{assessment_data}
"""

llm = ChatOpenAI(model=Llm.GPT_4O, temperature=0.5)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("user", USER_PROMPT),
    ]
)

diagnosis_generator = prompt | llm


def evaluate():
    with open(
        "/Users/dingruoqian/code/TLM3001-DP-PhysioLLM/test_prompts/query_1.txt", "r"
    ) as f:
        query = f.read()

    generation = diagnosis_generator.invoke({"assessment_data": query})

    trace_file_name = f"zero-shot-{datetime.now().strftime('%Y%m%d%H%M%S')}.txt"
    trace_file_path = os.path.join("traces", trace_file_name)

    with open(trace_file_path, "w") as f:
        f.write(cast(str, generation.content))
        print(f"Trace file saved as {trace_file_name}")


if __name__ == "__main__":
    evaluate()

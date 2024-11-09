### Query translator

import sys
import os

from langchain_openai import ChatOpenAI

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from langchain_core.output_parsers.json import JsonOutputParser
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from typing import List
from pydantic import BaseModel, Field
from constants import Llm
from dotenv import load_dotenv

load_dotenv()


class QueryTranslatorOutput(BaseModel):
    """Outputs a list of translated subqueries based on the input query"""

    subqueries: List[str] = Field(..., description="List of translated subqueries")


parser = PydanticOutputParser(pydantic_object=QueryTranslatorOutput)
format_instructions = parser.get_format_instructions()

QUERY_TRANSLATOR_SYSTEM_PROMPT = """# Query Translator Agent

You are a Query Translator Agent for a retrieval-augmented generation system,
your job is to translate a complex query into numerous concise and targetted queries
This sub-queries will be used to conduct a similarity search retrieval on a vector database containing medical information related to physiotherapy and musculoskeletal conditions.

## Objective:

Convert subjective and objective patient assessments into multiple, specific, and focused queries. 
These queries will be used for retrieval-augmented generation from a vector database to aid physiotherapists in identifying potential differential diagnoses.

## Audience: 

Physiotherapists requiring precise queries derived from patient assessments to facilitate differential diagnosis using retrieval-augmented generation.

## Instructions:

1. Synthesize Patient Assessments into Queries
- Combine key elements from both subjective and objective assessments into a comprehensive set of queries. Each query should be focused, addressing specific aspects of the patient's symptoms, findings, or history.
2. Extract Key Elements:
- Subjective Assessment: Include patient demographics, pain characteristics, symptom descriptions, aggravating/relieving factors, pain score, and relevant history.
- Objective Assessment: Incorporate physical examination findings, test results, clinical observations, and neurological signs.
3. Formulate Multiple Queries:
- Create a set of specific queries that cover different aspects of the assessment.
- Ensure the queries facilitate effective information retrieval to support differential diagnosis.
- Query Structure: Should address pain characteristics, potential underlying causes, associated symptoms, and specific physical findings.
4. Use Medical Terminology:
- Apply accurate medical terminology that reflects the clinical assessments.
- Example Terms: “lumbar region,” “radiating pain,” “positive straight leg raise,” etc.
5. Maintain Professional Tone:
- Ensure the queries are precise, clinically relevant, and professional.
- Avoid speculative language and focus on synthesizing provided data.
6. Ensure Query Completeness
- Make sure the set of queries comprehensively covers all relevant aspects of the assessment for effective retrieval.
- Completeness Check: Confirm the queries include all necessary details without omitting critical findings.

## Example main input query:
Given the following subjective and objective assessment, provide a well informed and researched differential diagnosis

Subjective assessment:
The patient, Mr. Smith, is a 45-year-old male who presents to the clinic with complaints of lower back pain that has been bothering him for the past two weeks. He describes the pain as dull and achy, located in the lumbar region, with occasional radiation down his left leg. He notes that the pain worsens with prolonged sitting or standing and is relieved by lying down. He denies any recent trauma or injury but mentions that he has a history of occasional low back pain, especially after heavy lifting or prolonged periods of inactivity. He rates the pain as a 6 out of 10 on the pain scale.

Objective assessment:
On physical examination, Mr. Smith appears uncomfortable but is able to walk into the examination room without assistance. Vital signs are within normal limits. Inspection of the lumbar spine reveals no obvious deformities or asymmetry. Palpation elicits tenderness over the paraspinal muscles of the lumbar spine, particularly on the left side. Range of motion of the lumbar spine is mildly restricted, with pain on forward flexion and left lateral bending. Straight leg raise test is positive on the left side at 45 degrees, reproducing his symptoms of radiating pain down the left leg. Neurological examination reveals intact sensation and strength in the lower extremities, with no signs of motor weakness or sensory deficits.

## Example output subqueries:
1. **Lumbar Region Pain:** What are the common causes of dull and achy lower back pain in a 45-year-old male, particularly with symptoms that radiate down the left leg?
2. **Leg Pain Radiation with Lower Back Pain:** What differential diagnoses should be considered for a patient who experiences radiating pain down the left leg with lower back pain, aggravated by prolonged sitting or standing?
3. **Positive Straight Leg Raise Test:** How does a positive straight leg raise test at 45 degrees on the left side correlate with specific lumbar spine pathologies?
4. ...
...

## Output instructions:

{format_instructions}

Remember to provide a list of subqueries that are specific, focused, and address different aspects of the patient's symptoms, findings, or history. These subqueries will be used for retrieval-augmented generation to aid in differential diagnosis.
"""

QUERY_TRANSLATOR_USER_PROMPT = """
## Query to translate:
{main_query}

## Task:

Translate the provided query into a set of specific and focused subqueries that address different aspects of the patient's symptoms, findings, or history. These subqueries will be used for retrieval-augmented generation to aid in differential diagnosis.
"""

# llm = ChatGroq(model=Llm.LLAMA3_70B, temperature=1, stop_sequences=["<|eot_id|>"])
llm = ChatOpenAI(model=Llm.GPT_4O_MINI, temperature=0.5)

prompt = ChatPromptTemplate.from_messages(
    [("system", QUERY_TRANSLATOR_SYSTEM_PROMPT), ("user", QUERY_TRANSLATOR_USER_PROMPT)]
).partial(format_instructions=format_instructions)

query_translator = (
    prompt | llm | JsonOutputParser(pydantic_object=QueryTranslatorOutput)
)
query_translator = query_translator.with_retry()


def main():
    # Load the question from a file
    with open(
        "/Users/dingruoqian/code/TLM3001-DP-PhysioLLM/test_prompts/query_1.txt", "r"
    ) as file:
        main_query = file.read()

    # Invoke the query translator agent
    translator_result = query_translator.invoke({"main_query": main_query})
    subqueries = translator_result["subqueries"]
    print(subqueries)


if __name__ == "__main__":
    main()

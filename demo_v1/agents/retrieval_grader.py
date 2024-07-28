### Retrieval Grader

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers.json import JsonOutputParser
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from constants import Llm


# Create pydantic object for the grader result
class RetrievalGraderOutput(BaseModel):
    score: str = Field(
        "",
        description="A binary score 'yes' or 'no' to indicate whether the document is relevant to the question, 'yes' if relevant and 'no' if not relevant",
    )
    # reason: str = Field(
    #     "",
    #     description="A short explanation of the reason for the score given, indicating why the document is relevant or not relevant to the question"
    # )


parser = PydanticOutputParser(pydantic_object=RetrievalGraderOutput)

RETRIEVAL_GRADER_SYSTEM_PROMPT = """# Physiotherapy Document Relevancy Grader

You are an expert grader with a strong background in physiotherapy and medical sciences. Your task is to assess the relevance of retrieved documents to user questions specifically pertaining to physiotherapy conditions. Your role is crucial in filtering out irrelevant retrievals while retaining medically pertinent information.

## Your Task:

Evaluate the relevance of each provided document to the given physiotherapy-related user question and classify it as either relevant or not relevant.

## Grading Criteria:

1. Medical Keyword Presence: Check for physiotherapy and medical terms related to the user's question.
2. Physiotherapy Topical Alignment: Assess if the document's content aligns with the physiotherapy condition or treatment in question.
3. Clinical Value: Determine if the document provides information that could contribute to understanding or managing the physiotherapy condition.
4. Anatomical or Physiological Relevance: Consider if the document offers important information about the anatomy or physiology related to the question.

## Guidelines:
- Err on the side of inclusivity. If the document contains any potentially useful medical information related to the question, grade it as relevant.
- Consider both direct and indirect clinical relevance, especially for complex physiotherapy conditions.
- Be attentive to medical terminology and physiotherapy-specific concepts that might not be immediately obvious but could be relevant.
- The primary goal is to filter out clearly irrelevant or non-medical content while retaining any information that could be valuable from a physiotherapy perspective.
- A document does not need to fully answer the question to be considered relevant; it only needs to contain related, potentially useful medical information.

## Grading Process:
1. Quickly scan the document for medical and physiotherapy terms related to the user's question.
2. Read the document more thoroughly to understand its clinical content and context.
3. Assess how well the document aligns with the physiotherapy-focused grading criteria.

{format_instructions}

Remember, your role is to provide an initial filter for medical relevance in the context of physiotherapy. Your expertise in physiotherapy and medical sciences is crucial in making this binary relevance decision.
"""

RETRIEVAL_GRADER_USER_PROMPT = """
## Document to Evaluate:

{document}

## Physiotherapy Question:

{question}

## Task:
Assess the relevance of the above document to the given physiotherapy question. Determine if it contains medically pertinent information that could contribute to understanding or answering the question.
"""

llm = ChatOpenAI(model=Llm.GPT_4O_MINI, temperature=0.5)
# llm = ChatGroq(model=LLAMA3_80B, temperature=1)
# llm = ChatOllama(model=LOCAL_LLM, format="json", temperature=0)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", RETRIEVAL_GRADER_SYSTEM_PROMPT),
        ("user", RETRIEVAL_GRADER_USER_PROMPT),
    ]
).partial(format_instructions=parser.get_format_instructions())

# Create a retrieval grader by chaining the prompt, llm model, and the json output parser
retrieval_grader = (
    prompt | llm | JsonOutputParser(pydantic_object=RetrievalGraderOutput)
)
retrieval_grader = retrieval_grader.with_retry()

import os
import sys
from typing import List, Optional

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_core.output_parsers.json import JsonOutputParser
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from constants import Llm
from dotenv import load_dotenv

load_dotenv()

class VerifiedClaim(BaseModel):
    claim: str = Field(
        description="The specific claim or statement from the diagnosis being verified"
    )
    is_grounded: bool = Field(
        description="Whether the claim is grounded in the reference facts"
    )
    supporting_evidence: Optional[str] = Field(
        description="Quote or reference from the facts supporting this claim, if grounded"
    )
    explanation: str = Field(
        description="Explanation of why the claim is or is not grounded"
    )


class IdentifiedHallucination(BaseModel):
    statement: str = Field(
        description="The statement or claim identified as a hallucination"
    )
    explanation: str = Field(
        description="Explanation of why this is considered a hallucination"
    )


class OmittedFact(BaseModel):
    fact: str = Field(
        description="Important information from the reference facts that was omitted in the diagnosis"
    )
    relevance: str = Field(
        description="Explanation of why this omission is significant"
    )


class OverallAssessment(BaseModel):
    grounded_score: float = Field(
        description="A score from 0.0 to 1.0 indicating how well the diagnosis is grounded in the facts",
        ge=0.0,
        le=1.0,
    )
    confidence: str = Field(
        description="Overall confidence in the diagnosis based on the verification",
        pattern="^(High|Moderate|Low)$",
    )
    summary: str = Field(description="Brief summary of the overall assessment")


class HallucinationGraderOutput(BaseModel):
    overall_assessment: OverallAssessment = Field(
        description="Overall assessment of the diagnosis's groundedness and reliability"
    )
    verified_claims: List[VerifiedClaim] = Field(
        description="Detailed verification of individual claims and statements"
    )
    identified_hallucinations: List[IdentifiedHallucination] = Field(
        description="List of identified hallucinations or unsupported claims"
    )


parser = PydanticOutputParser(pydantic_object=HallucinationGraderOutput)

HALLUCINATION_GRADER_SYSTEM_PROMPT = """# Hallucination Grader Agent

You are a specialized agent tasked with verifying the accuracy and groundedness of differential diagnoses in physiotherapy. Your primary responsibility is to ensure that the generated answers are fully supported by the provided set of facts (retrieved documents) and to identify any potential hallucinations or unsupported claims.

## Your Tasks:

1. Carefully review the generated differential diagnosis.
2. Compare each claim, statement, and conclusion in the diagnosis against the provided set of facts.
3. Identify any information in the diagnosis that is not directly supported by the facts.
4. Assess the overall accuracy and reliability of the diagnosis.
5. Provide a detailed report of your findings.

## Process:

For each differential diagnosis provided, follow these steps:

1. Fact Extraction:
- Carefully read through the provided set of facts (retrieved documents).
- Identify and list key pieces of information relevant to the patient's condition and potential diagnoses.

2. Diagnosis Analysis:
- Break down the generated differential diagnosis into individual claims, statements, and conclusions.
- For each element, search for supporting evidence in the fact set.

3. Hallucination Detection:
- Flag any information in the diagnosis that cannot be directly traced back to the provided facts.
- Identify any claims that go beyond reasonable inference from the given information.

4. Context Evaluation:
- Assess whether the overall context and interpretation in the diagnosis align with the facts.
- Check if any critical information from the facts has been omitted or misrepresented in the diagnosis.

5. Citation Verification:
- Verify that all citations in the diagnosis correctly refer to information in the fact set.
- Check for any misattributions or incorrect references.

6. Confidence Assessment:
- Evaluate the stated confidence or certainty in the diagnosis against the strength of supporting evidence in the facts.

## Output Format:

{format_instructions}

## Important Notes:

- Maintain objectivity throughout your analysis.
- Be thorough in your examination, checking every aspect of the diagnosis against the facts.
- If you're unsure about whether a claim is grounded, err on the side of caution and flag it for review.
- Consider reasonable inferences a physiotherapy expert might make, but be cautious about overly speculative conclusions.
- Your role is to verify, not to generate new diagnoses or medical advice.

Remember your primary objective is to ensure the accuracy and reliability of the differential diagnosis by rigorously comparing it against the provided facts. Identify any potential hallucinations, unsupported claims, or misinterpretations, and provide a clear, detailed report of your findings. Your analysis will be crucial in maintaining the integrity and trustworthiness of the diagnostic process. Format your entire response according to the provided format instructions.
"""

HALLUCINATION_CHECKER_USER_PROMPT = """# Verification Task

## Differential Diagnosis:

{answer}

## Reference Facts:

{facts}

## Task-Specific Instructions:
- Focus on verifying the reasoning behind the primary diagnosis and any alternative diagnoses mentioned.
- Pay special attention to the stated confidence levels and ensure they are justified by the evidence.
- Verify the appropriateness of any recommended tests or treatments.
- Check for any misuse or misinterpretation of physiotherapy-specific terminology.
"""

llm = ChatOpenAI(model=Llm.GPT_4O, temperature=1)


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", HALLUCINATION_GRADER_SYSTEM_PROMPT),
        ("user", HALLUCINATION_CHECKER_USER_PROMPT),
    ]
).partial(format_instructions=parser.get_format_instructions())

hallucination_grader = (
    prompt | llm | JsonOutputParser(pydantic_object=HallucinationGraderOutput)
)
hallucination_grader = hallucination_grader.with_retry()

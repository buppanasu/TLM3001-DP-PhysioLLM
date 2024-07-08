import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_core.output_parsers.json import JsonOutputParser
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from constants import Llm


class HallucinationGraderOutput(BaseModel):
    score: str = Field(
        "",
        description="A binary score 'yes' or 'no' to indicate whether the answer is grounded in / supported by the facts provided.",
    )
    reason: str = Field(
        "",
        description="A reason for the score given, identifying the specific part of the answer that is not grounded in the facts.",
    )


parser = PydanticOutputParser(pydantic_object=HallucinationGraderOutput)

PROMPT = """
<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing whether
an answer is grounded in / supported by as set of facts. 
Your will be grading based on a context of a medical scenerio and the differential diagnosis provided by an expert in physiotherapy.
The retrieved facts are sourced from medical sources such as medical journals, textbooks, and other reliable sources.
This sources might contain examples of different musculoskeletal conditions, their symptoms, and the differential diagnosis for each condition.
The facts might contain example conditions of arbitrary patients, and the differential diagnosis provided by an expert in physiotherapy, hence the facts might not be directly related to the user question.
Therefore You should not focus on the names of the patients or the specific conditions mentioned in the facts, but rather on the general medical knowledge and the differential diagnosis provided by the expert in physiotherapy.
For example if the facts contains conditions for a patient called Mr X, and patient in the user question is called Mr Y, you should not grade the answer as not grounded in the facts just because the names of the patients are different.
Even if the answer contains a condition that is not mentioned in the facts, grade it as grounded if there is valid reference to the condition in the medical literature provided.

{format_instructions}<|eot_id|>
<|start_header_id|>user<|end_header_id|>

Here are the facts:

----------------------------------------
{facts}
----------------------------------------

Here is the answer: {answer}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""

llm = ChatOpenAI(model=Llm.GPT_4O, temperature=1)


prompt = PromptTemplate(
    template=PROMPT,
    input_variables=["facts", "answer"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

hallucination_grader = (
    prompt | llm | JsonOutputParser(pydantic_object=HallucinationGraderOutput)
)
hallucination_grader = hallucination_grader.with_retry()

### Retrieval Grader

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers.json import JsonOutputParser
from langchain.prompts import PromptTemplate
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

RETRIEVAL_GRADER_PROMPT = """
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a grader assessing relevance of a retrieved document to a user question. 
If the document contains keywords related to the user question or provides relevant information to answer the question,
grade it as relevant. It does not need to be a stringent test. The goal is to filter out erroneous retrievals.

{format_instructions}<|eot_id|>
<|start_header_id|>user<|end_header_id|>

Here is the retrieved document:

{document}

Here is the user question: {question}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""

llm = ChatOpenAI(model=Llm.GPT_4O_MINI, temperature=0)
# llm = ChatGroq(model=LLAMA3_80B, temperature=1)
# llm = ChatOllama(model=LOCAL_LLM, format="json", temperature=0)

prompt = PromptTemplate(
    template=RETRIEVAL_GRADER_PROMPT,
    input_variables=["document", "question"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

# Create a retrieval grader by chaining the prompt, llm model, and the json output parser
retrieval_grader = (
    prompt | llm | JsonOutputParser(pydantic_object=RetrievalGraderOutput)
)
retrieval_grader = retrieval_grader.with_retry()

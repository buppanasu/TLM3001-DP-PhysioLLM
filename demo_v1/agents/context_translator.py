# context translator

import pprint
import sys
import os

from langchain_openai import ChatOpenAI

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from langchain_core.output_parsers.json import JsonOutputParser
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain_groq import ChatGroq
from typing import List
from pydantic import BaseModel, Field
from constants import Llm
from dotenv import load_dotenv

load_dotenv()


class ContextDocument(BaseModel):
    thought_process: str = Field(
        ..., description="Thought process of how the document was translated"
    )
    content: str = Field(..., description="Translated content of the document")
    source: str = Field(..., description="Source of the document")


class ContextTranslatorOutput(BaseModel):
    """Outputs a list of translated context documents based on the input query"""

    context_documents: List[ContextDocument] = Field(
        ..., description="List of translated context documents"
    )


parser = PydanticOutputParser(pydantic_object=ContextTranslatorOutput)

CONTEXT_TRANSLATOR_SYSTEM_PROMPT = """
You are a Context Translator Agent for a retrieval-augmented generation system,
your job is to translate and rewrite a list of context documents that were retrieved from a vector database.
These documents contain medical information related to physiotherapy and musculoskeletal conditions.
Your task is to summarize and rephrase the information in these documents to make it more concise and easier to understand.
The goal is to provide a clear and informative summary of the key points in each document.

## Format instructions:
{format_instructions}
"""

CONTEXT_TRANSLATOR_USER_PROMPT = """
## Documents retrieved:
{documents}

Based on the following query, summarise and rewrite the information in the documents above to make it more concise and easier to understand.

Query: {query}
"""

llm = ChatOpenAI(model=Llm.GPT_4O, temperature=0.5)

test_query = "How does a history of occasional low back pain after heavy lifting relate to current symptoms in a 45-year-old male?"
test_documents = """
source:documents/Norkin&White_Joint Motion.pdf
content:
Alaranta et al16Fingertip-to-
Thigh
Lindell et al7Fingertip-to-
Thigh
Jones et al15Fingertip-to-
Floor
Haywood et al66Fingertip-to-
Floor
Pile et al68
Sample 508 employed 
workers*
35–45 yr20 healthy and 
30 patients with 
back or neck pain
22–55 yr89 healthy and
30 children
with LBP
11–16 yrPatients 
with AS 
18–75 yrPatients 
with AS† 
28–73 yr
n = 34 n = 93 n = 20 n =30 n = 89 n = 30 n = 26 n = 51 n = 10
MotionIntra 
RInter 
RIntra 
ICCInter 
ICCIntra 
RIntra 
RIntra 
ICCInter 
ICCInter
Right and 
left0.81
0.91
Right 0.99 0.93 0.99 0.93 0.98 0.98 0.83
Left 0.94 0.95 0.99 0.95 0.95 0.95 0.79
AS = Ankylosing spondylitis; ICC = Intraclass correlation coefﬁ  cient; LBP = Low-back pain; r = Pearson product moment correlation 
coefﬁ  cient; Intra = Intratester reliability; Inter = Intertester reliability.
* Some workers had back or neck pain and some had no pain.
4566_Norkin_Ch12_469-518.indd   5144566_Norkin_Ch12_469-518.indd   514 10/13/16   12:15 PM10/13/16   12:15 PM

---

source:documents/Norkin&White_Joint Motion.pdf
content:
back pain 
20–65 yrUniversal goniometer
Dual inclinometers+Flexion
ExtensionR lateral ﬂ  exion
FlexionExtensionR lateral ﬂ  exion0.920.810.76
0.90
0.700.900.840.630.62
0.52
0.350.18
Williams et al
1715 Children with 
low back painDual inclinometers* Flexion
Extension0.600.48
Kachingwe and 
Phillips
7291 Adults with 
low back pain
Mean 
age = 28 yrBROM* with 2 testers Flexion
ExtensionR lateral ﬂ  exion
R rotationL rotation0.79, 0.840.60, 0.740.84, 0.850.68, 0.760.58, 0.690.740.550.790.600.64
ICC = Intraclass correlation coefﬁ  cient; BROM = Back range of motion device; OSI CA-6000 = Spine Motion Analyzer; R = Right; L = Left.
*Lumbar ROM.
†Thoracolumbar ROM.
4566_Norkin_Ch12_469-518.indd   5114566_Norkin_Ch12_469-518.indd   511 10/13/16   12:15 PM10/13/16   12:15 PM
"""

prompt = ChatPromptTemplate.from_messages(
    messages=[
        (
            "system",
            CONTEXT_TRANSLATOR_SYSTEM_PROMPT,
        ),
        (
            "human",
            CONTEXT_TRANSLATOR_USER_PROMPT,
        ),
    ],
).partial(
    format_instructions=parser.get_format_instructions(),
)

context_translator = (
    prompt | llm | JsonOutputParser(pydantic_object=ContextTranslatorOutput)
)
context_translator = context_translator.with_retry()


def main():
    res = context_translator.invoke({"documents": test_documents, "query": test_query})

    print("Query:", test_query, "\n\n")

    for i, doc in enumerate(res["context_documents"]):
        print(f"Translated Document {i+1}:")
        print(doc)
        print("\n")


if __name__ == "__main__":
    main()

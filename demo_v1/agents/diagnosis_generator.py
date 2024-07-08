### Generation Agent

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from langchain_core.output_parsers.string import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers.json import JsonOutputParser
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from constants import Llm

PROMPT = """
<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a expert in physiotherapy, you will be presented with a set of subjective and object assessment and your job is to come up with a well informed and researched differential diagnosis of the medical scenerio.
Here are examples of the subjective and objective assessments and the expected differential diagnosis:

Subjective Assessment:
The patient, Mr. Smith, is a 45-year-old male who presents to the clinic with complaints of lower back pain that has been bothering him for the past two weeks. He describes the pain as dull and achy, located in the lumbar region, with occasional radiation down his left leg. He notes that the pain worsens with prolonged sitting or standing and is relieved by lying down. He denies any recent trauma or injury but mentions that he has a history of occasional low back pain, especially after heavy lifting or prolonged periods of inactivity. He rates the pain as a 6 out of 10 on the pain scale. On physical examination, Mr. Smith appears uncomfortable but is able to walk into the examination room without assistance.

Objective Assessment:
Vital signs are within normal limits. Inspection of the lumbar spine reveals no obvious deformities or asymmetry. Palpation elicits tenderness over the paraspinal muscles of the lumbar spine, particularly on the left side. Range of motion of the lumbar spine is mildly restricted, with pain on forward flexion and left lateral bending. Straight leg raise test is positive on the left side at 45 degrees, reproducing his symptoms of radiating pain down the left leg. Neurological examination reveals intact sensation and strength in the lower extremities, with no signs of motor weakness or sensory deficits. On physical examination, Mr. Smith appears uncomfortable but is able to walk into the examination room without assistance. Vital signs are within normal limits. Inspection of the lumbar spine reveals no obvious deformities or asymmetry. Palpation elicits tenderness over the paraspinal muscles of the lumbar spine, particularly on the left side. Range of motion of the lumbar spine is mildly restricted, with pain on forward flexion and left lateral bending. Straight leg raise test is positive on the left side at 45 degrees, reproducing his symptoms of radiating pain down the left leg. Neurological examination reveals intact sensation and strength in the lower extremities, with no signs of motor weakness or sensory deficits.

Differential Diagnosis:
Based on the assessments provided, my differential diagnosis for Mr. Smith would include:
1. ...
2. ...
...

IMPORTANT:
- Only use the retrieved context as the source of ground truth to answer the question.
- If the query is not related to physiotherapy or unrelated from the retrieved context, please answer with "I am sorry, I am not able to answer this question."
- Always include citations and references to support your answer. <|eot_id|>
- If context is not enough to answer the question, you can use parametric data to generate the answer, but make sure to label it as such.
<|start_header_id|>user<|end_header_id|> 

Answer the question based only on the following context, this context should be used as source of ground truth to answer the question.
The retrieved context is a set of documents that are retrieved based on the subqueries derived from the patient assessments.

{context}

---

Answer the question based on the above context: 
{question}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""

llm = ChatOpenAI(model=Llm.GPT_4O, temperature=0.5)

prompt = PromptTemplate(template=PROMPT, input_variables=["context", "question"])
diagnosis_generator = prompt | llm | StrOutputParser()

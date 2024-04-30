import argparse
from os import system
import ollama
from qdrant_client import QdrantClient
from vector_embedding import get_vector_embeddings
from langchain.prompts import ChatPromptTemplate
from langchain_community.vectorstores.qdrant import Qdrant

EMBEDDING_MODEL = "NeuML/pubmedbert-base-embeddings"
VECTOR_DB_URL = "http://localhost:6333"
SYSTEM_PROMPT = """
You are a expert in physiotherapy, you will be presented with a set of subjective and object assessment and your job is to come up with a well informed and researched differential diagnosis of the medical scenerio.

Here are examples of the subjective and objective assessments:

Subjective Assessment:
The patient, Mr. Smith, is a 45-year-old male who presents to the clinic with complaints of lower back pain that has been bothering him for the past two weeks. He describes the pain as dull and achy, located in the lumbar region, with occasional radiation down his left leg. He notes that the pain worsens with prolonged sitting or standing and is relieved by lying down. He denies any recent trauma or injury but mentions that he has a history of occasional low back pain, especially after heavy lifting or prolonged periods of inactivity. He rates the pain as a 6 out of 10 on the pain scale.On physical examination, Mr. Smith appears uncomfortable but is able to walk into the examination room without assistance.

Objective Assessment:
Vital signs are within normal limits. Inspection of the lumbar spine reveals no obvious deformities or asymmetry. Palpation elicits tenderness over the paraspinal muscles of the lumbar spine, particularly on the left side. Range of motion of the lumbar spine is mildly restricted, with pain on forward flexion and left lateral bending. Straight leg raise test is positive on the left side at 45 degrees, reproducing his symptoms of radiating pain down the left leg. Neurological examination reveals intact sensation and strength in the lower extremities, with no signs of motor weakness or sensory deficits. On physical examination, Mr. Smith appears uncomfortable but is able to walk into the examination room without assistance. Vital signs are within normal limits. Inspection of the lumbar spine reveals no obvious deformities or asymmetry. Palpation elicits tenderness over the paraspinal muscles of the lumbar spine, particularly on the left side. Range of motion of the lumbar spine is mildly restricted, with pain on forward flexion and left lateral bending. Straight leg raise test is positive on the left side at 45 degrees, reproducing his symptoms of radiating pain down the left leg. Neurological examination reveals intact sensation and strength in the lower extremities, with no signs of motor weakness or sensory deficits.

Differential Diagnosis:
Based on the assessments provided, my differential diagnosis for Mr. Smith would include:
1. Lumbar herniated disc (L4-L5 or L5-S1)
2. Degenerative disc disease (DDD) in the lumbar spine
3. Lumbar spondylosis with radiculopathy (radiating pain down the left leg)
4. Piriformis syndrome, causing compression of the sciatic nerve


IMPORTANT:
- If the query is not related to physiotherapy or unrelated from the retrieved context, please answer with "I am sorry, I am not able to answer this question."
- Include citations and references to support your answer.

Think step by step and provide a well informed and researched answer.
"""
PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""


def parse_arguments() -> str:
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    return args.query_text


def main():
    embeddings = get_vector_embeddings(EMBEDDING_MODEL)
    client = QdrantClient(VECTOR_DB_URL)
    db = Qdrant(
        client=client,
        embeddings=embeddings,
        collection_name="test-collection",
    )

    # Get query and perform similarity search
    query = parse_arguments()
    docs = db.similarity_search_with_score(query=query, k=3)

    # Add the query to the context
    retrieved_docs = []
    for doc, score in docs:
        retrieved_docs.append(
            f"chunk:{doc.page_content}\nmetadata:{doc.metadata}\nsimilarity_score:{score}"
        )
    context = "\n\n---\n\n".join(retrieved_docs)

    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(question=query, context=context)

    print("Here is the generated prompt:")
    print("*" * 50, end="\n\n")
    print(prompt)

    stream = ollama.chat(
        model="llama3",
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {"role": "user", "content": prompt},
        ],
        stream=True,
    )

    for chunk in stream:
        print(chunk["message"]["content"], end="", flush=True)


if __name__ == "__main__":
    main()

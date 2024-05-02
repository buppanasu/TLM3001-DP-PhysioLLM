import argparse
from datetime import datetime
import os
import ollama
from ollama import Message
from qdrant_client import QdrantClient
from vector_embedding import get_vector_embeddings
from langchain.prompts import ChatPromptTemplate
from langchain_community.vectorstores.qdrant import Qdrant

EMBEDDING_MODEL = "NeuML/pubmedbert-base-embeddings"
VECTOR_DB_URL = "http://localhost:6333"

SYSTEM_PROMPT = """
You are a expert in physiotherapy, you will be presented with a set of subjective and object assessment and your job is to come up with a well informed and researched differential diagnosis of the medical scenerio.

Here are examples of the subjective and objective assessments and the expected differential diagnosis:

Subjective Assessment:
The patient, Mr. Smith, is a 45-year-old male who presents to the clinic with complaints of lower back pain that has been bothering him for the past two weeks. He describes the pain as dull and achy, located in the lumbar region, with occasional radiation down his left leg. He notes that the pain worsens with prolonged sitting or standing and is relieved by lying down. He denies any recent trauma or injury but mentions that he has a history of occasional low back pain, especially after heavy lifting or prolonged periods of inactivity. He rates the pain as a 6 out of 10 on the pain scale.On physical examination, Mr. Smith appears uncomfortable but is able to walk into the examination room without assistance.

Objective Assessment:
Vital signs are within normal limits. Inspection of the lumbar spine reveals no obvious deformities or asymmetry. Palpation elicits tenderness over the paraspinal muscles of the lumbar spine, particularly on the left side. Range of motion of the lumbar spine is mildly restricted, with pain on forward flexion and left lateral bending. Straight leg raise test is positive on the left side at 45 degrees, reproducing his symptoms of radiating pain down the left leg. Neurological examination reveals intact sensation and strength in the lower extremities, with no signs of motor weakness or sensory deficits. On physical examination, Mr. Smith appears uncomfortable but is able to walk into the examination room without assistance. Vital signs are within normal limits. Inspection of the lumbar spine reveals no obvious deformities or asymmetry. Palpation elicits tenderness over the paraspinal muscles of the lumbar spine, particularly on the left side. Range of motion of the lumbar spine is mildly restricted, with pain on forward flexion and left lateral bending. Straight leg raise test is positive on the left side at 45 degrees, reproducing his symptoms of radiating pain down the left leg. Neurological examination reveals intact sensation and strength in the lower extremities, with no signs of motor weakness or sensory deficits.

Differential Diagnosis:
Based on the assessments provided, my differential diagnosis for Mr. Smith would include:
1. ...
2. ...
...


IMPORTANT:
- If the query is not related to physiotherapy or unrelated from the retrieved context, please answer with "I am sorry, I am not able to answer this question."
- Include citations and references to support your answer.

Think step by step and provide a well informed and researched answer.
"""

PROMPT_TEMPLATE = """
Answer the question based only on the following context, this context should be used as source of ground truth to answer the question:

{context}

---

Answer the question based on the above context, include in the answer the context that supports your answer: 

{question}


"""


def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate a response using LLM")
    parser.add_argument(
        "-q",
        "--query",
        type=str,
        help="The query to generate a response for",
    )
    args = parser.parse_args()
    return args.query


def generate_initial_prompt(db: Qdrant) -> str:
    # Get query and perform similarity search
    query = parse_arguments()
    if not query:
        with open("test_query.txt", "r") as file:
            query = file.read()
    docs = db.similarity_search_with_score(query=query, k=5)

    # Add the query to the context
    retrieved_docs = []
    for doc, score in docs:
        retrieved_docs.append(
            f"metadata:{doc.metadata}\nsimilarity_score:{score}\n\nchunk:{doc.page_content}"
        )
    context = "\n\n---\n\n".join(retrieved_docs)

    primer = "Anaylsis the following subjective and objective assessments and provide a well informed and researched differential diagnosis.\n\n"
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(question=primer + query, context=context)
    return prompt


def main():
    # Get vector embeddings and create Qdrant client to interact with the vector database
    print("Loading vector embeddings and creating Qdrant client...")
    embeddings = get_vector_embeddings(EMBEDDING_MODEL)
    client = QdrantClient(VECTOR_DB_URL)
    db = Qdrant(
        client=client,
        embeddings=embeddings,
        collection_name="test-collection",
    )

    # Generate initial prompt to start the conversation
    initial_prompt = generate_initial_prompt(db)
    messages = [
        Message(
            role="system",
            content=SYSTEM_PROMPT,
        ),
        Message(
            role="user",
            content=initial_prompt,
        ),
    ]

    # Start conversation with LLM
    while True:
        print("\nAssistant reply:\n")

        # Stream response from LLM
        stream = ollama.chat(model="llama3", messages=messages, stream=True)

        # Create response from stream
        response = ""
        for chunk in stream:
            if isinstance(chunk, str):
                continue
            text_chunk = chunk["message"]["content"]
            response += text_chunk
            print(text_chunk, end="", flush=True)

        # Add it to message history
        messages.append(
            {
                "role": "assistant",
                "content": response,
            }
        )

        # Get user input and add it to message history to continue the conversation
        query = input("\n\nUser prompt: ")
        if query == "-1":
            print("Exiting conversation")
            break

        messages.append(
            {
                "role": "user",
                "content": query,
            }
        )

    # Generate chat history from messages
    texts = []
    for message in messages:
        text = f"{message['role']}:\n"
        text += message["content"]
        texts.append(text)
    history = "\n\n---\n\n".join(texts)

    # Save chat history to a text file
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    file_name = f"chat_history-{timestamp}.txt"
    path = os.path.join(os.getcwd(), "llm_logs", file_name)
    with open(path, "w") as file:
        file.write(history)


if __name__ == "__main__":
    main()

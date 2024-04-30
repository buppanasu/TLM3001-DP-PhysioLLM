import argparse
from os import system
import ollama
from qdrant_client import QdrantClient
from vector_embedding import get_vector_embeddings
from langchain.prompts import ChatPromptTemplate
from langchain_community.vectorstores.qdrant import Qdrant

EMBEDDING_MODEL = "NeuML/pubmedbert-base-embeddings"
VECTOR_DB_URL = "http://localhost:6333"

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
    context = ""
    for doc, score in docs:
        context += "\n\n---\n\n"
        context += doc.page_content

    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(question=query, context=context)

    print("Here is the generated prompt:", prompt, end="\n\n")

    system_prompt = """
    You are a expert in physiotherapy, answer the following questions based on the context in a professional manner and with proper medical terms. Always cite your sources.
    """
    stream = ollama.chat(
        model="llama3",
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {"role": "user", "content": prompt},
        ],
        stream=True,
    )

    for chunk in stream:
        print(chunk["message"]["content"], end="", flush=True)


if __name__ == "__main__":
    main()

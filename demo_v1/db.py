from langchain_qdrant import Qdrant
from qdrant_client import QdrantClient
from langchain_huggingface import HuggingFaceEmbeddings
from constants import VectorDb


def get_vector_embeddings(embedding_model: str):
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    return embeddings


def get_qdrant_client():
    embeddings = get_vector_embeddings(VectorDb.EMBEDDING_MODEL)
    client = QdrantClient(VectorDb.VECTOR_DB_URL)
    return Qdrant(
        client=client,
        embeddings=embeddings,
        collection_name="physio-textbooks",
    )

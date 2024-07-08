from langchain_qdrant import Qdrant
from qdrant_client import QdrantClient
from langchain_huggingface import HuggingFaceEmbeddings
from constants import VectorDb


def get_vector_embeddings(embedding_model: str):
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    return embeddings


print("Loading vector embeddings and creating Qdrant client...")
embeddings = get_vector_embeddings(VectorDb.EMBEDDING_MODEL)
client = QdrantClient(VectorDb.VECTOR_DB_URL)
db = Qdrant(
    client=client,
    embeddings=embeddings,
    collection_name="physio-textbooks",
)

import os
from dotenv import load_dotenv
load_dotenv()
MAX_FILE_SIZE_MB=50  # Max file size in MB
OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")
QDRANT_ENDPOINT=os.getenv("QDRANT_ENDPOINT")
QDRANT_COLLECTION_NAME=os.getenv("QDRANT_COLLECTION_NAME")
LLM_MODEL_FOR_GENERATION="gpt-4o-mini"
VECTOR_EMBEDDING_MODEL="NeuML/pubmedbert-base-embeddings"
DOCUMENT_DIR = "documents"
MARKDOWN_DIR = "./markdown_files"
JSON_DIR = "./json_files"
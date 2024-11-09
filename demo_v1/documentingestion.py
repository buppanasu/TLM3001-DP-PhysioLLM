# Filename: fastapi_app.py

from fastapi import FastAPI, UploadFile
from typing import List
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import qdrant
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings

# FastAPI app
app = FastAPI()

# Qdrant setup
VECTOR_DB_URL = "http://localhost:6333"
COLLECTION_NAME = "physio-textbooks"

# Directory to temporarily store uploaded files
UPLOAD_DIR = "./uploaded_documents"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/ingest-documents/")
async def ingest_documents(files: List[UploadFile]):
    results = {"success": [], "errors": []}
    
    # Initialize embeddings model
    embeddings = HuggingFaceEmbeddings(model_name="NeuML/pubmedbert-base-embeddings")

    for file in files:
        try:
            # Save uploaded file temporarily
            save_path = os.path.join(UPLOAD_DIR, file.filename)
            with open(save_path, "wb") as f:
                f.write(await file.read())
            
            # Load and split document into chunks
            loader = PyPDFLoader(save_path)
            documents = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
            chunks = text_splitter.split_documents(documents)
            
            # Embed and store chunks in Qdrant
            qdrant.Qdrant.from_documents(
                chunks,
                embeddings,
                url=VECTOR_DB_URL,
                collection_name=COLLECTION_NAME,
                prefer_grpc=False,
            )
            
            # If successful, add to results
            results["success"].append(file.filename)

            # Optional: remove the file after processing
            os.remove(save_path)
        
        except Exception as e:
            # Log errors for each file
            results["errors"].append({"file": file.filename, "error": str(e)})
    
    # Determine overall success
    overall_success = bool(results["success"])
    return {"success": overall_success, "results": results}

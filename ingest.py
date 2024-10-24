from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import qdrant
from langchain_community.document_loaders.directory import DirectoryLoader
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
import re


def main():
    vector_db_url = "http://localhost:6333"

    print("Ingesting documents...")
    loader = DirectoryLoader(
        "documents", glob="*.pdf", show_progress=True, loader_cls=PyPDFLoader  # type: ignore
    )
    documents = loader.load()
    print(f"Loaded {len(documents)} documents")

    # Split the documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)

    for chunk in chunks:
        # Add web source into metadata
        source = chunk.metadata["source"]
        webSource = re.findall(r".+_(.*?)\..+", source)

        if webSource:
            chunk.metadata["WebSource"] = webSource[0]
        else:
            chunk.metadata["WebSource"] = "-"

    # Embed the chunks
    embeddings = HuggingFaceEmbeddings(model_name="NeuML/pubmedbert-base-embeddings")
    print("Embedding chunks...")
    qdrant.Qdrant.from_documents(
        chunks,
        embeddings,
        url=vector_db_url,
        collection_name="physio-textbooks",
        prefer_grpc=False,
    )


if __name__ == "__main__":
    main()

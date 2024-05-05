from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import qdrant
from langchain_community.document_loaders.directory import DirectoryLoader
from langchain_community.document_loaders.pdf import PyPDFLoader
from vector_embedding import get_vector_embeddings


def main():
    vector_db_url = "http://localhost:6333"

    print("Ingesting documents...")
    loader = DirectoryLoader(
        "documents", glob="*.pdf", show_progress=True, loader_cls=PyPDFLoader
    )
    documents = loader.load()
    print(f"Loaded {len(documents)} documents")

    # Split the documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)
    for chunk in chunks:
        print("Splitted chunk", chunk.metadata)

    # Embed the chunks
    embeddings = get_vector_embeddings("NeuML/pubmedbert-base-embeddings")
    print("Embedding chunks...")
    qdrant.Qdrant.from_documents(
        chunks,
        embeddings,
        url=vector_db_url,
        collection_name="test-collection",
        prefer_grpc=True,
    )


if __name__ == "__main__":
    main()

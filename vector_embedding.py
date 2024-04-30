from langchain_community.embeddings import sentence_transformer


def get_vector_embeddings(embedding_model: str):
    embeddings = sentence_transformer.SentenceTransformerEmbeddings(
        model_name="NeuML/pubmedbert-base-embeddings"
    )
    return embeddings

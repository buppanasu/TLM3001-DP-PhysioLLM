
from langchain_qdrant import QdrantVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains import RetrievalQA
from langchain_qdrant import FastEmbedSparse, RetrievalMode


# Create the vector store and fill it with embeddings
qdrant = QdrantVectorStore.from_existing_collection(
    embedding=HuggingFaceEmbeddings(model_name="NeuML/pubmedbert-base-embeddings"),
    #sparse_embedding = sparse_embeddings,
    collection_name="physio-textbooks",
    url="http://localhost:6333",
    retrieval_mode=RetrievalMode.DENSE,
    #retrieval_mode = RetrievalMode.HYBRID,
)

retriever = qdrant.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Define llm
llm = ChatOpenAI(model="gpt-4o-mini")

# Define the prompt, making sure to include "context" and "question" as input variables
QA_CHAIN_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""
    Context: The user is learning about musculoskeletal and physiotherapy issues.

    Your role is to help out by providing relevant, concise and accurate information.

    Outcome: The goal is for the user to properly understand the response with regard to the question they asked.



    Use only the knowledge and context supplied below here, as context:
    {context}

    Question: {question}

    Provide a helpful answer in 3-4 sentences. 
    """
)

llm_chain = LLMChain(
    llm=llm,
    prompt=QA_CHAIN_PROMPT,
    verbose=True
)

document_prompt = PromptTemplate(
    input_variables=["page_content", "source"],
    template="Context:\ncontent:{page_content}\nsource:{source}",
)

combine_documents_chain = StuffDocumentsChain(
    llm_chain=llm_chain,
    document_variable_name="context",  # This must match the input variable in the QA_CHAIN_PROMPT
    document_prompt=document_prompt,
)

qa = RetrievalQA(
    combine_documents_chain=combine_documents_chain,
    retriever=retriever,
    return_source_documents=True,
    verbose=True
)

# Define the response function
def respond(question):
    return qa({"query": question})["result"]

# Example usage
if __name__ == "__main__":
    question = "Tell me about non-specific low back pain"
    response = respond(question)
    print("Response:", response)

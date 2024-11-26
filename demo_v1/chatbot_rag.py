"""
This file handles the retrieval and generation portion of the PhysioBot RAG system.
It provides details on the retriever, the LLM being used, and the prompt used for generation

"""
from langchain_qdrant import QdrantVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains import RetrievalQA
from langchain_qdrant import FastEmbedSparse, RetrievalMode
from config import *

# Step 1: Set up the Qdrant Vector Store
# Create the vector store and fill it with embeddings from an existing collection
qdrant = QdrantVectorStore.from_existing_collection(
    embedding=HuggingFaceEmbeddings(model_name=VECTOR_EMBEDDING_MODEL),
    collection_name=QDRANT_COLLECTION_NAME,
    url=QDRANT_ENDPOINT,
    retrieval_mode=RetrievalMode.DENSE,
)

# Step 2: Define the retriever
# Use the Qdrant vector store to retrieve the top k similar documents
retriever = qdrant.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Step 3: Define the Language Model (LLM)
# Specify the LLM to use, here ChatGPT's mini variant is used
llm = ChatOpenAI(model=LLM_MODEL_FOR_GENERATION)

# Step 4: Define the QA prompt template
# Create a prompt template for the QA chain, ensuring the inputs are "context" and "question"
QA_CHAIN_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""
    Context: 

    The user is a physiotherapist that requires credible information when it comes to physiotherapy.
    
    Use only the knowledge and context supplied below here, as context:
    {context}

    Question: {question}






    """
)

# Step 5: Create the LLM Chain
# Chain the LLM with the QA prompt template
llm_chain = LLMChain(
    llm=llm,
    prompt=QA_CHAIN_PROMPT,
    verbose=True
)

# Step 6: Define the document formatting prompt
# This specifies how individual documents are formatted before being passed to the LLM
document_prompt = PromptTemplate(
    input_variables=["page_content", "source"],
    template="Context:\ncontent:{page_content}\nsource:{source}",
)

# Step 7: Combine documents into a single context
# Use the LLM chain and document prompt to combine documents
combine_documents_chain = StuffDocumentsChain(
    llm_chain=llm_chain,
    document_variable_name="context",  # This must match the input variable in the QA_CHAIN_PROMPT
    document_prompt=document_prompt,
)

# Step 8: Define the Retrieval QA chain
# Use the retriever and combined document chain to create the QA pipeline
qa = RetrievalQA(
    combine_documents_chain=combine_documents_chain,
    retriever=retriever,
    return_source_documents=True,
    verbose=True
)

# Step 9: Define the response function
# This function takes a user question and retrieves an answer from the QA pipeline
def respond(question):
    return qa({"query": question})["result"]

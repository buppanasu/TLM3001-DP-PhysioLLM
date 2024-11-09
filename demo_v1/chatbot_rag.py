
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

    Your role is to help out by providing a treatmwent plan that caters towards the patient's specific condition
    
    Use only the knowledge and context supplied below here, as context:
    {context}

    Question: {question}

    Provide a treatment plan for the user.

    Example 1 of a treatment plan:

    Early Stage (Weeks 0-6): Passive Motion Exercises
    For ages 20-60; exercises focus on flexibility and preventing stiffness without stressing the rotator cuff.

    Pendulum Swings

    How: Lean forward, support yourself with your non-injured arm on a chair, and let the affected arm hang down.
    Movement: Gently swing the arm in small circles (10 in each direction).
    Sets/Reps: 1 set, 10-15 seconds per direction.
    Frequency: 3 times daily.
    Assisted Shoulder Flexion (Using a Wand or Stick)

    How: Lie on your back, hold a stick with both hands, and gently lift it over your head.
    Movement: Use your non-injured arm to guide your injured arm in lifting overhead.
    Sets/Reps: 3 sets of 10 reps.
    Frequency: Once daily.
    Passive External Rotation with Wand

    How: Sit or stand holding a stick horizontally in front of you, with elbows at a 90° angle.
    Movement: Gently push with your non-injured arm to rotate the injured arm outward.
    Sets/Reps: 3 sets of 10 reps.
    Frequency: Once daily.

    Mid-Stage (Weeks 6-12): Active Assisted Motion & Light Strengthening
    For ages 20-60; progress toward active motion and light resistance.

    Active Assisted Shoulder Abduction (Using Wall Support)

    How: Stand next to a wall, and use your fingers to walk up the wall sideways until reaching shoulder height.
    Sets/Reps: 3 sets of 10 reps.
    Frequency: 3 times weekly.
    Scapular Retractions

    How: Sit or stand with your arms by your side.
    Movement: Pinch shoulder blades together gently, holding for 5 seconds.
    Sets/Reps: 3 sets of 10 reps.
    Frequency: 3 times weekly.
    External Rotation with Resistance Band

    How: Anchor a resistance band at elbow height, and hold the other end with your injured arm, keeping your elbow close to your body.
    Movement: Rotate your forearm outward against the band.
    Sets/Reps: 2-3 sets of 8-10 reps.
    Frequency: 2-3 times weekly.
    Internal Rotation with Resistance Band

    How: Anchor a resistance band at elbow height, and hold it with your injured arm, keeping your elbow close to your body.
    Movement: Rotate your forearm inward against the resistance band.
    Sets/Reps: 2-3 sets of 8-10 reps.
    Frequency: 2-3 times weekly.

    Age-Related Modifications
    Older Adults (60+): Start with lighter resistance, fewer reps (6-8 reps), and focus on form to prevent overloading.
    Younger Adults (20-40): Increase reps to 12-15 as strength builds and consider heavier weights as tolerated in later stages.
    
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

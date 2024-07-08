from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import StateGraph, END
from pprint import pprint
from typing import List, Literal
from typing_extensions import TypedDict
from zmq import has
from agents.query_translator import query_translator, QueryTranslatorOutput
from agents.retrieval_grader import retrieval_grader, RetrievalGraderOutput
from agents.diagnosis_generator import diagnosis_generator
from agents.halluncination_grader import hallucination_grader, HallucinationGraderOutput
from db import db
import asyncio

### Langgraph State


class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        web_search: whether to add search
        documents: list of documents
    """

    main_query: str
    subqueries: List[str]
    generation: str
    web_search: str
    documents: List[dict]
    has_hallucinations: bool
    halluncination_check_balance: int


# * Nodes


### Node - Translates main query into subqueries
def translate_query(graph_state: GraphState) -> GraphState:
    """Translates the main query into multiple subqueries"""

    print("--- TRANSLATING MAIN QUERY INTO SUBQUERIES ---")

    # Get the main query
    main_query = graph_state["main_query"]

    # Translate the main query into subqueries
    translator_result = query_translator.invoke(
        {"main_query": main_query}, {"run_name": "query-translator"}
    )
    parsed_translator_result = QueryTranslatorOutput(**translator_result)

    return {**graph_state, "subqueries": parsed_translator_result.subqueries}


### Node - conduct retrieval from vector database using subqueries
async def retrieve(graph_state: GraphState) -> GraphState:
    """Retrieves documents from a vector database using the subqueries"""

    print("--- RETRIEVING DOCUMENTS FROM VECTOR DATABASE FOR EACH SUBQUERY ---")

    # Get the subqueries
    subqueries = graph_state["subqueries"]

    # Retrieve documents from a vector database using the subqueries
    documents = []
    for subquery in subqueries:
        query_result = {"question": subquery, "documents": []}
        results = db.similarity_search_with_score(subquery, k=3)

        for doc, score in results:
            query_result["documents"].append(doc.page_content)

        documents.append(query_result)

    return {
        **graph_state,
        "documents": documents,
    }


### Node - grade the retrieved documents
async def grade_documents(graph_state: GraphState) -> GraphState:
    """Grades the retrieved documents based on relevance"""

    print("--- GRADING RETRIEVED DOCUMENTS ---")

    # Get the ungraded documents and web search flag
    documents = graph_state["documents"]
    web_search = "No"

    async def grade_query_document_pair(query, document, query_index, document_index):
        llm_result = await retrieval_grader.ainvoke(
            {"document": document, "question": query}
        )
        parsed_llm_result = RetrievalGraderOutput(**llm_result)

        return {
            "query_index": query_index,
            "document_index": document_index,
            "is_relevant": parsed_llm_result.score == "yes",
        }

    # Create a list of coroutines for grading each query-document pair
    invocations = []
    for i, item in enumerate(documents):
        query = item["question"]
        docs = item["documents"]

        for j, doc in enumerate(docs):
            invocations.append(grade_query_document_pair(query, doc, i, j))

    grading_results = await asyncio.gather(*invocations)

    # Create a list of filtered documents based on the grading results
    filtered_documents = [
        {"question": item["question"], "documents": []} for item in documents
    ]
    for result in grading_results:
        query_index = result["query_index"]
        document_index = result["document_index"]
        is_relevant = result["is_relevant"]

        if is_relevant:
            filtered_documents[query_index]["documents"].append(
                documents[query_index]["documents"][document_index]
            )

    # check for query with no relevant documents
    for item in filtered_documents:
        if not item["documents"]:
            web_search = "Yes"
            break

    return {**graph_state, "documents": filtered_documents, "web_search": web_search}


### Node - conduct web search for subqueries with no relevant documents
async def web_search(graph_state: GraphState) -> GraphState:
    """Conducts web search for subqueries with no relevant documents"""

    print("--- CONDUCTING WEB SEARCH FOR SUBQUERIES WITH NO RELEVANT DOCUMENTS ---")

    web_search_tool = TavilySearchResults(max_results=3)
    documents = graph_state["documents"]

    async def query_web_search(query, query_index):
        web_results = await web_search_tool.ainvoke(query)
        return {"query_index": query_index, "documents": web_results}

    # Web search for each query that did not have relevant documents
    invocations = []
    for i, item in enumerate(documents):
        # Skip web search for queries that had relevant documents
        if item["documents"]:
            continue

        # Perform web search for the query with no relevant documents
        query = item["question"]
        invocations.append(query_web_search(query, i))

    # Gather the results of the web search
    results = await asyncio.gather(*invocations)

    for result in results:
        query_index = result["query_index"]
        docs = result["documents"]

        documents[query_index]["documents"] = [doc["content"] for doc in docs]

    return {**graph_state, "documents": documents}


### Node - Generate differential diagnosis based on the retrieved documents
async def generate(graph_state: GraphState) -> GraphState:
    print("---GENERATE---")
    question = graph_state["main_query"]
    documents = graph_state["documents"]

    formatted_documents = []
    for item in documents:
        docs = item["documents"]
        query_str = item["question"]
        docs_str = "\n\n---\n\n".join(docs)
        text = f"Subquery:\n{query_str}\n\nDocuments:\n{docs_str}"
        formatted_documents.append(text)

    formatted_context = "\n\n***\n\n".join(formatted_documents)

    # RAG generation
    generation_result = diagnosis_generator.invoke(
        {"context": formatted_context, "question": question}
    )

    return {
        **graph_state,
        "generation": generation_result,
    }


### Node - Check the generated differential diagnosis for hallucinations
def check_hallucinations(graph_state: GraphState) -> GraphState:
    """Check the generated differential diagnosis for hallucinations"""

    print("--- CHECKING IF GENERATION IS GROUNDED IN THE DOCUMENTS ---")

    generation = graph_state["generation"]
    documents = graph_state["documents"]
    hallucination_count_balance = graph_state["halluncination_check_balance"]

    # Run hallucination grader on the generated differential diagnosis
    hallucination_result = hallucination_grader.invoke(
        {"facts": documents, "answer": generation}, {"run_name": "hallucination-grader"}
    )
    parsed_hallucination_result = HallucinationGraderOutput(**hallucination_result)

    if parsed_hallucination_result.score == "yes":
        has_hallucinations = False
        print("No hallucinations detected in the generated differential diagnosis")
    else:
        has_hallucinations = True
        print(
            "Hallucinations detected in the generated differential diagnosis",
            parsed_hallucination_result.reason,
        )

    return {
        **graph_state,
        "has_hallucinations": has_hallucinations,
        "halluncination_check_balance": hallucination_count_balance - 1,
    }


# * Edges


### Conditional edge - Determines whether to go to web search or generation based on the web_search flag
def decide_to_generate_or_websearch(
    graph_state: GraphState,
) -> Literal["generate", "websearch"]:
    """Decides whether to generate differential diagnosis or conduct web search"""

    print("---ASSESS GRADED DOCUMENTS---")
    web_search = graph_state["web_search"]

    if web_search == "Yes":
        print(
            "---DECISION: THERE ARE SUBQUERIES WITH NO RELEVANT DOCUMENTS, CONDUCT WEB SEARCH---"
        )
        return "websearch"
    else:
        print("---DECISION: GENERATE DIFFERENTIAL DIAGNOSIS---")
        return "generate"


### Conditional edge - Determines whether to go end the process or check for hallucinations based on the hallucination_check_balance
def decide_to_check_hallucinations_or_end(
    graph_state: GraphState,
) -> Literal["retry", "end"]:
    """Decides whether to check for hallucinations or end the process"""

    print("---ASSESS HALLUCINATIONS---")
    has_hallucinations = graph_state["has_hallucinations"]
    hallucination_check_balance = graph_state["halluncination_check_balance"]

    if has_hallucinations and hallucination_check_balance > 0:
        print("---DECISION: HALLUCINATIONS DETECTED, RETRY GENERATION---")
        return "retry"

    if hallucination_check_balance == 0:
        print(
            "---DECISION: HALLUCINATIONS DETECTED BUT RETRY LIMIT REACHED, END PROCESS---"
        )
        return "end"

    print("---DECISION: NO HALLUCINATIONS DETECTED, END PROCESS---")
    return "end"


async def test_nodes():
    with open(
        "/Users/dingruoqian/code/TLM3001-DP-PhysioLLM/test_prompts/query_1.txt", "r"
    ) as file:
        main_query = file.read()

    graph_state = GraphState(
        {
            "main_query": main_query,
            "subqueries": [],
            "documents": [],
            "web_search": "No",
            "generation": "",
            "has_hallucinations": False,
            "halluncination_check_balance": 3,
        }
    )

    # Translate the main query into subqueries
    graph_state = translate_query(graph_state)

    # Retrieve documents from a vector database using the subqueries
    graph_state = await retrieve(graph_state)

    # Grade the retrieved documents based on relevance
    graph_state = await grade_documents(graph_state)

    # Check if there are any subqueries with no relevant documents
    for item in graph_state["documents"]:
        if not item["documents"]:
            print(f"Subquery - {item['question']} has no relevant documents")

    # Conduct web search for subqueries with no relevant documents
    graph_state = await web_search(graph_state)

    # Check if there are any subqueries with no relevant documents
    for item in graph_state["documents"]:
        if not item["documents"]:
            raise ValueError("Some subqueries have no relevant documents.")
    else:
        print("All subqueries have relevant documents.")

    # Generate differential diagnosis based on the retrieved documents
    graph_state = await generate(graph_state)
    pprint(graph_state["generation"])

    # Check the generated differential diagnosis for hallucinations
    graph_state = check_hallucinations(graph_state)


def construct_graph() -> StateGraph:
    ### Create Graph and add nodes

    workflow = StateGraph(GraphState)

    # Define the nodes
    workflow.add_node("translate_query", translate_query)  # translate query
    workflow.add_node("retrieve", retrieve)  # retrieve
    workflow.add_node("websearch", web_search)  # web search
    workflow.add_node("generate", generate)  # generate
    workflow.add_node("grade_documents", grade_documents)  # grade documents
    workflow.add_node(
        "check_hallucinations", check_hallucinations
    )  # check hallucinations

    # Build the graph
    workflow.set_entry_point("translate_query")  # entry point
    workflow.add_edge("translate_query", "retrieve")  # translate query -> retrieve
    workflow.add_edge("retrieve", "grade_documents")  # retrieve -> grade documents
    workflow.add_conditional_edges(  # grade documents -> decide to generate or websearch
        "grade_documents",
        decide_to_generate_or_websearch,
        {
            "generate": "generate",
            "websearch": "websearch",
        },
    )
    workflow.add_edge("websearch", "generate")  # web search -> grade documents
    workflow.add_edge(
        "generate", "check_hallucinations"
    )  # generate -> check hallucinations
    workflow.add_conditional_edges(  # check hallucinations -> decide to check hallucinations or end
        "check_hallucinations",
        decide_to_check_hallucinations_or_end,
        {
            "retry": "generate",
            "end": END,
        },
    )

    return workflow


async def run_graph(graph: StateGraph):
    graph = construct_graph()
    app = graph.compile()

    with open(
        "/Users/dingruoqian/code/TLM3001-DP-PhysioLLM/test_prompts/query_1.txt", "r"
    ) as f:
        main_query = f.read()

    inputs = GraphState(
        main_query=main_query,
        subqueries=[],
        documents=[],
        web_search="No",
        generation="",
        halluncination_check_balance=3,
        has_hallucinations=False,
    )

    async for output in app.astream(inputs):
        for key, value in output.items():
            pprint(f"Finished running: {key}")
    pprint(value["generation"])


async def main():
    graph = construct_graph()
    await run_graph(graph)


if __name__ == "__main__":
    asyncio.run(main())

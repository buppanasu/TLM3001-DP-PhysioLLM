from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import StateGraph, END
from pprint import pprint
from typing import List, Literal, Optional
from typing_extensions import TypedDict
from agents.query_translator import query_translator, QueryTranslatorOutput
from agents.retrieval_grader import retrieval_grader, RetrievalGraderOutput
from agents.diagnosis_generator import diagnosis_generator, DiagnosisGeneratorOutput
from agents.halluncination_grader import hallucination_grader, HallucinationGraderOutput
from agents.context_translator import context_translator, ContextTranslatorOutput
from db import get_qdrant_client
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
    hallucination_grader_output: Optional[HallucinationGraderOutput]
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
    db = get_qdrant_client()

    # Retrieve documents from a vector database using the subqueries
    documents = []
    for subquery in subqueries:
        query_result = {"question": subquery, "documents": []}
        results = db.similarity_search_with_score(subquery, k=3)

        for doc, score in results:
            query_result["documents"].append(
                f"source:{doc.metadata['source']}\ncontent:{doc.page_content}"
            )

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
            {"document": document, "question": query},
            {"run_name": f"retrieval-grader-{query_index}-{document_index}"},
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
        web_results = await web_search_tool.ainvoke(
            query, {"run_name": f"web-search-{query_index}"}
        )
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

        documents[query_index]["documents"] = [
            f'source:{doc["url"]}\n{doc["content"]}' for doc in docs
        ]

    return {**graph_state, "documents": documents}


### Node - translate the retrieved documents into a format suitable for the generation model
async def translate_documents(graph_state: GraphState) -> GraphState:
    """Translate the retrieved documents into a format suitable for the generation model"""

    print(
        "--- TRANSLATING RETRIEVED DOCUMENTS INTO A SUITABLE FORMAT FOR GENERATION ---"
    )

    documents = graph_state["documents"]

    async def translate_query_documents(query_index, query_item):
        question = query_item["question"]
        documents = query_item["documents"]

        llm_result = await context_translator.ainvoke(
            {
                "query": question,
                "documents": "\n\n---\n\n".join(documents),
            }
        )
        parsed_llm_result = ContextTranslatorOutput(**llm_result)
        formatted_documents = [
            f"source:{item.source}\ncontent:{item.content}"
            for item in parsed_llm_result.context_documents
        ]

        return {
            "query_index": query_index,
            "documents": formatted_documents,
        }

    invocations = []
    for i, item in enumerate(documents):
        invocations.append(translate_query_documents(i, item))
    translated_documents = await asyncio.gather(*invocations)

    for result in translated_documents:
        query_index = result["query_index"]
        docs = result["documents"]
        documents[query_index]["documents"] = docs

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
        {"context": formatted_context, "question": question},
        {"run_name": "diagnosis-generator"},
    )
    parsed_generation_result = DiagnosisGeneratorOutput(**generation_result)

    generation_result = (
        "## Differential diagnoses based on assessments of the patient:\n\n"
    )

    # Add summary
    generation_result += f"### Summary\n{parsed_generation_result.summary}\n"

    # Add differential diagnoses
    for index, diagnosis in enumerate(
        parsed_generation_result.differential_diagnoses, start=1
    ):
        generation_result += f"### Diagnosis {index}: {diagnosis.diagnosis}\n"
        generation_result += f"**Rationale:** {diagnosis.rational}\n\n"

        generation_result += "##### In-text citations\n"
        for quote in diagnosis.relevant_quotes:
            generation_result += (
                f"\\[{quote.ieee_intext_citation}\\]: {quote.source}\n\n"
            )
            generation_result += f"    - {quote.text}\n"
            generation_result += "\n\n"

        generation_result += "\n"

    # Add references
    if parsed_generation_result.ieee_references:
        generation_result += "### References\n"
        for citation in parsed_generation_result.ieee_references:
            generation_result += f"- {citation}\n"

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

    formatted_documents = []
    for item in documents:
        docs = item["documents"]
        query_str = item["question"]
        docs_str = "\n\n---\n\n".join(docs)
        text = f"Subquery:\n{query_str}\n\nDocuments:\n{docs_str}"
        formatted_documents.append(text)

    formatted_context = "\n\n***\n\n".join(formatted_documents)

    # Run hallucination grader on the generated differential diagnosis
    hallucination_result = hallucination_grader.invoke(
        {"facts": formatted_context, "answer": generation},
        {"run_name": "hallucination-grader"},
    )
    parsed_hallucination_result = HallucinationGraderOutput(**hallucination_result)

    if parsed_hallucination_result.overall_assessment.grounded_score > 0.7:
        has_hallucinations = False
        print("No hallucinations detected in the generated differential diagnosis")
    else:
        has_hallucinations = True
        print(
            "Hallucinations detected in the generated differential diagnosis",
        )
        pprint(parsed_hallucination_result)

    return {
        **graph_state,
        "has_hallucinations": has_hallucinations,
        "hallucination_grader_output": parsed_hallucination_result,
        "halluncination_check_balance": hallucination_count_balance - 1,
    }


# * Edges


### Conditional edge - Determines whether to go to web search or generation based on the web_search flag
def decide_to_do_additional_search(
    graph_state: GraphState,
) -> Literal["translate_documents", "websearch"]:
    """Decides whether to translate documents or conduct web search"""

    print("---ASSESS GRADED DOCUMENTS---")
    web_search = graph_state["web_search"]

    if web_search == "Yes":
        print(
            "---DECISION: THERE ARE SUBQUERIES WITH NO RELEVANT DOCUMENTS, CONDUCT WEB SEARCH---"
        )
        return "websearch"
    else:
        print("---DECISION: TRANSLATE ALL RETRIEVED DOCUMENTS---")
        return "translate_documents"


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

    print("Running test nodes")

    graph_state = GraphState(
        {
            "main_query": main_query,
            "subqueries": [],
            "documents": [],
            "web_search": "No",
            "generation": "",
            "has_hallucinations": False,
            "halluncination_check_balance": 3,
            "hallucination_grader_output": None,
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

    # Translate the retrieved documents into a format suitable for the generation model
    graph_state = await translate_documents(graph_state)
    pprint(graph_state["documents"])

    # Generate differential diagnosis based on the retrieved documents
    # graph_state = await generate(graph_state)
    # pprint(graph_state["generation"])

    # # Check the generated differential diagnosis for hallucinations
    # graph_state = check_hallucinations(graph_state)


def construct_graph() -> StateGraph:
    ### Create Graph and add nodes

    workflow = StateGraph(GraphState)

    # Define the nodes
    workflow.add_node("translate_query", translate_query)  # translate query
    workflow.add_node("retrieve", retrieve)  # retrieve
    workflow.add_node("grade_documents", grade_documents)  # grade documents
    workflow.add_node("websearch", web_search)  # web search
    workflow.add_node("translate_documents", translate_documents)  # translate documents
    workflow.add_node("generate", generate)  # generate
    workflow.add_node(
        "check_hallucinations", check_hallucinations
    )  # check hallucinations

    # Build the graph
    workflow.set_entry_point("translate_query")  # entry point
    workflow.add_edge("translate_query", "retrieve")  # translate query -> retrieve
    workflow.add_edge("retrieve", "grade_documents")  # retrieve -> grade documents
    workflow.add_conditional_edges(  # grade documents -> decide to translate documents or websearch
        "grade_documents",
        decide_to_do_additional_search,
        {
            "translate_documents": "translate_documents",
            "websearch": "websearch",
        },
    )
    workflow.add_edge(
        "websearch", "translate_documents"
    )  # web search -> grade documents
    workflow.add_edge(
        "translate_documents", "generate"
    )  # translate documents -> generate
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


async def run_graph(graph: StateGraph) -> str:
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
        hallucination_grader_output=None,
    )

    async for output in app.astream(inputs):
        for key, value in output.items():
            pprint(f"Finished running: {key}")

    return value["generation"]


async def main():
    # await test_nodes()
    graph = construct_graph()
    await run_graph(graph)


if __name__ == "__main__":
    asyncio.run(main())

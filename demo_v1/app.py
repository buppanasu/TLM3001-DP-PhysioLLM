import asyncio
import streamlit as st
from graph import GraphState, construct_graph

st.title("🩺💪🩻 PhysioTriage")

with st.sidebar:
    """Hello! 👋"""
    st.markdown(
        """
        This is a demo of the PhysioTriage system, which uses an LLM to generate potential differential diagnoses based on subjective and objective patient assessments.
        """
    )


async def run_graph(query: str):
    st.info("🚀 Running the agentic AI workflow")
    graph = construct_graph()
    app = graph.compile()

    inputs = GraphState(
        main_query=query,
        subqueries=[],
        documents=[],
        web_search="No",
        generation="",
        halluncination_check_balance=3,
        has_hallucinations=False,
    )

    step_count = 0

    def translate_query_output(graph_state: GraphState, step_count: int):
        step_count += 1
        for i, subquery in enumerate(graph_state["subqueries"], start=1):
            st.info(f"Subquery {i}: {subquery}", icon="ℹ️")

        st.info(
            f"{step_count}. Retrieving information from database with subqueries...",
            icon="🔍",
        )
        return step_count

    def retrieve_info_output(graph_state: GraphState, step_count: int):
        step_count += 1
        st.info("Retrieved information from the database", icon="📚")
        st.info(f"{step_count}. Checking for irrelevant information", icon="👁️")
        return step_count

    def grade_documents_output(graph_state: GraphState, step_count: int):
        step_count += 1
        documents = graph_state["documents"]
        web_search = graph_state["web_search"]

        subqueries_without_docs = [
            f'ℹ️ {i}. {item["question"]}'
            for i, item in enumerate(documents, start=1)
            if not item["documents"]
        ]
        if subqueries_without_docs:
            formatted_queries = "\n\n".join(subqueries_without_docs)
            st.info(f"⚠️ Subqueries without relevant documents:\n\n {formatted_queries}")

        if web_search == "Yes":
            st.info(
                f"{step_count}. Conducting web search for additional information...",
                icon="🌐",
            )
        else:
            st.info(
                f"{step_count}. Generating potential differential diagnoses...",
                icon="🧬",
            )

        return step_count

    def web_search_output(graph_state: GraphState, step_count: int):
        step_count += 1
        st.info(
            "Web search complete, found relevant information for all subqueries",
            icon="🌐",
        )
        st.info(f"{step_count}. Translating retrieved documents...", icon="⚙️")
        return step_count

    def translate_documents_output(graph_state: GraphState, step_count: int):
        step_count += 1
        st.info("Translated retrieved documents", icon="📚")
        st.info(
            f"{step_count}. Generating potential differential diagnoses...", icon="🧬"
        )
        return step_count

    def generation_output(graph_state: GraphState, step_count: int):
        step_count += 1
        st.info("Potential differential diagnoses generated", icon="📋")
        st.info(
            f"{step_count}. Checking for hallucinations in the generated text...",
            icon="👻",
        )
        return step_count

    def check_hallucinations_output(graph_state: GraphState, step_count: int):
        step_count += 1
        has_hallucinations = graph_state["has_hallucinations"]

        if has_hallucinations:
            st.error(
                "Hallucinations detected in the generated text, retry generation",
                icon="🚨",
            )
        else:
            st.info("No hallucinations detected in the generated text")
            st.info(f"{step_count}. Workflow complete!", icon="🎉")

        return step_count

    node_action_output = {
        "translate_query": translate_query_output,
        "retrieve": retrieve_info_output,
        "grade_documents": grade_documents_output,
        "websearch": web_search_output,
        "translate_documents": translate_documents_output,
        "generate": generation_output,
        "check_hallucinations": check_hallucinations_output,
    }

    step_count += 1
    st.info(f"{step_count}. Translating the main query into subqueries...", icon="🧠")
    async for output in app.astream(inputs):
        for key, value in output.items():
            if key in node_action_output:
                step_count = node_action_output[key](value, step_count)
            else:
                st.info(f"Finished running: {key}")

    st.info(value["generation"])


default_subjective_assessment = "The patient, is a 45-year-old male who presents to the clinic with complaints of lower back pain that has been bothering him for the past two weeks. He describes the pain as dull and achy, located in the lumbar region, with occasional radiation down his left leg. He notes that the pain worsens with prolonged sitting or standing and is relieved by lying down. He denies any recent trauma or injury but mentions that he has a history of occasional low back pain, especially after heavy lifting or prolonged periods of inactivity. He rates the pain as a 6 out of 10 on the pain scale."
default_objective_assessment = "On physical examination, the patient appears uncomfortable but is able to walk into the examination room without assistance. Vital signs are within normal limits. Inspection of the lumbar spine reveals no obvious deformities or asymmetry. Palpation elicits tenderness over the paraspinal muscles of the lumbar spine, particularly on the left side. Range of motion of the lumbar spine is mildly restricted, with pain on forward flexion and left lateral bending. Straight leg raise test is positive on the left side at 45 degrees, reproducing his symptoms of radiating pain down the left leg. Neurological examination reveals intact sensation and strength in the lower extremities, with no signs of motor weakness or sensory deficits."


with st.form("my_form"):
    subjective_assessment = st.text_area(
        "Enter the subjective assessment of the patient:",
        default_subjective_assessment,
        height=200,
    )
    objective_assessment = st.text_area(
        "Enter the objective assessment of the patient:",
        default_objective_assessment,
        height=200,
    )

    submitted = st.form_submit_button("Submit")

    if submitted:
        query = f"""
Given the following subjective and objective assessment, provide a well informed and researched differential diagnosis

Subjective assessment:
{subjective_assessment}

Objective assessment:
{objective_assessment}
"""
        asyncio.run(run_graph(query=query))

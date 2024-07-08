import asyncio
import streamlit as st
from graph import GraphState, construct_graph

st.title(
    "🩺💪🩻 PhysioTriage - LLM system for generating potential differential diagnoses"
)

with st.sidebar:
    openai_api_key = st.text_input("OpenAI API Key", type="password")
    "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"


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

    async for output in app.astream(inputs):
        for key, value in output.items():
            st.info(f"Finished running: {key}")
    st.info(value["generation"])


with st.form("my_form"):
    subjective_assessment = st.text_area(
        "Enter the subjective assessment of the patient:",
    )
    objective_assessment = st.text_area(
        "Enter the objective assessment of the patient:",
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

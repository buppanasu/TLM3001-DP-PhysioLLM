import asyncio
import streamlit as st
from agents.halluncination_grader import HallucinationGraderOutput
from graph import GraphState, construct_graph
from chatbot_ui import chatbot_page  # <-- Import the chatbot module

# Main title
st.title("🩺💪🩻 PhysioTriage")

# Sidebar with tab options
tab = st.sidebar.radio("Navigate", ["PhysioTriage", "Chatbot"])

with st.sidebar:
    """Hello! 👋"""
    st.markdown(
        """
        This is a demo of the PhysioTriage system, which uses an LLM to generate potential differential diagnoses based on subjective and objective patient assessments.
        """
    )

# Check which tab is active
if tab == "PhysioTriage":
    # Original diagnosis logic here...

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
            hallucination_grader_output=None,
        )

        step_count = 0

        def format_to_markdown(data: HallucinationGraderOutput):
            md = []
            oa = data.overall_assessment

            md.append("# Hallucination Grader Report\n")

            # Overall Assessment
            md.append("## Overall Assessment\n")
            md.append(f"**Grounded Score**: {oa.grounded_score}\n")
            md.append(f"**Confidence**: {oa.confidence}\n")
            md.append(f"**Summary**: {oa.summary}\n")

            # Verified Claims
            md.append("\n## Verified Claims\n")
            for i, claim in enumerate(data.verified_claims, start=1):
                md.append(f"#### Claim\n")
                md.append(f"**Claim**: {claim.claim}\n")
                md.append(f"**Is Grounded**: {'Yes' if claim.is_grounded else 'No'}\n")
                md.append(f"**Supporting Evidence**: {claim.supporting_evidence}\n")
                md.append(f"**Explanation**: {claim.explanation}\n")

            # Identified Hallucinations
            md.append("\n## Identified Hallucinations\n")
            if data.identified_hallucinations:
                for hallucination in data.identified_hallucinations:
                    md.append(f"- {hallucination}\n")
            else:
                md.append("None\n")

            return "\n".join(md)

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
            halluncation_grader_output = graph_state["hallucination_grader_output"]

            st.info(f"{step_count}. Workflow complete!", icon="🎉")

            if halluncation_grader_output:
                halluncination_grader_report = format_to_markdown(
                    halluncation_grader_output
                )
                st.info(halluncination_grader_report)

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

    default_subjective_assessment = "The patient, is a 45-year-old male who presents to the clinic with complaints of lower back pain..."
    default_objective_assessment = "On physical examination, the patient appears uncomfortable..."

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
            Given the following subjective and objective assessment, provide a well-informed and researched differential diagnosis.
            Subjective assessment:
            {subjective_assessment}
            Objective assessment:
            {objective_assessment}
            """
            asyncio.run(run_graph(query=query))

elif tab == "Chatbot":
    # Call the chatbot page from chatbot.py
    chatbot_page()
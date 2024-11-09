# Filename: streamlit_app.py

import streamlit as st
import requests

def ingest_documents(files):
    api_url = "http://localhost:8000/ingest-documents/"  # Adjust if your backend is hosted elsewhere
    files_to_upload = [("files", (file.name, file, file.type)) for file in files]
    
    try:
        response = requests.post(api_url, files=files_to_upload)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"success": False, "error": str(e)}

def document_ingestion_page():
    st.header("Ingest Documents for PhysioBot")

    # Allow multiple file uploads
    uploaded_files = st.file_uploader(
        "Choose PDF files to upload",
        type=['pdf'],
        accept_multiple_files=True
    )

    if uploaded_files:
        for file in uploaded_files:
            st.write(f"📄 {file.name}")

        # Process documents when button is pressed
        if st.button("Ingest Documents"):
            with st.spinner("Processing documents..."):
                results = ingest_documents(uploaded_files)
                
                if results.get("success"):
                    st.success("Documents ingested successfully!")
                    for success_file in results["results"]["success"]:
                        st.write(f"✅ {success_file} ingested.")
                    for error in results["results"]["errors"]:
                        st.write(f"❌ Failed to ingest {error['file']}: {error['error']}")
                else:
                    st.error("An error occurred during ingestion.")
                    st.write(results.get("error"))

if __name__ == "__main__":
    document_ingestion_page()

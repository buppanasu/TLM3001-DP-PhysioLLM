import io
import streamlit as st
import os
import openai
import json
from dotenv import load_dotenv
from ingest_funcs import *  # Import your ingestion functions from the previous code
from helper_funcs import (
    postprocess_json,
    is_valid_pdf,
)  # Make sure to import the postprocess_json function
from config import *



openai.api_key = OPENAI_API_KEY

def document_ingestion_page():
    """
    Streamlit page for uploading, processing, and ingesting PDF documents into a vector database.
    """

    st.header("Ingest Documents for PhysioBot")

    # File uploader widget to allow multiple PDF files to be uploaded
    uploaded_files = st.file_uploader(
        "Choose PDF files to upload", type=["pdf"], accept_multiple_files=True
    )

    if uploaded_files:
        # Separate valid and invalid files based on size and type
        valid_files = [
            file
            for file in uploaded_files
            if is_valid_pdf(file)
            and file.size <= MAX_FILE_SIZE_MB * 1024 * 1024
        ]
        invalid_files = [
            file
            for file in uploaded_files
            if not is_valid_pdf(file)
            or file.size > MAX_FILE_SIZE_MB * 1024 * 1024
        ]

        # Display valid files to the user
        for file in valid_files:
            st.write(f"📄 {file.name}")

        # Display warnings for invalid files with specific reasons
        if invalid_files:
            for file in invalid_files:
                if not is_valid_pdf(file):
                    st.warning(
                        f"⚠️ {file.name} is not a PDF file and will not be uploaded."
                    )
                if file.size > MAX_FILE_SIZE_MB * 1024 * 1024:
                    st.warning(
                        f"⚠️ {file.name} exceeds the {MAX_FILE_SIZE_MB}MB size limit and will not be uploaded."
                    )

        # Button to start processing and storing the uploaded documents
        if st.button("Process and Store Documents"):
            if valid_files:
                # Display a loading spinner while processing the documents
                with st.spinner("Processing documents..."):
                    all_documents = []  # List to store processed documents
                    for file in valid_files:
                        # Read the file content into memory
                        file_content = file.read()
                        # Temporarily save the file content as a PDF
                        with open(file.name, "wb") as temp_pdf:
                            temp_pdf.write(file_content)

                        try:
                            # Convert the PDF to markdown and JSON
                            json_path, condition_name = (
                                convert_to_markdown_and_json(file.name)
                            )
                            metadata = {"condition": condition_name}

                            # Load JSON content from the converted file
                            with open(json_path, "r") as json_file:
                                json_data = json.load(json_file)

                            # Post-process the JSON to add hierarchical structure and chunking information
                            postprocessed_data = postprocess_json(json_data)

                            # Create documents using the post-processed data and metadata
                            for entry in postprocessed_data:
                                # Initialize an empty list to collect heading strings
                                heading_content = []

                                # Loop through possible heading levels (heading1, heading2, etc.)
                                for i in range(1, 5):
                                    heading_key = f"heading{i}"
                                    # Add heading levels to the list if present in metadata
                                    if heading_key in entry["metadata"]:
                                        heading_content.append(
                                            str(entry["metadata"][heading_key])
                                        )

                                # Join the heading parts with a colon and add the content at the end
                                page_content = (
                                    ": ".join(heading_content)
                                    + ": "
                                    + entry["content"]
                                )

                                # Create a document object with page content and metadata
                                document = Document(
                                    page_content=page_content,
                                    metadata={
                                        "chunk_content": entry["metadata"].get(
                                            "chunk_content", ""
                                        ),
                                        "source": file.name,
                                    },  # Safely fetch chunk_content
                                )
                                all_documents.append(document)

                            # Optionally, clean up the temporary saved PDF after processing
                            os.remove(file.name)

                        except Exception as e:
                            # Display an error message if processing fails for a file
                            st.error(f"Error processing {file.name}: {e}")

                    # Initialize embeddings and store documents in Qdrant
                    if all_documents:
                        st.spinner("Storing documents in Qdrant...")
                        try:
                            # Load embeddings model
                            embeddings = HuggingFaceEmbeddings(
                                model_name=VECTOR_EMBEDDING_MODEL
                            )
                            # Store documents in the Qdrant vector database
                            qdrant.Qdrant.from_documents(
                                all_documents,
                                embeddings,
                                url= QDRANT_ENDPOINT,  # Specify the vector database URL
                                collection_name=QDRANT_COLLECTION_NAME,  # Name of the collection
                                prefer_grpc=False,  # Disable GRPC (optional)
                            )
                            st.success(
                                "Documents successfully processed and stored in Qdrant."
                            )
                        except Exception as e:
                            # Display an error message if storing fails
                            st.error(f"Error storing in Qdrant: {e}")
            else:
                # Warn the user if no valid files were uploaded
                st.warning(
                    "No valid PDF files were uploaded. Please upload PDF files within the size limit."
                )


if __name__ == "__main__":
    # Run the Streamlit app
    document_ingestion_page()

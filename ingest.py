import os
import re
import json
from langchain.schema import Document
from langchain_community.vectorstores import qdrant
from langchain_community.document_loaders.directory import DirectoryLoader
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from pymupdf4llm import to_markdown
import openai

DOCUMENT_DIR = "documents"
MARKDOWN_DIR = "./markdown_files"
JSON_DIR = "./json_files"

os.makedirs(MARKDOWN_DIR, exist_ok=True)
os.makedirs(JSON_DIR, exist_ok=True)

openai.api_key = "YOUR KEY"

# to extract main condtion from the content of the md file
def extract_condition_from_content(markdown_content: str, word_limit=200):
    """Extract the main condition from the content of the Markdown file using ChatGPT."""
    words = markdown_content.split()
    truncated_content = " ".join(words[:word_limit])

    prompt = f"""The following text is extracted from a medical PDF. Based on the text below, identify the main medical condition or topic discussed. Provide a concise name for the condition.

    Text: {truncated_content}

    Main Condition:"""

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
        condition = response["choices"][0]["message"]["content"].strip()
        return condition
    except Exception as e:
        print(f"Error extracting condition from content: {e}")
        return "Unknown Condition"



def convert_to_markdown_and_json(pdf_path: str):
    """Converts a PDF file to Markdown and then to JSON."""
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    markdown_path = os.path.join(MARKDOWN_DIR, f"{base_name}.md")
    json_path = os.path.join(JSON_DIR, f"{base_name}.json")

    # Convert to Markdown
    if not os.path.exists(markdown_path):
        print(f"Converting {pdf_path} to Markdown...")
        markdown_content = to_markdown(pdf_path)
        with open(markdown_path, "w") as md_file:
            md_file.write(markdown_content)
        print(f"Markdown saved to {markdown_path}")
    else:
        print(f"Markdown for {pdf_path} already exists. Skipping conversion.")

    # Extract condition from the first 200 words of the content
    with open(markdown_path, "r") as md_file:
        markdown_content = md_file.read()
    condition_name = extract_condition_from_content(markdown_content, word_limit=200)

    # Convert Markdown to JSON via ChatGPT
    if not os.path.exists(json_path):
        print(f"Converting Markdown to JSON for {base_name}...")
        if len(markdown_content) > 10_000:
            print(f"Markdown content too large, truncating for {base_name}.")
            markdown_content = markdown_content[:10_000] + "\n\n[Content truncated]"

        prompt = f"""Convert the following structured text into JSON format. Follow these specific instructions to handle split sections, footnotes, and create a clean, structured JSON output.

        The JSON output should contain the below keys, and the content from the markdown will be mapped to one of these keys:

        Pathophysiology: General nature and mechanisms of PFPS
        Aetiology: Additional names and causes
        Clinical Presentations: Symptoms, pain factors, specific tests
        Differential Diagnosis: Conditions to consider and rule out
        Subjective and Objective Assessments: Clinicians' assessments
        Physiotherapy Assessments: Tests by physiotherapists
        Physiotherapy Interventions: Treatment strategies
        Medical Interventions: Surgery or treatments beyond physiotherapy
        Contraindications and Precautions: Safety considerations
        Criteria to Progress: Milestones for rehabilitation
        Patient Education: Educating patients about recovery
        Medical/Radiological Assessments: Diagnostic methods
        Convert the structured text into JSON format based on the following keys. Avoid using "Phase" as a key; instead, integrate its information into the relevant keys. If the markdown lacks details for a key, include the key with the string "PDF does not have information about this." Contextual interpretation is encouraged for accurate mapping.

        Content: {markdown_content}
        """
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}]
            )
            json_content = response["choices"][0]["message"]["content"]
            json_data = json.loads(json_content)

            with open(json_path, "w") as json_file:
                json.dump(json_data, json_file, indent=4)
            print(f"JSON saved to {json_path}")
        except json.JSONDecodeError as e:
            print(f"Failed to decode JSON response for {pdf_path}: {e}")
        except Exception as e:
            print(f"Error converting Markdown to JSON for {pdf_path}: {e}")
    else:
        print(f"JSON for {pdf_path} already exists. Skipping conversion.")

    return json_path, condition_name



def create_documents_from_json(json_data, pdf_path, condition_name, metadata):
    """Create documents from JSON where each key-value pair becomes a single document."""
    documents = []
    for key, value in json_data.items():
        if isinstance(value, dict) or isinstance(value, list):
            value = json.dumps(value, indent=2)
        document = Document(
            page_content=f"{key}: {value}",
            metadata={
                **metadata,
                "key": key,
                "source": os.path.basename(pdf_path),
                "condition": condition_name,
            },
        )
        documents.append(document)
    return documents


def main():
    vector_db_url = "http://127.0.0.1:6333"
    processed_files = set()

    print("=== Step 1: Starting Document Ingestion ===")
    loader = DirectoryLoader(
        DOCUMENT_DIR, glob="*.pdf", show_progress=True, loader_cls=PyPDFLoader
    )
    print("Loading documents from the directory...")
    documents = loader.load()
    print(f"Total documents loaded: {len(documents)}")

    if len(documents) == 0:
        print("No documents found. Ensure the 'documents' folder contains PDF files.")
        return

    print("=== Step 2: Converting PDFs to Markdown and JSON ===")
    all_documents = []
    for doc in documents:
        pdf_path = doc.metadata["source"]
        if pdf_path not in processed_files:
            try:
                
                json_path, condition_name = convert_to_markdown_and_json(pdf_path)
                metadata = {"condition": condition_name}

                # Load JSON content
                with open(json_path, "r") as json_file:
                    json_data = json.load(json_file)
                    
                    # Create documents using the JSON data and metadata
                    documents = create_documents_from_json(json_data, pdf_path, condition_name, metadata)
                    all_documents.extend(documents)

                # Mark the file as processed
                processed_files.add(pdf_path)
            except Exception as e:
                print(f"Error processing {pdf_path}: {e}")

    print(f"Total documents created: {len(all_documents)}")

    print("=== Step 3: Initializing Embeddings ===")
    embeddings = HuggingFaceEmbeddings(model_name="NeuML/pubmedbert-base-embeddings")
    print("Embeddings model initialized.")

    print("=== Step 4: Storing Documents in Qdrant ===")
    try:
        qdrant.Qdrant.from_documents(
            all_documents,
            embeddings,
            url=vector_db_url,
            collection_name="physio-textbooks",
            prefer_grpc=False,
        )
        print("Successfully stored all documents in Qdrant.")
    except Exception as e:
        print(f"An error occurred while storing in Qdrant: {e}")
        return

    print("=== Process Completed Successfully ===")


if __name__ == "__main__":
    main()

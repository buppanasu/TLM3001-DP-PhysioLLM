import os
import re
import json
from langchain.schema import Document
from langchain_community.vectorstores import qdrant
from langchain_community.document_loaders.directory import DirectoryLoader
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from pymupdf4llm import to_markdown
from dotenv import load_dotenv
from doclingparser import doc_converter
from helper_funcs import clean_json_response
from prompts import get_json_generation_prompt, get_medical_condition_prompt
import openai
from config import *

# ENV
openai.api_key = os.getenv("OPENAI_API_KEY")

# DIRECTORIES
os.makedirs(MARKDOWN_DIR, exist_ok=True)
os.makedirs(JSON_DIR, exist_ok=True)


# Extract condition from content
def extract_condition_from_content(markdown_content: str, word_limit=200):
    """Extract the main condition from the content of the Markdown file using ChatGPT."""
    words = markdown_content

    print(
        f"Debug: Extracting condition from content with word limit of {word_limit}"
    )

    prompt = get_medical_condition_prompt(words)

    try:

        print(
            f"Debug: Prompt for OpenAI API:\n{prompt[:300]}..."
        )  # Print the first 300 characters of the prompt for clarity
        # Correct method for calling OpenAI's API
        response = openai.chat.completions.create(
            model="gpt-4o-mini", messages=[{"role": "user", "content": prompt}]
        )

        print(f"Debug: OpenAI response received: {response}")

        # Access the response properly
        condition = response.choices[0].message.content.strip()
        print(f"Debug: Extracted condition: {condition}")
        return condition

    except Exception as e:
        print(f"Error extracting condition from content: {e}")
        return "Unknown Condition"


# Convert to markdown utilizing Docling and then convert with LLM into JSON
def convert_to_markdown_and_json(pdf_path: str):
    """Converts a PDF file to Markdown and then to JSON."""
    print(f"Debug: Starting conversion for PDF: {pdf_path}")
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    markdown_path = os.path.join(
        MARKDOWN_DIR, f"{base_name}.md"
    )  # To set the name to save MD
    json_path = os.path.join(
        JSON_DIR, f"{base_name}.json"
    )  # To set the name to sav JSON

    # Convert to Markdown
    if not os.path.exists(
        markdown_path
    ):  # If we currently do not have the MD file in our MD folder
        print(f"Converting {pdf_path} to Markdown...")
        result = doc_converter.convert(pdf_path)
        docling_markdown_content = (
            result.document.export_to_markdown()
        )  # Here we store the markdown output
        pymupdf4llm_markdown_content = to_markdown(pdf_path)
        # For Debugging purposes
        with open(markdown_path, "w") as md_file:
            md_file.write(
                docling_markdown_content
            )  # We write this output into the path
        print(f"Markdown saved to {markdown_path}")
    else:
        print(f"Markdown for {pdf_path} already exists. Skipping conversion.")

    with open(markdown_path, "r") as md_file:
        markdown_content = md_file.read()  # we dont really need this
    condition_name = extract_condition_from_content(
        markdown_content, word_limit=200
    )  # Extract condition from content, but where is this extracted condition used?

    # Convert Markdown to JSON via ChatGPT
    if not os.path.exists(json_path):
        print(f"Debug: Converting Markdown to JSON for {base_name}...")

        prompt = get_json_generation_prompt(
            markdown_content, pymupdf4llm_markdown_content
        )

        try:
            print(
                f"Debug: Prompt for JSON conversion:\n{prompt[:300]}..."
            )  # Print the first 300 characters of the prompt
            response = openai.chat.completions.create(
                model="gpt-4o-mini",  # Correct model name from "gpt-4o" to "gpt-4"
                messages=[{"role": "user", "content": prompt}],
            )
            print(f"Debug: OpenAI JSON response: {response}")
            json_content = response.choices[
                0
            ].message.content  # Correcting the subscripting of response
            print(f"JSON CONTENT: {json_content}")
            # Strip the backticks
            cleaned_json_content = clean_json_response(json_content)
            print(f"CLEANED JSON: {cleaned_json_content}")
            json_data = json.loads(cleaned_json_content)

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
    print(f"Debug: Creating documents from JSON for {pdf_path}")
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

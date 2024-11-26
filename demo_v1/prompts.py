# prompts.py


def get_json_generation_prompt(markdown_content, pymupdf4llm_markdown_content):
    """
    Function to generate the prompt for converting markdown content into structured JSON.
    """
    return f"""
You are given two versions of markdown content from a PDF document. One version contains the hierarchical structure and headings but lacks some of the content (docling_markdown_content), while the other contains more complete content but may not fully preserve the structure (pymupdf4llm_markdown_content). Your task is to convert this content into a clean, structured JSON output by:

1. **Hierarchical Structure**: Treat each heading or subheading as a key, creating a nested dictionary for each level.
   - Only the lowest level of headings should contain actual content.
   - Use the structure from docling_markdown_content for hierarchy, supplementing with content from pymupdf4llm_markdown_content where missing.

2. **Joining Split Content**: If a sentence or phrase is cut off due to page or line breaks:
   - Detect incomplete sentences or phrases that end abruptly, especially if they end mid-sentence.
   - Look at the start of the following section to determine if it completes the previous content, and merge them seamlessly.
   - Ensure there are no duplicate words when merging (e.g., if merging "conversation" and "conversation then," keep only one instance of "conversation").

3. **Handling Explicit References**: If an item includes an explicit reference to another section or list (e.g., "See Attendee List below" or "refer to the details provided later"), replace the reference with the actual content from the specified section.
   - Detect phrases like "see below," "refer to," "see list," or other similar expressions that indicate a link to content elsewhere in the document.
   - Embed the referenced content directly in place of the phrase, ensuring it is integrated seamlessly and contextually.
   - Only form relationships when there is a clear and explicit reference, and avoid linking content that is not specifically indicated as a reference.

4. **Content Accuracy**: Copy the content exactly as it appears without summarizing, rephrasing, or adding new words. The final JSON should contain verbatim text.

5. **Second Final Validation**: After forming the JSON structure, perform a final check to confirm that:
   - All sentences are complete and coherent, with no fragments or hanging phrases.
   - Split sections and footnotes are correctly merged into a single, continuous text block where necessary.

6. **Final Validation**: Ensure that the final JSON is clean, well-structured, and complete with no missing or fragmented content.

Here’s the structured text to convert:
- **Docling Markdown Content**: {markdown_content}
- **Pymupdf4llm Markdown Content**: {pymupdf4llm_markdown_content}
"""


def get_medical_condition_prompt(words):
    """
    Function to generate the prompt for identifying the main medical condition or topic from text.
    """
    return f"""The following text is extracted from a medical PDF. Based on the text below, identify the main medical condition or topic discussed. Provide a concise name for the condition.

    Text: {words}

    Main Condition:"""

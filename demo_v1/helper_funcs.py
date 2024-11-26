import re


def is_valid_pdf(file):
    """
    Check if the uploaded file is a valid PDF.
    Returns True if the file type is 'application/pdf', False otherwise.
    """
    return file.type == "application/pdf"


def process_content(key, value, metadata, processed_data):
    """
    Process the content and handle the nested structure and lists.
    """
    # If the value is a list, join it into a string
    if isinstance(value, list):
        # Ensure all items in the list are strings
        value = "\n".join(
            str(item) if not isinstance(item, dict) else str(item)
            for item in value
        )
        chunk_content = "no"  # Mark as "no" for chunked content
    else:
        chunk_content = "yes"  # Mark as "yes" for non-chunked content

    # Add processed data to the output with content and metadata
    processed_data.append(
        {
            "content": value,
            "metadata": {**metadata, "chunk_content": chunk_content},
        }
    )


def traverse_json(obj, parent_metadata, processed_data, heading_level=1):
    """
    Recursively traverse the JSON and add metadata for each heading.
    """
    for key, value in obj.items():
        # Adjust heading based on level of nesting
        new_metadata = {**parent_metadata, f"heading{heading_level}": key}

        # If the value is a dictionary, recurse further
        if isinstance(value, dict):
            traverse_json(
                value, new_metadata, processed_data, heading_level + 1
            )
        else:
            # Process the content at the current level
            process_content(key, value, new_metadata, processed_data)


def postprocess_json(json_data):
    """
    Post-process the nested JSON to extract content with hierarchical metadata and chunk_content.
    """
    processed_data = []

    # Start traversing the JSON from the root
    for key, value in json_data.items():
        traverse_json({key: value}, {}, processed_data, heading_level=1)

    return processed_data


def clean_json_response(response_content):
    """
    Cleans the OpenAI JSON response to ensure that only valid JSON is returned
    by extracting the content between the first ```json and the ending ``` markers.
    """
    # Match content strictly between the ```json and ```
    match = re.search(r"```json(.*?)```", response_content, re.DOTALL)
    if match:
        # Extract the JSON content and return it after stripping whitespace
        return match.group(1).strip()
    else:
        # Return an empty string or raise an error if no match is found
        return ""

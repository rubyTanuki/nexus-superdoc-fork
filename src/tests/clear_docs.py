#!/usr/bin/env python3
import json
from pathlib import Path


# Updated import to match your project's lowercase naming convention
from src.superdoc import superdoc 
import os 
COURSE_ID   = os.getenv("TEST_COURSE_ID", "TestCourse101")

def clear_document(doc_id: str):
    """Wipes all text content from a Google Doc using the native superdoc wrapper."""
    try:
        # Initialize with the correct uppercase parameter name (DOCUMENT_ID)
        # If COURSE_ID is strictly required by your class constructor, 
        # you can append COURSE_ID="dev" or similar dummy string here.

        sd = superdoc(DOCUMENT_ID=doc_id,COURSE_ID=COURSE_ID)

        # Fetch the current document layout structure to calculate the total length (endIndex)
        doc = sd.docs_editor.get_document_structure(document_id=doc_id)
        body_content = doc.get("body", {}).get("content", [])

        if not body_content:
            return

        # Determine the final index boundaries of the document text
        end_index = body_content[-1].get("endIndex", 0)

        # A completely empty Google Doc has a minimum endIndex of 2 (just a permanent trailing newline)
        if end_index > 2:
            print(f"Emptying content for Doc ID: {doc_id}")
            
            # Formulate the API request to clear text while protecting the trailing newline character
            clear_request = {
                "deleteContentRange": {
                    "range": {
                        "startIndex": 1, 
                        "endIndex": end_index - 1
                    }
                }
            }
            
            # Execute the update batch via your native class architecture
            sd.docs_editor.batch_update([clear_request])
        else:
            print(f"Doc ID {doc_id} is already completely empty.")

    except Exception as e:
        print(f"Error clearing doc {doc_id} via superdoc wrapper: {e}")


def main():
    test_docs_path = Path("/var/task/src/tests/test_docs.json")

    if not test_docs_path.exists():
        print("No test_docs.json tracking file found inside the container.")
        return

    try:
        with open(test_docs_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        files = data.get("files", {})
        if not files:
            print("Tracked file list is empty inside test_docs.json.")
            return

        print("\nInitiating Google Document Clear via superdoc Class...")
        print("-" * 50)
        for name, val in files.items():
            doc_id = val.get("id") if isinstance(val, dict) else val
            if doc_id and not str(doc_id).startswith("http"):
                print(f"• Processing '{name}'")
                clear_document(str(doc_id).strip())
        print("-" * 50 + "\n")

    except Exception as e:
        print(f"Failed to run documentation wiping sequence: {e}")


if __name__ == "__main__":
    main()
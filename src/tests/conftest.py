"""
conftest.py — Shared fixtures for the superdoc test suite.

Handles:
  - Loading / saving test_docs.json (pdf filename → Google Doc ID map)
  - Auto-creating Google Docs for PDFs that don't have one yet
  - Providing a ready-to-use superdoc instance per test
"""

import json
import os
import pytest
from io import BytesIO
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

TESTS_DIR   = Path(__file__).parent
TEST_DOCS   = TESTS_DIR / "test_docs.json"
FILES_DIR   = TESTS_DIR.parent / "files"          # default PDF location
COURSE_ID   = os.getenv("TEST_COURSE_ID", "TestCourse101")


# --------------------------------------------------------------------------
# test_docs.json helpers
# --------------------------------------------------------------------------

def load_test_docs() -> dict:
    """Return the full test_docs mapping (creates file if missing)."""
    if not TEST_DOCS.exists():
        TEST_DOCS.write_text(json.dumps({"files": {}}, indent=2))
    return json.loads(TEST_DOCS.read_text())


def save_test_docs(data: dict) -> None:
    TEST_DOCS.write_text(json.dumps(data, indent=2))


def get_or_create_doc_id(pdf_name: str, gdoc_editor) -> str:
    """
    Look up the Google Doc ID for *pdf_name* in test_docs.json.
    If not present, create a new Google Doc and persist the ID.
    """
    data = load_test_docs()
    files = data.setdefault("files", {})

    if pdf_name in files:
        doc_id = files[pdf_name]
        print(f"[test_docs] Found existing doc for '{pdf_name}': {doc_id}")
        return doc_id

    # Create a new doc
    print(f"[test_docs] No doc found for '{pdf_name}', creating one...")
    response = gdoc_editor.create_google_doc(name=pdf_name, courseid=COURSE_ID)
    if response is None:
        raise RuntimeError(f"Failed to create Google Doc for '{pdf_name}'")

    doc_id = response.get("documentId")
    files[pdf_name] = doc_id
    save_test_docs(data)
    print(f"[test_docs] Saved new doc ID for '{pdf_name}': {doc_id}")
    return doc_id


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------

@pytest.fixture(scope="session")
def gdoc_editor():
    """Single GoogleDocsEditor instance shared across the whole test session."""
    from services.gdocs_client import GoogleDocsEditor
    return GoogleDocsEditor()


@pytest.fixture
def make_superdoc(gdoc_editor):
    """
    Factory fixture: returns a callable that builds a superdoc instance for a
    given PDF file, resolving (or creating) its Google Doc ID automatically.

    Usage inside a test:
        def test_something(make_superdoc):
            sd, stream, doc_id = make_superdoc("my_lecture.pdf")
            ...
    """
    from src.superdoc import superdoc

    def _factory(pdf_filename: str, files_dir: Path = FILES_DIR):
        pdf_path = Path(files_dir) / pdf_filename
        if not pdf_path.exists():
            pytest.skip(f"PDF not found: {pdf_path}")

        doc_id = get_or_create_doc_id(pdf_filename, gdoc_editor)

        sd = superdoc(
            DOCUMENT_ID=doc_id,
            COURSE_ID=COURSE_ID,
        )

        stream = BytesIO(pdf_path.read_bytes())
        return sd, stream, doc_id

    return _factory


@pytest.fixture
def pdf_stream():
    """
    Convenience fixture: returns a factory that opens a PDF as a BytesIO stream.
    Useful when you only need the stream without a full superdoc instance.
    """
    def _open(pdf_filename: str, files_dir: Path = FILES_DIR):
        pdf_path = Path(files_dir) / pdf_filename
        if not pdf_path.exists():
            pytest.skip(f"PDF not found: {pdf_path}")
        return BytesIO(pdf_path.read_bytes())

    return _open
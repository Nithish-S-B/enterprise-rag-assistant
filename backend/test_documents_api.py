"""Tests for POST /api/documents/upload: validation, ingestion, and indexing."""
import os
import sys
import tempfile
import unittest.mock
from pathlib import Path

import chromadb
from fastapi.testclient import TestClient

from app.main import app
from app.vector_store import CHROMA_DIR, COLLECTION_NAME


MINIMAL_PDF_CONTENT = (
    b"%PDF-1.4\n"
    b"1 0 obj\n<< /Type /Catalog >>\nendobj\n"
    b"trailer\n<< /Root 1 0 R >>\n%%EOF\n"
)
CORRUPT_PDF_CONTENT = b"%PDF-1.4\ncorrupt payload without xref or eof"

UPLOAD_ENDPOINT = "/api/documents/upload"
TEMP_FILE_PREFIX = "rag_upload_"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REAL_PDF_NAME = "EMPLOYEE REMOTE WORK POLICY.pdf"
REAL_PDF_PATH = PROJECT_ROOT / "documents" / REAL_PDF_NAME


def _upload_file(client: TestClient, filename: str, content: bytes,
                 content_type: str = "application/pdf"):
    return client.post(
        UPLOAD_ENDPOINT,
        files={"file": (filename, content, content_type)},
    )


def _temp_leftovers() -> set[str]:
    temp_dir = tempfile.gettempdir()
    return {
        name for name in os.listdir(temp_dir)
        if name.startswith(TEMP_FILE_PREFIX)
    }


def test_valid_pdf_upload_ingests_and_indexes() -> bool:
    """A: A real repository PDF passes validation and is fully indexed."""
    assert REAL_PDF_PATH.exists(), f"Expected real PDF at {REAL_PDF_PATH}."
    client = TestClient(app)
    with open(REAL_PDF_PATH, "rb") as handle:
        pdf_bytes = handle.read()

    response = _upload_file(client, REAL_PDF_NAME, pdf_bytes)

    assert response.status_code == 200, (
        f"Expected HTTP 200, got {response.status_code}: {response.text}"
    )

    body = response.json()
    assert body["filename"] == REAL_PDF_NAME, "Expected original filename echoed."
    assert body["status"] == "indexed", "Expected status 'indexed'."
    assert body["pages"] > 0, "Expected at least one parsed page."
    assert body["chunks"] > 0, "Expected at least one generated chunk."
    assert body["document_id"], "Expected a document_id to be present."

    document_id = body["document_id"]
    assert "/" not in document_id and "\\" not in document_id, (
        "document_id must not leak filesystem paths."
    )
    assert document_id == "employee_remote_work_policy", (
        f"Unexpected deterministic document_id: {document_id}"
    )
    print("POST /api/documents/upload ->", body)
    return True


def test_txt_upload_rejected_without_ingestion() -> bool:
    """B: Non-PDF extensions are rejected with 415 before any processing."""
    client = TestClient(app)
    with unittest.mock.patch("app.api.documents.load_pdf") as spy_loader:
        response = _upload_file(client, "notes.txt", b"plain text content")

    assert response.status_code == 415, (
        f"Expected HTTP 415 for .txt upload, got {response.status_code}: {response.text}"
    )
    spy_loader.assert_not_called(), "Ingestion must not run for invalid extensions."
    return True


def test_missing_file_rejected() -> bool:
    """C1: A request with no file part is rejected by request validation."""
    client = TestClient(app)
    response = client.post(UPLOAD_ENDPOINT)

    assert response.status_code == 422, (
        f"Expected HTTP 422 for missing file, got {response.status_code}"
    )
    return True


def test_blank_filename_rejected() -> bool:
    """C2: A whitespace-only filename is rejected before any processing."""
    client = TestClient(app)
    response = _upload_file(client, "   ", b"%PDF-1.4 fake")

    assert response.status_code == 422, (
        f"Expected HTTP 422 for blank filename, got {response.status_code}: {response.text}"
    )
    return True


def test_empty_file_rejected() -> bool:
    """D: A zero-byte PDF is syntactically addressed but has no content."""
    client = TestClient(app)
    response = _upload_file(client, "empty.pdf", b"")

    assert response.status_code == 400, (
        f"Expected HTTP 400 for empty upload, got {response.status_code}: {response.text}"
    )
    return True


def test_oversized_file_rejected() -> bool:
    """E: Streaming size enforcement aborts uploads beyond the configured limit."""
    previous_limit = os.environ.get("MAX_UPLOAD_SIZE_MB")
    os.environ["MAX_UPLOAD_SIZE_MB"] = "1"
    try:
        client = TestClient(app)
        oversized_content = MINIMAL_PDF_CONTENT + b"A" * (2 * 1024 * 1024)
        response = _upload_file(client, "large.pdf", oversized_content)

        assert response.status_code == 413, (
            f"Expected HTTP 413 for oversized upload, got "
            f"{response.status_code}: {response.text[:200]}"
        )
    finally:
        if previous_limit is None:
            os.environ.pop("MAX_UPLOAD_SIZE_MB", None)
        else:
            os.environ["MAX_UPLOAD_SIZE_MB"] = previous_limit
    return True


def test_pdf_extension_with_fake_content_rejected() -> bool:
    """Signature check: .pdf name with non-PDF bytes must not pass validation."""
    client = TestClient(app)
    response = _upload_file(client, "disguised.pdf", b"definitely not a pdf")

    assert response.status_code == 415, (
        f"Expected HTTP 415 for fake PDF content, got {response.status_code}"
    )
    return True


def test_corrupt_pdf_returns_safe_error_and_cleans_temp() -> bool:
    """F1: Signed-but-corrupt PDF yields a safe error and leaves no temp file."""
    client = TestClient(app)
    before = _temp_leftovers()

    response = _upload_file(client, "corrupt.pdf", CORRUPT_PDF_CONTENT)

    assert response.status_code >= 400, (
        f"Expected an error status for corrupt PDF, got {response.status_code}"
    )
    detail = response.json()["detail"]
    assert detail == "Failed to process the uploaded PDF.", (
        f"Expected a safe generic error detail, got: {detail}"
    )
    assert _temp_leftovers() == before, (
        "Temporary PDF files were left behind after failed ingestion."
    )
    return True


def test_duplicate_upload_is_idempotent() -> bool:
    """F2: Re-uploading the same PDF must not double the ChromaDB records."""
    assert REAL_PDF_PATH.exists(), f"Expected real PDF at {REAL_PDF_PATH}."
    client = TestClient(app)
    with open(REAL_PDF_PATH, "rb") as handle:
        pdf_bytes = handle.read()

    first = _upload_file(client, REAL_PDF_NAME, pdf_bytes)
    assert first.status_code == 200, f"First upload failed: {first.text}"

    collection = chromadb.PersistentClient(path=CHROMA_DIR).get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    count_after_first = collection.count()
    assert count_after_first > 0, "Expected records in the collection after indexing."

    second = _upload_file(client, REAL_PDF_NAME, pdf_bytes)
    assert second.status_code == 200, f"Second upload failed: {second.text}"

    count_after_second = collection.count()
    assert count_after_second == count_after_first, (
        f"Re-upload changed record count from {count_after_first} to "
        f"{count_after_second}; duplicates were created."
    )

    first_body = first.json()
    second_body = second.json()
    assert first_body["document_id"] == second_body["document_id"], (
        "Same file must map to the same deterministic document_id."
    )
    assert first_body["chunks"] == second_body["chunks"], (
        "Same file must produce the same chunk count."
    )
    print(
        f"Duplicate upload idempotent: {count_after_second} records "
        f"(unchanged after re-upload)."
    )
    return True


def test_docs_advertise_upload_endpoint() -> bool:
    """G: /docs and /openapi.json advertise POST /api/documents/upload."""
    client = TestClient(app)
    docs_response = client.get("/docs")
    openapi_response = client.get("/openapi.json")

    assert docs_response.status_code == 200, "Expected /docs to remain available."
    assert openapi_response.status_code == 200, "Expected /openapi.json to work."

    paths = openapi_response.json()["paths"]
    assert "/api/documents/upload" in paths, "Expected /api/documents/upload in OpenAPI paths."
    assert "post" in paths["/api/documents/upload"], "Expected upload endpoint to accept POST."

    schema_props = (
        openapi_response.json()["components"]["schemas"]
        ["UploadIngestionResponse"]["properties"]
    )
    expected_fields = {"document_id", "filename", "pages", "chunks", "status"}
    assert expected_fields.issubset(schema_props), (
        f"Expected typed ingestion fields in schema, got: {sorted(schema_props)}"
    )

    print("OpenAPI paths:", sorted(paths.keys()))
    return True


if __name__ == "__main__":
    tests = [
        test_valid_pdf_upload_ingests_and_indexes,
        test_txt_upload_rejected_without_ingestion,
        test_missing_file_rejected,
        test_blank_filename_rejected,
        test_empty_file_rejected,
        test_oversized_file_rejected,
        test_pdf_extension_with_fake_content_rejected,
        test_corrupt_pdf_returns_safe_error_and_cleans_temp,
        test_duplicate_upload_is_idempotent,
        test_docs_advertise_upload_endpoint,
    ]
    success = all(test() for test in tests)
    print("\nDocuments API tests complete." if success else "\nDocuments API tests FAILED.")
    sys.exit(0 if success else 1)

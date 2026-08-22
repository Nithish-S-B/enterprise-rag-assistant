"""Validation tests for POST /api/documents/upload (upload-only, no ingestion)."""
import os
import sys

from fastapi.testclient import TestClient

from app.main import app


MINIMAL_PDF_CONTENT = (
    b"%PDF-1.4\n"
    b"1 0 obj\n<< /Type /Catalog >>\nendobj\n"
    b"trailer\n<< /Root 1 0 R >>\n%%EOF\n"
)

UPLOAD_ENDPOINT = "/api/documents/upload"


def _upload_file(client: TestClient, filename: str, content: bytes,
                 content_type: str = "application/pdf"):
    return client.post(
        UPLOAD_ENDPOINT,
        files={"file": (filename, content, content_type)},
    )


def test_valid_pdf_upload_succeeds() -> bool:
    """A: A well-formed PDF passes validation and returns typed metadata."""
    client = TestClient(app)
    response = _upload_file(client, "EMPLOYEE HANDBOOK.pdf", MINIMAL_PDF_CONTENT)

    assert response.status_code == 200, (
        f"Expected HTTP 200, got {response.status_code}: {response.text}"
    )

    body = response.json()
    assert body["filename"] == "EMPLOYEE HANDBOOK.pdf", "Expected original filename echoed."
    assert body["size_bytes"] == len(MINIMAL_PDF_CONTENT), "Expected exact byte count."
    assert body["content_type"] == "application/pdf", "Expected declared content type."
    assert body["message"] == "PDF upload validated successfully.", "Expected success message."

    print("POST /api/documents/upload ->", body)
    return True


def test_txt_upload_rejected() -> bool:
    """B: Non-PDF extensions are rejected with 415 before any processing."""
    client = TestClient(app)
    response = _upload_file(client, "notes.txt", b"plain text content")

    assert response.status_code == 415, (
        f"Expected HTTP 415 for .txt upload, got {response.status_code}: {response.text}"
    )
    return True


def test_missing_file_rejected() -> bool:
    """C: A request with no file part is rejected by request validation."""
    client = TestClient(app)
    response = client.post(UPLOAD_ENDPOINT)

    assert response.status_code == 422, (
        f"Expected HTTP 422 for missing file, got {response.status_code}"
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


def test_docs_advertise_upload_endpoint() -> bool:
    """F: /docs and /openapi.json advertise POST /api/documents/upload."""
    client = TestClient(app)
    docs_response = client.get("/docs")
    openapi_response = client.get("/openapi.json")

    assert docs_response.status_code == 200, "Expected /docs to remain available."
    assert openapi_response.status_code == 200, "Expected /openapi.json to work."

    paths = openapi_response.json()["paths"]
    assert "/api/documents/upload" in paths, "Expected /api/documents/upload in OpenAPI paths."
    assert "post" in paths["/api/documents/upload"], "Expected upload endpoint to accept POST."

    print("OpenAPI paths:", sorted(paths.keys()))
    return True


if __name__ == "__main__":
    tests = [
        test_valid_pdf_upload_succeeds,
        test_txt_upload_rejected,
        test_missing_file_rejected,
        test_empty_file_rejected,
        test_pdf_extension_with_fake_content_rejected,
        test_docs_advertise_upload_endpoint,
        test_oversized_file_rejected,
    ]
    success = all(test() for test in tests)
    print("\nDocuments API tests complete." if success else "\nDocuments API tests FAILED.")
    sys.exit(0 if success else 1)

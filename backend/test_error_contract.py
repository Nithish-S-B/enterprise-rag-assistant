"""Tests for the global error-response contract (Step 7.10.1).

Every application-level error must use the uniform ErrorResponse shape::

    {
        "error_type": "...",
        "message": "...",
        "request_id": null
    }

These tests verify the contract across all endpoints and error categories
without calling the real LLM provider (provider calls are mocked).
"""
import sys
import unittest.mock

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _assert_error_shape(body: dict, expected_keys: frozenset[str] = frozenset({"error_type", "message", "request_id"})) -> None:
    """Verify the response body conforms to the ErrorResponse shape."""
    assert set(body) == expected_keys, (
        f"ErrorResponse must contain exactly {sorted(expected_keys)}, got: {sorted(body)}"
    )
    assert isinstance(body["error_type"], str) and body["error_type"], (
        f"error_type must be a non-empty string, got: {body['error_type']!r}"
    )
    assert isinstance(body["message"], str) and body["message"], (
        f"message must be a non-empty string, got: {body['message']!r}"
    )
    assert body["request_id"] is None, (
        f"request_id must be null in Step 7.10.1, got: {body['request_id']!r}"
    )


# ---------------------------------------------------------------------------
# Health endpoint must remain unaffected
# ---------------------------------------------------------------------------

def test_health_endpoint_unchanged() -> bool:
    """GET /api/health must continue to return 200 with the HealthResponse shape."""
    response = client.get("/api/health")
    assert response.status_code == 200, (
        f"Expected HTTP 200 from health, got {response.status_code}"
    )
    body = response.json()
    assert body["status"] == "ok", f"Expected status 'ok', got: {body.get('status')}"
    assert "error_type" not in body, "Health response must not contain error fields."
    print("Health endpoint unchanged: 200 ok")
    return True


# ---------------------------------------------------------------------------
# 422 - Validation errors (Pydantic / request validation)
# ---------------------------------------------------------------------------

def test_missing_chat_question_returns_422() -> bool:
    """POST /api/chat with missing question -> 422 + validation_error."""
    response = client.post("/api/chat", json={})
    assert response.status_code == 422
    body = response.json()
    _assert_error_shape(body)
    assert body["error_type"] == "validation_error"
    assert body["message"] == "Request validation failed."
    print("Missing question -> 422:", body)
    return True


def test_invalid_final_k_returns_422() -> bool:
    """POST /api/chat with final_k=0 -> 422 + validation_error."""
    response = client.post(
        "/api/chat",
        json={"question": "What is the policy?", "final_k": 0},
    )
    assert response.status_code == 422
    body = response.json()
    _assert_error_shape(body)
    assert body["error_type"] == "validation_error"
    print("Invalid final_k -> 422:", body)
    return True


def test_missing_upload_file_returns_422() -> bool:
    """POST /api/documents/upload with no file -> 422 + validation_error."""
    response = client.post("/api/documents/upload")
    assert response.status_code == 422
    body = response.json()
    _assert_error_shape(body)
    assert body["error_type"] == "validation_error"
    print("Missing upload file -> 422:", body)
    return True


def test_invalid_delete_id_returns_422() -> bool:
    """DELETE /api/documents/{id} with unsafe id -> 422 + validation_error."""
    response = client.delete("/api/documents/UPPER_CASE_ID")
    assert response.status_code == 422
    body = response.json()
    _assert_error_shape(body)
    assert body["error_type"] == "validation_error"
    print("Invalid delete id -> 422:", body)
    return True


# ---------------------------------------------------------------------------
# 404 - Not found
# ---------------------------------------------------------------------------

def test_nonexistent_document_returns_404() -> bool:
    """DELETE of a document_id that doesn't exist -> 404 + not_found."""
    response = client.delete("/api/documents/nonexistent_document_xyz")
    assert response.status_code == 404
    body = response.json()
    _assert_error_shape(body)
    assert body["error_type"] == "not_found"
    assert body["message"] == "Document not found."
    assert "nonexistent_document_xyz" not in response.text, (
        "User-supplied document_id must not be reflected in the response."
    )
    print("Nonexistent document -> 404:", body)
    return True


# ---------------------------------------------------------------------------
# 415 - Unsupported media type
# ---------------------------------------------------------------------------

def test_unsupported_upload_type_returns_415() -> bool:
    """Non-PDF upload -> 415 + unsupported_media_type."""
    response = client.post(
        "/api/documents/upload",
        files={"file": ("notes.txt", b"plain text", "text/plain")},
    )
    assert response.status_code == 415
    body = response.json()
    _assert_error_shape(body)
    assert body["error_type"] == "unsupported_media_type"
    print("Unsupported upload type -> 415:", body)
    return True


# ---------------------------------------------------------------------------
# 413 - Payload too large (maps to bad_request)
# ---------------------------------------------------------------------------

def test_oversized_upload_returns_413() -> bool:
    """Upload exceeding size limit -> 413 + bad_request."""
    import os
    previous_limit = os.environ.get("MAX_UPLOAD_SIZE_MB")
    os.environ["MAX_UPLOAD_SIZE_MB"] = "1"
    try:
        oversized = b"%PDF-1.4\n" + b"A" * (2 * 1024 * 1024)
        response = client.post(
            "/api/documents/upload",
            files={"file": ("large.pdf", oversized, "application/pdf")},
        )
        assert response.status_code == 413
        body = response.json()
        _assert_error_shape(body)
        assert body["error_type"] == "bad_request"
        print("Oversized upload -> 413:", body)
    finally:
        if previous_limit is None:
            os.environ.pop("MAX_UPLOAD_SIZE_MB", None)
        else:
            os.environ["MAX_UPLOAD_SIZE_MB"] = previous_limit
    return True


# ---------------------------------------------------------------------------
# 400 - Bad request
# ---------------------------------------------------------------------------

def test_empty_upload_returns_400() -> bool:
    """Zero-byte upload -> 400 + bad_request."""
    response = client.post(
        "/api/documents/upload",
        files={"file": ("empty.pdf", b"", "application/pdf")},
    )
    assert response.status_code == 400
    body = response.json()
    _assert_error_shape(body)
    assert body["error_type"] == "bad_request"
    print("Empty upload -> 400:", body)
    return True


# ---------------------------------------------------------------------------
# 503 - Provider unavailable
# ---------------------------------------------------------------------------

def test_provider_failure_returns_503() -> bool:
    """LLM provider failure during chat -> 503 + provider_unavailable."""
    with unittest.mock.patch(
        "app.rag.generate_text",
        side_effect=RuntimeError("OpenRouter request failed with HTTP status 502."),
    ):
        response = client.post(
            "/api/chat",
            json={"question": "What is the leave policy?"},
        )
    assert response.status_code == 503
    body = response.json()
    _assert_error_shape(body)
    assert body["error_type"] == "provider_unavailable"
    assert body["message"] == "Language model provider is currently unavailable."
    assert "OpenRouter" not in response.text, "Provider details must not leak."
    assert "502" not in response.text, "Upstream status must not leak."
    print("Provider failure -> 503:", body)
    return True


# ---------------------------------------------------------------------------
# 500 - Internal error (unexpected exception)
# ---------------------------------------------------------------------------

def test_unexpected_exception_returns_500() -> bool:
    """An unexpected bug in a route -> 500 + internal_error, no details leaked."""
    with unittest.mock.patch(
        "app.api.documents.list_documents",
        side_effect=TypeError("unexpected internal type error"),
    ):
        response = client.get("/api/documents")
    assert response.status_code == 500
    body = response.json()
    _assert_error_shape(body)
    assert body["error_type"] == "internal_error"
    assert body["message"] == "An internal server error occurred."
    assert "unexpected internal type error" not in response.text, (
        "Exception message must not leak to the client."
    )
    assert "TypeError" not in response.text, (
        "Exception class name must not leak to the client."
    )
    print("Unexpected exception -> 500:", body)
    return True


# ---------------------------------------------------------------------------
# 500 - Ingestion pipeline failure (documents)
# ---------------------------------------------------------------------------

def test_ingestion_failure_returns_500() -> bool:
    """Embedding failure during upload -> 500 + internal_error, no details leaked."""
    with unittest.mock.patch(
        "app.api.documents.embed_texts",
        side_effect=RuntimeError("embedding model exploded"),
    ):
        response = client.post(
            "/api/documents/upload",
            files={"file": ("test.pdf", b"%PDF-1.4\ndata", "application/pdf")},
        )
    assert response.status_code == 500
    body = response.json()
    _assert_error_shape(body)
    assert body["error_type"] == "internal_error"
    assert body["message"] == "An internal server error occurred."
    assert "embedding model exploded" not in response.text, (
        "Internal error details must not leak."
    )
    print("Ingestion failure -> 500:", body)
    return True


# ---------------------------------------------------------------------------
# No internal details leak in ANY error response
# ---------------------------------------------------------------------------

def test_no_exception_class_names_in_any_error() -> bool:
    """None of the tested error responses should contain Python exception
    class names (RuntimeError, ValueError, etc.)."""
    sensitive_strings = [
        "RuntimeError", "ValueError", "TypeError", "KeyError",
        "FileNotFoundError", "Exception",
        "Traceback", "traceback",
        "chroma_db", "chromadb",
        "OPENROUTER_API_KEY", "sk-or-v1",
        "/backend/", "D:\\",
    ]
    test_cases = [
        client.post("/api/chat", json={}),
        client.post("/api/chat", json={"question": "test", "final_k": 0}),
        client.delete("/api/documents/nonexistent_xyz"),
        client.post(
            "/api/documents/upload",
            files={"file": ("notes.txt", b"text", "text/plain")},
        ),
        client.post("/api/documents/upload"),
        client.get("/api/documents/../../etc/passwd"),
    ]
    for response in test_cases:
        if response.status_code < 400:
            continue
        body_text = response.text
        for sensitive in sensitive_strings:
            assert sensitive not in body_text, (
                f"Sensitive string '{sensitive}' leaked in response "
                f"({response.status_code}): {body_text[:200]}"
            )
    print("No internal details leaked in any error response.")
    return True


# ---------------------------------------------------------------------------
# Error type coverage - all 6 semantic types appear at least once
# ---------------------------------------------------------------------------

def test_all_error_types_represented() -> bool:
    """Every defined semantic error_type must appear in at least one test case."""
    error_types_seen: set[str] = set()

    def _collect(response):
        if response.status_code >= 400:
            body = response.json()
            if "error_type" in body:
                error_types_seen.add(body["error_type"])

    _collect(client.post("/api/chat", json={}))                              # validation_error
    _collect(client.post("/api/documents/upload"))                           # validation_error
    _collect(client.delete("/api/documents/nonexistent_xyz"))                # not_found
    _collect(client.post(                                                    # unsupported_media_type
        "/api/documents/upload",
        files={"file": ("x.txt", b"t", "text/plain")},
    ))
    _collect(client.post(                                                    # bad_request (empty)
        "/api/documents/upload",
        files={"file": ("empty.pdf", b"", "application/pdf")},
    ))

    with unittest.mock.patch("app.rag.generate_text", side_effect=RuntimeError("boom")):
        _collect(client.post("/api/chat", json={"question": "hi"}))          # provider_unavailable

    with unittest.mock.patch(
        "app.api.documents.embed_texts",
        side_effect=RuntimeError("boom"),
    ):
        _collect(client.post(                                                # internal_error
            "/api/documents/upload",
            files={"file": ("test.pdf", b"%PDF-1.4\ndata", "application/pdf")},
        ))

    expected = {
        "validation_error",
        "not_found",
        "unsupported_media_type",
        "bad_request",
        "provider_unavailable",
        "internal_error",
    }
    missing = expected - error_types_seen
    assert not missing, f"Missing error_type coverage: {sorted(missing)}"
    print(f"All error types represented: {sorted(error_types_seen)}")
    return True


# ---------------------------------------------------------------------------
# Provider failure does NOT leak upstream response body
# ---------------------------------------------------------------------------

def test_provider_failure_no_upstream_body_leak() -> bool:
    """The upstream OpenRouter error body must never appear in the response."""
    upstream_body = '{"error": {"message": "Rate limit exceeded", "code": 429}}'
    with unittest.mock.patch(
        "app.rag.generate_text",
        side_effect=RuntimeError(
            f"OpenRouter request failed with HTTP status 429: {upstream_body}"
        ),
    ):
        response = client.post(
            "/api/chat",
            json={"question": "What is the policy?"},
        )
    assert response.status_code == 503
    assert "Rate limit exceeded" not in response.text
    assert "429" not in response.text
    assert "upstream" not in response.text.lower()
    print("Provider failure: no upstream body leaked.")
    return True


if __name__ == "__main__":
    tests = [
        test_health_endpoint_unchanged,
        test_missing_chat_question_returns_422,
        test_invalid_final_k_returns_422,
        test_missing_upload_file_returns_422,
        test_invalid_delete_id_returns_422,
        test_nonexistent_document_returns_404,
        test_unsupported_upload_type_returns_415,
        test_oversized_upload_returns_413,
        test_empty_upload_returns_400,
        test_provider_failure_returns_503,
        test_unexpected_exception_returns_500,
        test_ingestion_failure_returns_500,
        test_no_exception_class_names_in_any_error,
        test_all_error_types_represented,
        test_provider_failure_no_upstream_body_leak,
    ]
    success = all(test() for test in tests)
    print(
        "\nError contract tests complete."
        if success
        else "\nError contract tests FAILED."
    )
    sys.exit(0 if success else 1)

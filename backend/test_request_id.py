"""Tests for the Request-ID middleware (Step 7.10.2).

Verifies that:
A. Normal request gets X-Request-ID response header.
B. Error response contains the same request_id.
C. Response header and JSON request_id match.
D. Client-supplied valid UUID4 is preserved.
E. Invalid supplied X-Request-ID is replaced with a generated UUID4.
F. Every request receives a unique ID when no ID is supplied.
G. Health endpoint still returns its existing response body.
H. Existing error-contract tests still pass (see test_error_contract.py).
I. Existing document/chat API tests still pass (see test_*_api.py).
"""
import sys
import uuid

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# A. Normal request gets X-Request-ID response header
# ---------------------------------------------------------------------------

def test_auto_generated_request_id_on_success() -> bool:
    """GET /api/health with no X-Request-ID header should get a server-generated UUID4."""
    response = client.get("/api/health")
    assert response.status_code == 200
    rid = response.headers.get("x-request-id")
    assert rid is not None, "X-Request-ID response header must be set."
    uuid.UUID(rid, version=4)  # raises ValueError if invalid
    print(f"A: Auto-generated request_id on success: {rid}")
    return True


def test_auto_generated_request_id_on_error() -> bool:
    """422 error response must include a server-generated UUID4 request_id."""
    response = client.post("/api/chat", json={})
    assert response.status_code == 422
    body = response.json()
    rid = body.get("request_id")
    assert rid is not None, "request_id must be present in error body."
    uuid.UUID(rid, version=4)
    print(f"A: Auto-generated request_id on error: {rid}")
    return True


# ---------------------------------------------------------------------------
# B. Error response contains the same request_id
# ---------------------------------------------------------------------------

def test_error_response_contains_request_id() -> bool:
    """Every error body must contain a non-null request_id string."""
    test_cases = [
        client.post("/api/chat", json={}),
        client.delete("/api/documents/nonexistent_xyz"),
        client.post(
            "/api/documents/upload",
            files={"file": ("notes.txt", b"text", "text/plain")},
        ),
    ]
    for response in test_cases:
        if response.status_code < 400:
            continue
        body = response.json()
        rid = body.get("request_id")
        assert rid is not None, (
            f"Error {response.status_code} missing request_id: {body}"
        )
        uuid.UUID(rid, version=4)
    print("B: Error responses contain valid request_id.")
    return True


# ---------------------------------------------------------------------------
# C. Response header and JSON request_id match
# ---------------------------------------------------------------------------

def test_header_and_body_request_id_match() -> bool:
    """On error responses, the X-Request-ID header and JSON request_id should match."""
    response = client.post("/api/chat", json={})
    assert response.status_code == 422
    header_rid = response.headers.get("x-request-id")
    body_rid = response.json().get("request_id")
    assert header_rid is not None, "X-Request-ID header must be set on errors."
    assert header_rid == body_rid, (
        f"Header {header_rid!r} != body {body_rid!r}"
    )
    print(f"C: Header and body match: {header_rid}")
    return True


# ---------------------------------------------------------------------------
# D. Client-supplied valid UUID4 is preserved
# ---------------------------------------------------------------------------

def test_client_supplied_valid_uuid_reused() -> bool:
    """A valid UUID4 X-Request-ID header should be reused in both the
    response header and the error body."""
    client_id = str(uuid.uuid4())
    response = client.post(
        "/api/chat",
        json={},
        headers={"X-Request-ID": client_id},
    )
    assert response.status_code == 422
    body = response.json()
    body_rid = body.get("request_id")
    header_rid = response.headers.get("x-request-id")
    assert body_rid == client_id, (
        f"Error body request_id {body_rid!r} must match client header {client_id!r}"
    )
    assert header_rid == client_id, (
        f"Response header {header_rid!r} must match client header {client_id!r}"
    )
    print(f"D: Client-supplied UUID reused: {client_id}")
    return True


# ---------------------------------------------------------------------------
# E. Invalid supplied X-Request-ID is replaced with a generated UUID4
# ---------------------------------------------------------------------------

def test_invalid_header_ignored_generates_new_id() -> bool:
    """A non-UUID X-Request-ID header should be ignored; server generates its own."""
    response = client.post(
        "/api/chat",
        json={},
        headers={"X-Request-ID": "not-a-valid-uuid"},
    )
    assert response.status_code == 422
    body = response.json()
    rid = body.get("request_id")
    assert rid is not None
    uuid.UUID(rid, version=4)
    assert rid != "not-a-valid-uuid", "Invalid header must not be propagated."
    print(f"E: Invalid header ignored, new ID: {rid}")
    return True


# ---------------------------------------------------------------------------
# F. Every request receives a unique ID when no ID is supplied
# ---------------------------------------------------------------------------

def test_different_requests_get_different_ids() -> bool:
    """Each request must receive a unique request_id."""
    r1 = client.get("/api/health")
    r2 = client.get("/api/health")
    rid1 = r1.headers.get("x-request-id")
    rid2 = r2.headers.get("x-request-id")
    assert rid1 != rid2, f"Request IDs must be unique, got {rid1} twice"
    print(f"F: Unique IDs: {rid1} != {rid2}")
    return True


# ---------------------------------------------------------------------------
# G. Health endpoint still returns its existing response body
# ---------------------------------------------------------------------------

def test_health_endpoint_body_unchanged() -> bool:
    """GET /api/health must still return 200 with the original HealthResponse shape."""
    response = client.get("/api/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok", f"Expected status 'ok', got: {body.get('status')}"
    assert "error_type" not in body, "Health response must not contain error fields."
    # Plus it should now have X-Request-ID header
    assert response.headers.get("x-request-id") is not None
    print("G: Health endpoint body unchanged, header present.")
    return True


# ---------------------------------------------------------------------------
# Truncated UUID header is also invalid
# ---------------------------------------------------------------------------

def test_truncated_uuid_header_ignored() -> bool:
    """A truncated UUID should be treated as invalid and a new ID generated."""
    truncated = "550e8400-e29b-41d4-a716-44665544"  # missing last chars
    response = client.post(
        "/api/chat",
        json={},
        headers={"X-Request-ID": truncated},
    )
    assert response.status_code == 422
    body = response.json()
    rid = body["request_id"]
    assert rid != truncated, "Truncated header must not be reused."
    uuid.UUID(rid, version=4)
    print(f"Truncated header ignored, new ID: {rid}")
    return True


# ---------------------------------------------------------------------------
# Empty header treated as missing
# ---------------------------------------------------------------------------

def test_empty_header_generates_new_id() -> bool:
    """An empty X-Request-ID header should be treated as missing."""
    response = client.post(
        "/api/chat",
        json={},
        headers={"X-Request-ID": "   "},
    )
    assert response.status_code == 422
    body = response.json()
    rid = body.get("request_id")
    assert rid is not None
    uuid.UUID(rid, version=4)
    print(f"Empty header generates new ID: {rid}")
    return True


if __name__ == "__main__":
    tests = [
        test_auto_generated_request_id_on_success,
        test_auto_generated_request_id_on_error,
        test_error_response_contains_request_id,
        test_header_and_body_request_id_match,
        test_client_supplied_valid_uuid_reused,
        test_invalid_header_ignored_generates_new_id,
        test_different_requests_get_different_ids,
        test_health_endpoint_body_unchanged,
        test_truncated_uuid_header_ignored,
        test_empty_header_generates_new_id,
    ]
    success = all(test() for test in tests)
    print(
        "\nRequest ID middleware tests complete."
        if success
        else "\nRequest ID middleware tests FAILED."
    )
    sys.exit(0 if success else 1)

"""Tests for Phase 7.10.4: Configurable CORS + Readiness.

Verifies:
A. GET /api/health remains 200 with unchanged body.
B. GET /api/ready returns 200 when dependencies are available.
C. Readiness failure maps through the global ErrorResponse contract.
D. CORS allows configured origins (localhost:3000, localhost:5173).
E. CORS rejects an arbitrary origin.
F. Request-ID middleware still adds X-Request-ID.
G. Request logging still logs readiness requests.
H. Existing tests continue to pass (verified separately).
"""
import os
import sys
import unittest.mock
import uuid

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app, raise_server_exceptions=False)

_LOG_TARGET = "app.middleware.request_logging.logger"


# ---------------------------------------------------------------------------
# A. GET /api/health remains 200 and its response body is unchanged
# ---------------------------------------------------------------------------

def test_health_unchanged() -> bool:
    """GET /api/health must still return 200 with the original HealthResponse shape."""
    response = client.get("/api/health")
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    body = response.json()
    assert body["status"] == "ok", f"Expected status 'ok', got: {body.get('status')}"
    assert body["service"] == "enterprise-rag-assistant"
    assert "error_type" not in body
    print("A: GET /api/health unchanged: 200, status='ok'.")
    return True


# ---------------------------------------------------------------------------
# B. GET /api/ready returns 200 when local dependencies are available
# ---------------------------------------------------------------------------

def test_ready_returns_200() -> bool:
    """GET /api/ready must return 200 with status='ready'."""
    response = client.get("/api/ready")
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    body = response.json()
    assert body["status"] == "ready", f"Expected status 'ready', got: {body.get('status')}"
    print("B: GET /api/ready returns 200: status='ready'.")
    return True


# ---------------------------------------------------------------------------
# C. Readiness failure maps through the global ErrorResponse contract
# ---------------------------------------------------------------------------

def test_ready_503_when_vector_store_unavailable() -> bool:
    """If ChromaDB is unreachable, /api/ready returns 503 + ErrorResponse shape."""
    with unittest.mock.patch(
        "app.vector_store._collection"
    ) as mock_vs:
        mock_vs.count.side_effect = RuntimeError("ChromaDB crashed")
        response = client.get("/api/ready")

    assert response.status_code == 503, f"Expected 503, got {response.status_code}"
    body = response.json()
    assert body["error_type"] == "provider_unavailable", f"Unexpected error_type: {body['error_type']}"
    assert body["message"] == "Vector store is not available.", f"Unexpected message: {body['message']}"
    assert isinstance(body["request_id"], str) and body["request_id"]
    print("C: Readiness failure -> 503 + ErrorResponse contract.")
    return True


def test_ready_503_when_embedding_model_unavailable() -> bool:
    """If embedding model is None, /api/ready returns 503."""
    with unittest.mock.patch("app.embeddings._model", None):
        response = client.get("/api/ready")
    assert response.status_code == 503, f"Expected 503, got {response.status_code}"
    body = response.json()
    assert body["error_type"] == "provider_unavailable", f"Unexpected error_type: {body['error_type']}"
    assert body["message"] == "Embedding model is not available.", f"Unexpected message: {body['message']}"
    assert isinstance(body["request_id"], str) and body["request_id"]
    print("C2: Embedding model unavailable -> 503.")
    return True


def test_ready_503_when_config_missing() -> bool:
    """If OPENROUTER_MODEL is unset, /api/ready returns 503."""
    previous = os.environ.pop("OPENROUTER_MODEL", None)
    try:
        response = client.get("/api/ready")
        assert response.status_code == 503, f"Expected 503, got {response.status_code}"
        body = response.json()
        assert body["error_type"] == "provider_unavailable", f"Unexpected error_type: {body['error_type']}"
        assert body["message"] == "Required configuration is missing.", f"Unexpected message: {body['message']}"
        assert isinstance(body["request_id"], str) and body["request_id"]
        print("C3: Missing config -> 503 + ErrorResponse contract.")
    finally:
        if previous is not None:
            os.environ["OPENROUTER_MODEL"] = previous
    return True


# ---------------------------------------------------------------------------
# D. CORS allows configured origins
# ---------------------------------------------------------------------------

def test_cors_allows_localhost_3000() -> bool:
    """Origin http://localhost:3000 must be allowed by CORS."""
    response = client.options(
        "/api/health",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "GET",
        },
    )
    allowed = response.headers.get("access-control-allow-origin")
    assert allowed == "http://localhost:3000", (
        f"Expected Allow-Origin=http://localhost:3000, got {allowed!r}"
    )
    print("D: CORS allows http://localhost:3000.")
    return True


def test_cors_allows_localhost_5173() -> bool:
    """Origin http://localhost:5173 must be allowed by CORS."""
    response = client.options(
        "/api/health",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "GET",
        },
    )
    allowed = response.headers.get("access-control-allow-origin")
    assert allowed == "http://localhost:5173", (
        f"Expected Allow-Origin=http://localhost:5173, got {allowed!r}"
    )
    print("D2: CORS allows http://localhost:5173.")
    return True


# ---------------------------------------------------------------------------
# E. CORS rejects an arbitrary origin
# ---------------------------------------------------------------------------

def test_cors_rejects_arbitrary_origin() -> bool:
    """An unconfigured origin must NOT receive Access-Control-Allow-Origin."""
    response = client.options(
        "/api/health",
        headers={
            "Origin": "https://evil.example.com",
            "Access-Control-Request-Method": "GET",
        },
    )
    allowed = response.headers.get("access-control-allow-origin")
    assert allowed is None, (
        f"Arbitrary origin must be rejected, got Allow-Origin={allowed!r}"
    )
    print("E: CORS rejects https://evil.example.com.")
    return True


# ---------------------------------------------------------------------------
# F. Request-ID middleware still adds X-Request-ID
# ---------------------------------------------------------------------------

def test_request_id_header_on_ready() -> bool:
    """GET /api/ready must still have X-Request-ID response header."""
    client_id = str(uuid.uuid4())
    response = client.get(
        "/api/ready",
        headers={"X-Request-ID": client_id},
    )
    assert response.status_code == 200
    header_rid = response.headers.get("x-request-id")
    assert header_rid == client_id, (
        f"Response header {header_rid!r} must match client header {client_id!r}"
    )
    print(f"F: X-Request-ID echoed on /api/ready: {client_id}")
    return True


# ---------------------------------------------------------------------------
# G. Request logging still logs readiness requests
# ---------------------------------------------------------------------------

def test_ready_request_logged() -> bool:
    """GET /api/ready must produce a log line from the logging middleware."""
    with unittest.mock.patch(_LOG_TARGET) as mock_logger:
        mock_logger.info = unittest.mock.MagicMock()
        response = client.get("/api/ready")
        assert response.status_code == 200
        assert mock_logger.info.call_count == 1, (
            f"Expected 1 log call, got {mock_logger.info.call_count}"
        )
        args, _ = mock_logger.info.call_args
        # args = (format, method, path, status_code, duration_ms, request_id)
        assert args[1] == "GET", f"Expected method GET, got {args[1]}"
        assert args[2] == "/api/ready", f"Expected path /api/ready, got {args[2]}"
        assert args[3] == 200, f"Expected status 200, got {args[3]}"
        print("G: /api/ready request is logged by request logging middleware.")
    return True


# ---------------------------------------------------------------------------
# Ready endpoint body shape
# ---------------------------------------------------------------------------

def test_ready_response_shape() -> bool:
    """GET /api/ready must return exactly {"status": "ready"}."""
    response = client.get("/api/ready")
    assert response.status_code == 200
    body = response.json()
    assert set(body.keys()) == {"status"}, f"Unexpected keys: {set(body.keys())}"
    assert body["status"] == "ready"
    print("Ready response shape: {'status': 'ready'}.")
    return True


if __name__ == "__main__":
    tests = [
        test_health_unchanged,
        test_ready_returns_200,
        test_ready_503_when_vector_store_unavailable,
        test_ready_503_when_embedding_model_unavailable,
        test_ready_503_when_config_missing,
        test_cors_allows_localhost_3000,
        test_cors_allows_localhost_5173,
        test_cors_rejects_arbitrary_origin,
        test_request_id_header_on_ready,
        test_ready_request_logged,
        test_ready_response_shape,
    ]
    success = all(test() for test in tests)
    print(
        "\nPhase 7.10.4 tests complete."
        if success
        else "\nPhase 7.10.4 tests FAILED."
    )
    sys.exit(0 if success else 1)

"""Tests for the request logging middleware (Step 7.10.3).

Verifies that:
A. GET /api/health returns 200.
B. A request generates a log line containing method/path, status, duration, request_id.
C. A successful request is logged.
D. An error request is still logged with its status.
E. request_id in the log matches the request's X-Request-ID header where appropriate.
F. Response body remains unchanged.
G. Existing request-ID behavior is unchanged.
"""
import sys
import unittest.mock
import uuid

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app, raise_server_exceptions=False)

_LOG_TARGET = "app.middleware.request_logging.logger"


# ---------------------------------------------------------------------------
# A. GET /api/health returns 200
# ---------------------------------------------------------------------------

def test_health_returns_200() -> bool:
    """GET /api/health must still return HTTP 200."""
    response = client.get("/api/health")
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    print("A: GET /api/health returns 200.")
    return True


# ---------------------------------------------------------------------------
# B. A request generates a log line containing method/path, status, duration, request_id
# ---------------------------------------------------------------------------

def test_log_contains_expected_fields() -> bool:
    """A request must produce a log line with method, path, status, duration, request_id."""
    with unittest.mock.patch(_LOG_TARGET) as mock_logger:
        mock_logger.info = unittest.mock.MagicMock()
        response = client.get("/api/health")
        assert response.status_code == 200

        mock_logger.info.assert_called_once()
        args, _ = mock_logger.info.call_args
        log_format = args[0]
        log_args = args[1:]

        # The format string should contain the expected placeholders
        assert "duration_ms" in log_format, f"Log format missing duration_ms: {log_format}"
        assert "request_id" in log_format, f"Log format missing request_id: {log_format}"
        assert "->" in log_format, f"Log format missing arrow separator: {log_format}"

        # Check that positional args are filled correctly
        assert log_args[0] == "GET", f"Expected method GET, got {log_args[0]}"
        assert log_args[1] == "/api/health", f"Expected path /api/health, got {log_args[1]}"
        assert isinstance(log_args[2], int), f"Expected int status, got {type(log_args[2])}"
        assert isinstance(log_args[3], float), f"Expected float duration_ms, got {type(log_args[3])}"
        assert isinstance(log_args[4], str), f"Expected str request_id, got {type(log_args[4])}"

        print("B: Log contains expected fields: method, path, status, duration, request_id.")
        return True


# ---------------------------------------------------------------------------
# C. A successful request is logged
# ---------------------------------------------------------------------------

def test_successful_request_logged() -> bool:
    """A 200 response must produce exactly one info log call."""
    with unittest.mock.patch(_LOG_TARGET) as mock_logger:
        mock_logger.info = unittest.mock.MagicMock()
        response = client.get("/api/health")
        assert response.status_code == 200
        assert mock_logger.info.call_count == 1, (
            f"Expected 1 log call, got {mock_logger.info.call_count}"
        )
        args, _ = mock_logger.info.call_args
        # args = (format, method, path, status_code, duration_ms, request_id)
        status_code = args[3]
        assert status_code == 200, f"Expected logged status 200, got {status_code}"
        print("C: Successful request is logged with status 200.")
        return True


# ---------------------------------------------------------------------------
# D. An error request is still logged with its status
# ---------------------------------------------------------------------------

def test_error_request_logged_with_status() -> bool:
    """A 422 validation error must still produce a log line with status 422."""
    with unittest.mock.patch(_LOG_TARGET) as mock_logger:
        mock_logger.info = unittest.mock.MagicMock()
        response = client.post("/api/chat", json={})
        assert response.status_code == 422
        assert mock_logger.info.call_count == 1, (
            f"Expected 1 log call, got {mock_logger.info.call_count}"
        )
        args, _ = mock_logger.info.call_args
        # args = (format, method, path, status_code, duration_ms, request_id)
        status_code = args[3]
        assert status_code == 422, f"Expected logged status 422, got {status_code}"
        print("D: Error request (422) is logged with status 422.")
        return True


# ---------------------------------------------------------------------------
# E. request_id in the log matches the request's X-Request-ID header
# ---------------------------------------------------------------------------

def test_log_request_id_matches_header() -> bool:
    """When a client supplies a valid X-Request-ID, the log must use that same ID."""
    client_id = str(uuid.uuid4())
    with unittest.mock.patch(_LOG_TARGET) as mock_logger:
        mock_logger.info = unittest.mock.MagicMock()
        response = client.get(
            "/api/health",
            headers={"X-Request-ID": client_id},
        )
        assert response.status_code == 200
        args, _ = mock_logger.info.call_args
        # args = (format, method, path, status_code, duration_ms, request_id)
        logged_request_id = args[5]
        assert logged_request_id == client_id, (
            f"Logged request_id {logged_request_id!r} != client header {client_id!r}"
        )
        print(f"E: Log request_id matches client header: {client_id}")
        return True


# ---------------------------------------------------------------------------
# E2. Auto-generated request_id is a valid UUID in the log
# ---------------------------------------------------------------------------

def test_log_request_id_is_valid_uuid() -> bool:
    """When no client ID is supplied, the log must contain a server-generated UUID4."""
    with unittest.mock.patch(_LOG_TARGET) as mock_logger:
        mock_logger.info = unittest.mock.MagicMock()
        response = client.get("/api/health")
        assert response.status_code == 200
        args, _ = mock_logger.info.call_args
        # args = (format, method, path, status_code, duration_ms, request_id)
        logged_request_id = args[5]
        assert logged_request_id != "-", "request_id must not be the fallback '-' on normal requests."
        uuid.UUID(logged_request_id, version=4)
        print(f"E2: Log request_id is a valid UUID4: {logged_request_id}")
        return True


# ---------------------------------------------------------------------------
# F. Response body remains unchanged
# ---------------------------------------------------------------------------

def test_response_body_unchanged() -> bool:
    """Logging middleware must not alter the response body."""
    with unittest.mock.patch(_LOG_TARGET):
        response = client.get("/api/health")
        body = response.json()
        assert body["status"] == "ok", f"Expected status 'ok', got: {body.get('status')}"
        print("F: Response body unchanged after logging middleware.")
        return True


# ---------------------------------------------------------------------------
# G. Existing request-ID behavior is unchanged
# ---------------------------------------------------------------------------

def test_request_id_header_echoed() -> bool:
    """The X-Request-ID response header must still be set by RequestIDMiddleware."""
    client_id = str(uuid.uuid4())
    with unittest.mock.patch(_LOG_TARGET):
        response = client.get(
            "/api/health",
            headers={"X-Request-ID": client_id},
        )
        assert response.status_code == 200
        header_rid = response.headers.get("x-request-id")
        assert header_rid == client_id, (
            f"Response header {header_rid!r} must match client header {client_id!r}"
        )
        print("G: Request-ID header echoed correctly through both middlewares.")
        return True


# ---------------------------------------------------------------------------
# H. Duration is positive
# ---------------------------------------------------------------------------

def test_duration_is_positive() -> bool:
    """The logged duration_ms must be a non-negative float."""
    with unittest.mock.patch(_LOG_TARGET) as mock_logger:
        mock_logger.info = unittest.mock.MagicMock()
        response = client.get("/api/health")
        assert response.status_code == 200
        args, _ = mock_logger.info.call_args
        # args = (format, method, path, status_code, duration_ms, request_id)
        duration_ms = args[4]
        assert isinstance(duration_ms, float), f"Expected float, got {type(duration_ms)}"
        assert duration_ms >= 0, f"Duration must be >= 0, got {duration_ms}"
        print(f"H: Duration is positive: {duration_ms} ms")
        return True


if __name__ == "__main__":
    tests = [
        test_health_returns_200,
        test_log_contains_expected_fields,
        test_successful_request_logged,
        test_error_request_logged_with_status,
        test_log_request_id_matches_header,
        test_log_request_id_is_valid_uuid,
        test_response_body_unchanged,
        test_request_id_header_echoed,
        test_duration_is_positive,
    ]
    success = all(test() for test in tests)
    print(
        "\nRequest logging middleware tests complete."
        if success
        else "\nRequest logging middleware tests FAILED."
    )
    sys.exit(0 if success else 1)

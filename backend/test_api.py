import sys

from fastapi.testclient import TestClient

from app.main import app


def test_health() -> bool:
    """Verify GET /api/health responds without loading the RAG stack."""
    client = TestClient(app)
    response = client.get("/api/health")

    assert response.status_code == 200, "Expected HTTP 200 from GET /api/health."

    body = response.json()
    assert body["status"] == "ok", "Expected status 'ok'."
    assert body.get("service"), "Expected the service field to be present."

    print("GET /api/health ->", body)
    print("API health test complete.")
    return True


if __name__ == "__main__":
    success = test_health()
    sys.exit(0 if success else 1)

import sys

from fastapi.testclient import TestClient

from app.main import app


def test_chat_answers_policy_question() -> bool:
    """End-to-end: real retrieval, reranking, and generation through the API."""
    client = TestClient(app)
    response = client.post(
        "/api/chat",
        json={"question": "Who approves leave requests?"},
    )

    assert response.status_code == 200, (
        f"Expected HTTP 200, got {response.status_code}: {response.text}"
    )

    body = response.json()
    assert body["answer"].strip(), "Expected a non-empty generated answer."
    assert isinstance(body["citations"], list), "Expected citations to be a list."
    assert body["citations"], "Expected at least one citation."
    for citation in body["citations"]:
        assert "citation_id" in citation, "Expected citation_id on each citation."
        assert "source" in citation, "Expected source on each citation."

    print("POST /api/chat -> answer:", body["answer"][:200])
    print("CITATIONS:")
    for citation in body["citations"]:
        print(f"- {citation['citation_id']} | {citation['source']}")
    return True


def test_chat_rejects_missing_question() -> bool:
    """Validation failures are rejected before the RAG pipeline is reached."""
    client = TestClient(app)
    response = client.post("/api/chat", json={})

    assert response.status_code == 422, (
        f"Expected HTTP 422 for missing question, got {response.status_code}"
    )
    return True


def test_chat_rejects_whitespace_question() -> bool:
    client = TestClient(app)
    response = client.post("/api/chat", json={"question": "   "})

    assert response.status_code == 422, (
        f"Expected HTTP 422 for whitespace-only question, got {response.status_code}"
    )
    return True


def test_chat_rejects_invalid_final_k() -> bool:
    client = TestClient(app)
    response = client.post(
        "/api/chat",
        json={"question": "Who approves leave requests?", "final_k": 0},
    )

    assert response.status_code == 422, (
        f"Expected HTTP 422 for final_k=0, got {response.status_code}"
    )
    return True


def test_docs_advertise_chat_endpoint() -> bool:
    client = TestClient(app)
    docs_response = client.get("/docs")
    openapi_response = client.get("/openapi.json")

    assert docs_response.status_code == 200, "Expected /docs to remain available."
    assert openapi_response.status_code == 200, "Expected /openapi.json to work."

    paths = openapi_response.json()["paths"]
    assert "/api/chat" in paths, "Expected POST /api/chat in OpenAPI paths."
    assert "post" in paths["/api/chat"], "Expected /api/chat to accept POST."

    print("OpenAPI paths:", sorted(paths.keys()))
    return True


if __name__ == "__main__":
    tests = [
        test_chat_rejects_missing_question,
        test_chat_rejects_whitespace_question,
        test_chat_rejects_invalid_final_k,
        test_docs_advertise_chat_endpoint,
        test_chat_answers_policy_question,
    ]
    success = all(test() for test in tests)
    print("\nChat API tests complete." if success else "\nChat API tests FAILED.")
    sys.exit(0 if success else 1)

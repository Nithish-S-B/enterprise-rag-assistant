import os
import sys

# Ensure the backend directory is on the import path when run from the project root.
backend_dir = os.path.dirname(os.path.abspath(__file__))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

import chromadb

from app.citations import build_citations
from app.rag import _format_context, answer_question
from app.reranker import rerank
from app.vector_store import CHROMA_DIR, COLLECTION_NAME, search


QUESTION = "Who approves leave requests?"
EXPECTED_DOCUMENTS = {
    "EMPLOYEE HANDBOOK.pdf",
    "EMPLOYEE LEAVE OF ABSENCE POLICY.pdf",
    "EMPLOYEE REMOTE WORK POLICY.pdf",
}


def test_citations() -> bool:
    """Verify read-only source-level citation construction over the RAG pipeline."""
    collection = chromadb.PersistentClient(path=CHROMA_DIR).get_collection(COLLECTION_NAME)
    indexed_sources = {
        os.path.basename(metadata["source"])
        for metadata in collection.get(include=["metadatas"])["metadatas"]
    }
    assert EXPECTED_DOCUMENTS <= indexed_sources, "Expected indexed policy PDFs are missing."

    result = answer_question(QUESTION, candidate_k=10, final_k=4)
    answer = result["answer"]
    citations = result["citations"]

    assert answer.strip(), "Expected a non-empty generated answer."
    assert citations, "Expected citations."
    assert [citation["citation_id"] for citation in citations] == [
        f"S{index}" for index in range(1, len(citations) + 1)
    ], "Citation IDs must be sequential starting at S1."

    for citation in citations:
        assert set(citation) == {
            "citation_id", "source", "page", "page_label", "chunk_id"
        }, "Citation has an unexpected structure."
        assert citation["source"] == os.path.basename(citation["source"]), (
            "Citation source must be a filename, not an absolute path."
        )
        assert citation["page_label"], "Citation must have a user-facing page label."

    fallback_citation = build_citations([
        {
            "id": "example#page0#chunk0",
            "metadata": {
                "source": r"D:\example\documents\EXAMPLE POLICY.pdf",
                "page": 0,
            },
        }
    ])[0]
    assert fallback_citation["source"] == "EXAMPLE POLICY.pdf"
    assert fallback_citation["page"] == 0
    assert fallback_citation["page_label"] == "1"

    final_chunks = rerank(QUESTION, search(QUESTION, top_k=10), final_k=4)
    sample_context = _format_context(final_chunks, build_citations(final_chunks))

    print("ANSWER:")
    print(answer)
    print("\nCITATIONS:")
    for citation in citations:
        print(citation)
    print("\nSAMPLE CONTEXT BLOCK:")
    print(sample_context.split("\n\n[CONTEXT", maxsplit=1)[0])
    return True


if __name__ == "__main__":
    success = test_citations()
    sys.exit(0 if success else 1)

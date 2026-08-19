import os
import sys

# Ensure backend directory is in the python path
backend_dir = os.path.dirname(os.path.abspath(__file__))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from app.vector_store import search


def test_retrieval_quality():
    """
    Diagnostic test: inspect top-10 retrieval results per question
    to assess whether the answer-bearing chunk is being surfaced.
    Read-only; no ranking changes or re-indexing.
    """
    questions = [
        "How many days can employees work remotely?",
        "How many annual leave days do employees get?",
        "Who approves leave requests?",
    ]
    top_k = 10

    for question in questions:
        print("\n" + "=" * 70)
        print(f"QUESTION: {question}")
        print("=" * 70)

        results = search(question, top_k=top_k)
        assert len(results) == top_k, (
            f"Expected {top_k} results, got {len(results)}."
        )

        for rank, record in enumerate(results, start=1):
            metadata = record["metadata"]
            source = os.path.basename(metadata.get("source", "Unknown"))
            print(f"\n  Rank {rank:2d} | distance={record['distance']:.4f}")
            print(f"    source={source} | page={metadata.get('page', -1)}")
            print(f"    id={record['id']}")
            print(f"    text={record['text'][:300]!r}")

        assert all(r["text"] for r in results), "Encountered an empty chunk."

    print("\n" + "=" * 70)
    print("Diagnostic inspection complete.")
    return True


if __name__ == "__main__":
    success = test_retrieval_quality()
    sys.exit(0 if success else 1)

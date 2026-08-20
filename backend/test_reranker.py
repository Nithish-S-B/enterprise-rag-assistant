import os
import sys

# Ensure the backend directory is on the import path when run from the project root.
backend_dir = os.path.dirname(os.path.abspath(__file__))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from app.reranker import rerank
from app.vector_store import search


CANDIDATE_K = 10
FINAL_K = 4
QUERIES = [
    "How many annual leave days do employees get?",
    "Who approves leave requests?",
]


def print_records(records: list[dict], ranking_name: str) -> None:
    """Print a ranking while retaining dense and reranker score visibility."""
    print(f"\n{ranking_name}")
    print("-" * 80)

    for rank, record in enumerate(records, start=1):
        metadata = record["metadata"]
        source = os.path.basename(metadata.get("source", "Unknown"))
        score = record.get("reranker_score")
        score_text = f"{score:.4f}" if score is not None else "not scored"

        print(f"Rank: {rank}")
        print(f"Dense distance: {record['distance']:.4f}")
        print(f"Reranker score: {score_text}")
        print(f"Source: {source}")
        print(f"Page: {metadata.get('page', -1)}")
        print(f"Chunk ID: {record['id']}")
        print(f"Text: {record['text'][:250]!r}")
        print()


def test_reranker() -> bool:
    """Read-only comparison of dense retrieval and cross-encoder reranking."""
    for query in QUERIES:
        print("\n" + "=" * 80)
        print(f"QUERY: {query}")
        print("=" * 80)

        dense_candidates = search(query, top_k=CANDIDATE_K)
        assert len(dense_candidates) == CANDIDATE_K, (
            f"Expected {CANDIDATE_K} dense candidates, got {len(dense_candidates)}."
        )
        print_records(dense_candidates, "Dense retrieval ranking")

        reranked_results = rerank(query, dense_candidates, final_k=FINAL_K)
        assert len(reranked_results) == FINAL_K, (
            f"Expected {FINAL_K} reranked results, got {len(reranked_results)}."
        )
        assert all("reranker_score" in record for record in reranked_results), (
            "Reranked result is missing its reranker score."
        )
        print_records(reranked_results, "Cross-encoder reranked top results")

    print("=" * 80)
    print("Reranker test complete.")
    return True


if __name__ == "__main__":
    success = test_reranker()
    sys.exit(0 if success else 1)

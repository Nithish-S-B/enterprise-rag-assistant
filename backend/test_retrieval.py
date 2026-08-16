import os
import sys

# Ensure backend directory is in the python path
backend_dir = os.path.dirname(os.path.abspath(__file__))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

import chromadb

from app.vector_store import CHROMA_DIR, COLLECTION_NAME, search


def test_retrieval():
    """
    Test semantic retrieval against the persistent ChromaDB collection
    with three queries, verifying count, metadata, and content.
    """
    # Open the existing persistent collection (read-only).
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_collection(COLLECTION_NAME)
    print(f"Collection '{COLLECTION_NAME}' contains {collection.count()} records.")
    print("=" * 60)

    queries = [
        "How many days can employees work remotely?",
        "How many annual leave days do employees get?",
        "Who approves leave requests?",
    ]
    top_k = 4

    for query in queries:
        print(f"\nQUERY: {query}")
        print("-" * 60)

        results = search(query, top_k=top_k)

        # Verify count
        assert len(results) == top_k, (
            f"Expected {top_k} results for query, got {len(results)}."
        )

        for rank, record in enumerate(results, start=1):
            metadata = record["metadata"]
            text = record["text"]
            print(f"  Rank {rank} | distance={record['distance']:.4f} | "
                  f"source={metadata.get('source', 'Unknown')} | "
                  f"page={metadata.get('page', -1)}")
            print(f"    Preview: {text[:120]}...")

            # Verify each record
            assert record["id"], "Record has no ID."
            assert text, "Record has empty document text."
            assert "source" in metadata, "Record is missing 'source' metadata."
            assert "page" in metadata, "Record is missing 'page' metadata."

        print("-" * 60)

    print("=" * 60)
    print("All checks passed! Retrieval verified for all queries.")
    return True


if __name__ == "__main__":
    success = test_retrieval()
    sys.exit(0 if success else 1)

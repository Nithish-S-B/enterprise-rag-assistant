import os
import sys

# Ensure the backend directory is on the import path when run from the project root.
backend_dir = os.path.dirname(os.path.abspath(__file__))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from app.vector_store import search


QUERY_SETS = [
    {
        "topic": "Annual leave entitlement",
        "queries": [
            ("Original", "How many annual leave days do employees get?"),
            (
                "Variant 1",
                "According to the employee leave policy, what is the annual leave entitlement in days for employees?",
            ),
            (
                "Variant 2",
                "What does the leave policy state about the number of annual paid leave days available to an employee?",
            ),
            (
                "Variant 3",
                "In the annual leave policy, how many days of annual leave are employees entitled to each year?",
            ),
        ],
    },
    {
        "topic": "Leave-request approval",
        "queries": [
            ("Original", "Who approves leave requests?"),
            (
                "Variant 1",
                "According to the employee leave policy, which person or role is responsible for approving employee leave requests?",
            ),
            (
                "Variant 2",
                "Under the leave approval process, who must approve an employee's request for leave?",
            ),
            (
                "Variant 3",
                "What role is identified in the leave policy as the approver of staff leave requests?",
            ),
        ],
    },
]


def run_query_variant_test():
    """Read-only comparison of dense-retrieval results for manual query variants."""
    top_k = 10

    for query_set in QUERY_SETS:
        print("\n" + "=" * 80)
        print(f"TOPIC: {query_set['topic']}")
        print("=" * 80)

        for label, query in query_set["queries"]:
            print(f"\n{label} query: {query}")
            print("-" * 80)

            results = search(query, top_k=top_k)
            assert len(results) == top_k, (
                f"Expected {top_k} results for {label}, got {len(results)}."
            )

            for rank, record in enumerate(results, start=1):
                metadata = record["metadata"]
                source = os.path.basename(metadata.get("source", "Unknown"))
                print(f"Rank: {rank}")
                print(f"Distance: {record['distance']:.4f}")
                print(f"Source: {source}")
                print(f"Page: {metadata.get('page', -1)}")
                print(f"Chunk ID: {record['id']}")
                print(f"Text: {record['text'][:250]!r}")
                print()

    print("=" * 80)
    print("Query-variant retrieval comparison complete.")
    return True


if __name__ == "__main__":
    success = run_query_variant_test()
    sys.exit(0 if success else 1)

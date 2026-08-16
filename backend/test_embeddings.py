import os
import sys

# Ensure backend directory is in the python path
backend_dir = os.path.dirname(os.path.abspath(__file__))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from app.embeddings import embed_text


def test_single_text_embedding():
    """
    Test embedding a single sentence and verifying the output properties.
    """
    text = "Employees can work remotely up to three days per week."

    # 1. Generate the embedding
    embedding = embed_text(text)

    # 2. Print embedding details
    print(f"Embedding type: {type(embedding)}")
    print(f"Embedding dimension: {len(embedding)}")
    print(f"First few values: {embedding[:5]}")
    print("=" * 60)

    # 3. Assertions
    assert embedding is not None, "No embedding was generated."
    assert len(embedding) == 384, (
        f"Expected 384 dimensions, got {len(embedding)}."
    )
    assert all(isinstance(v, float) and v == v for v in embedding), (
        "Embedding contains non-finite or non-float values."
    )

    print("=" * 60)
    print("All checks passed! Embedding generated successfully.")
    return True


if __name__ == "__main__":
    success = test_single_text_embedding()
    sys.exit(0 if success else 1)

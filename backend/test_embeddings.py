import os
import sys

# Ensure backend directory is in the python path
backend_dir = os.path.dirname(os.path.abspath(__file__))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from app.chunker import chunk_documents
from app.document_loader import load_pdf
from app.embeddings import embed_text, embed_texts


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


def test_batch_chunk_embeddings():
    """
    Test batch-embedding every chunk from the sample PDFs and
    verifying count, dimensions, and input order.
    """
    project_root = os.path.dirname(backend_dir)
    documents_dir = os.path.join(project_root, "documents")
    pdf_files = [
        "EMPLOYEE HANDBOOK.pdf",
        "EMPLOYEE LEAVE OF ABSENCE POLICY.pdf",
        "EMPLOYEE REMOTE WORK POLICY.pdf"
    ]

    # 1. Load all PDFs and chunk them
    all_pages = []
    for pdf_filename in pdf_files:
        pdf_path = os.path.join(documents_dir, pdf_filename)
        all_pages.extend(load_pdf(pdf_path))

    chunks = chunk_documents(all_pages)
    chunk_texts = [c.page_content for c in chunks]
    print(f"Loaded and chunked PDFs: {len(chunk_texts)} chunks.")

    # 2. Batch-embed all chunk texts
    embeddings = embed_texts(chunk_texts)

    # 3. Print embedding summary
    print(f"Number of chunks: {len(chunk_texts)}")
    print(f"Number of embeddings: {len(embeddings)}")
    print(f"Embedding dimensions: {len(embeddings[0]) if embeddings else 0}")
    print(f"First few values of first embedding: {embeddings[0][:5]}")
    print("=" * 60)

    # 4. Assertions
    assert embeddings, "No embeddings were generated."
    assert len(embeddings) == len(chunk_texts), (
        f"Expected {len(chunk_texts)} embeddings, got {len(embeddings)}."
    )
    for i, embedding in enumerate(embeddings):
        assert len(embedding) == 384, (
            f"Embedding {i} has {len(embedding)} dimensions, expected 384."
        )
        assert all(isinstance(v, float) and v == v for v in embedding), (
            f"Embedding {i} contains non-finite or non-float values."
        )

    # 5. Verify input order: first embedding matches a fresh single
    #    embedding of the first chunk text.
    first_recomputed = embed_text(chunk_texts[0])
    assert len(embeddings[0]) == len(first_recomputed), (
        "First embedding dimension mismatch with recomputed embedding."
    )
    assert all(abs(a - b) < 1e-6 for a, b in zip(embeddings[0], first_recomputed)), (
        "First embedding does not match chunk_texts[0]; order not preserved."
    )

    print("=" * 60)
    print("All checks passed! Batch embedding successful.")
    return True


if __name__ == "__main__":
    success = (
        test_single_text_embedding()
        and test_batch_chunk_embeddings()
    )
    sys.exit(0 if success else 1)

import os
import sys

# Ensure backend directory is in the python path
backend_dir = os.path.dirname(os.path.abspath(__file__))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from app.chunker import chunk_documents
from app.document_loader import load_pdf
from app.embeddings import embed_texts
from app.vector_store import COLLECTION_NAME, index_chunks

# Import the collection used by vector_store to inspect stored records.
import chromadb
from app.vector_store import CHROMA_DIR


def test_store_all_chunks():
    """
    Test loading, chunking, embedding, and storing all PDF chunks in
    ChromaDB, verifying count, sample records, and idempotency.
    """
    project_root = os.path.dirname(backend_dir)
    documents_dir = os.path.join(project_root, "documents")
    pdf_files = [
        "EMPLOYEE HANDBOOK.pdf",
        "EMPLOYEE LEAVE OF ABSENCE POLICY.pdf",
        "EMPLOYEE REMOTE WORK POLICY.pdf"
    ]

    # 1. Load all PDFs, chunk them, and batch-embed the chunks
    all_pages = []
    for pdf_filename in pdf_files:
        pdf_path = os.path.join(documents_dir, pdf_filename)
        all_pages.extend(load_pdf(pdf_path))

    chunks = chunk_documents(all_pages)
    chunk_texts = [c.page_content for c in chunks]
    embeddings = embed_texts(chunk_texts)
    print(f"Loaded {len(chunks)} chunks from {len(pdf_files)} PDFs.")
    print("=" * 60)

    # 2. Index all records into ChromaDB
    stored = index_chunks(chunks, embeddings)
    print(f"Stored {stored} records in collection '{COLLECTION_NAME}'.")

    # 3. Inspect the persisted collection
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    count = collection.count()

    # 4. Retrieve a small sample and verify all record components
    sample = collection.get(limit=5, include=["documents", "embeddings", "metadatas"])
    sample_ids = sample.get("ids", [])
    sample_docs = sample.get("documents", [])
    sample_embs = sample.get("embeddings", [])
    sample_metas = sample.get("metadatas", [])

    print(f"Collection count: {count}")
    print(f"Sample IDs: {sample_ids}")
    print(f"Sample metadata: {sample_metas}")
    print("=" * 60)

    # 5. Assertions
    assert stored == len(chunks), "index_chunks did not store every chunk."
    assert count == 98, f"Expected 98 records in collection, got {count}."
    assert len(sample_ids) > 0, "No IDs found in the sample."
    assert len(sample_docs) == len(sample_ids), "Documents missing for sample IDs."
    assert len(sample_embs) == len(sample_ids), "Embeddings missing for sample IDs."
    assert len(sample_metas) == len(sample_ids), "Metadata missing for sample IDs."
    assert all(len(e) == 384 for e in sample_embs), "Sample embedding dimension is not 384."
    assert all("source" in m and "page" in m for m in sample_metas), (
        "Sample records are missing source/page metadata."
    )

    # 6. Idempotency check: re-indexing must not duplicate records
    stored_again = index_chunks(chunks, embeddings)
    count_after = collection.count()
    assert count_after == count, (
        f"Re-indexing changed the count from {count} to {count_after}; not idempotent."
    )

    print("=" * 60)
    print(f"SUMMARY: {stored} chunks indexed ({count_after} total records, "
          f"re-index idempotent).")
    print("All checks passed! Vector store populated successfully.")
    return True


if __name__ == "__main__":
    success = test_store_all_chunks()
    sys.exit(0 if success else 1)

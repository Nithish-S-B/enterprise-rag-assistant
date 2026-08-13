import os
import sys

# Ensure backend directory is in the python path
backend_dir = os.path.dirname(os.path.abspath(__file__))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from app.chunker import chunk_documents
from app.document_loader import load_pdf


def test_all_pdfs_chunking():
    """
    Test loading all PDF files, chunking their combined page documents,
    and verifying that metadata is preserved across the split.
    """
    # List of our sample PDF files in the documents directory
    project_root = os.path.dirname(backend_dir)
    documents_dir = os.path.join(project_root, "documents")
    pdf_files = [
        "EMPLOYEE HANDBOOK.pdf",
        "EMPLOYEE LEAVE OF ABSENCE POLICY.pdf",
        "EMPLOYEE REMOTE WORK POLICY.pdf"
    ]

    # 1. Load all PDFs and combine their page documents
    all_pages = []
    for pdf_filename in pdf_files:
        pdf_path = os.path.join(documents_dir, pdf_filename)
        pages = load_pdf(pdf_path)
        all_pages.extend(pages)

    input_pages = len(all_pages)
    print(f"Loaded {input_pages} pages across {len(pdf_files)} PDFs.")
    print("=" * 60)

    # 2. Chunk the combined page documents
    chunks = chunk_documents(all_pages)
    output_chunks = len(chunks)
    print(f"Chunking produced {output_chunks} chunks.")
    print("=" * 60)

    # 3. Verify metadata is preserved on every chunk
    all_have_source = all("source" in c.metadata for c in chunks)
    all_have_page = all("page" in c.metadata for c in chunks)
    print(f"All chunks preserve 'source' metadata: {all_have_source}")
    print(f"All chunks preserve 'page' metadata: {all_have_page}")
    print("=" * 60)

    # 4. Print a short preview of several chunks (first and last few)
    preview_count = 4
    print(f"\nPreviewing the first {preview_count} chunks:")
    for i, chunk in enumerate(chunks[:preview_count]):
        print(f"\n  Chunk {i}:")
        print(f"    Source: {chunk.metadata.get('source', 'Unknown')}")
        print(f"    Page:   {chunk.metadata.get('page', -1)}")
        print(f"    Length: {len(chunk.page_content)} chars")
        print(f"    Content: {chunk.page_content[:150]}...")

    print(f"\nPreviewing the last {preview_count} chunks:")
    for i, chunk in enumerate(chunks[-preview_count:], start=output_chunks - preview_count):
        print(f"\n  Chunk {i}:")
        print(f"    Source: {chunk.metadata.get('source', 'Unknown')}")
        print(f"    Page:   {chunk.metadata.get('page', -1)}")
        print(f"    Length: {len(chunk.page_content)} chars")
        print(f"    Content: {chunk.page_content[:150]}...")

    # 5. Report chunk length statistics
    lengths = [len(c.page_content) for c in chunks]
    print("\n" + "=" * 60)
    print(f"Chunk length stats: min={min(lengths)}, max={max(lengths)}, "
          f"avg={sum(lengths) // len(lengths)}")

    # 6. Assertions
    assert output_chunks > 0, "No chunks were created."
    assert output_chunks >= input_pages, "Expected at least as many chunks as input pages."
    assert all_have_source, "Some chunks are missing 'source' metadata."
    assert all_have_page, "Some chunks are missing 'page' metadata."

    print("=" * 60)
    print(f"SUMMARY: {input_pages} pages -> {output_chunks} chunks")
    print("All checks passed! Metadata preserved and chunks created.")
    return True


if __name__ == "__main__":
    success = test_all_pdfs_chunking()
    sys.exit(0 if success else 1)

import os
import sys

# Ensure backend directory is in the python path
backend_dir = os.path.dirname(os.path.abspath(__file__))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from app.document_loader import load_pdf


def test_single_pdf_loading():
    """
    Test loading a single PDF file from the project root
    and printing its metadata and content snippet.
    """
    # Locate one of our sample PDF files at the root of the repository
    project_root = os.path.dirname(backend_dir)
    pdf_path = os.path.join(project_root, "Employee-remote-work-policy.pdf")

    print(f"Testing loader with PDF: {pdf_path}")

    try:
        # 1. Load the PDF
        documents = load_pdf(pdf_path)

        # 2. Verify and print page count
        page_count = len(documents)
        print(f"Successfully loaded: {page_count} pages.")

        if page_count > 0:
            first_page = documents[0]

            # 3. Print metadata of the first page
            print("\n--- METADATA ---")
            print(first_page.metadata)

            # 4. Print first 500 characters of the first page
            print("\n--- CONTENT SNIPPET (First 500 chars) ---")
            snippet = first_page.page_content[:500]
            print(snippet)
            if len(first_page.page_content) > 500:
                print("... [truncated] ...")
        else:
            print("Warning: The loaded document has 0 pages.")

    except Exception as e:
        print(f"Error occurred during test: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    test_single_pdf_loading()

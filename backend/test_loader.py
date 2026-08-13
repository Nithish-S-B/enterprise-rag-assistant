import os
import sys

# Ensure backend directory is in the python path
backend_dir = os.path.dirname(os.path.abspath(__file__))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from app.document_loader import load_pdf


def test_all_pdfs_loading():
    """
    Test loading all PDF files from the project root
    and printing summary information for each.
    """
    # List of our sample PDF files in the documents directory
    project_root = os.path.dirname(backend_dir)
    documents_dir = os.path.join(project_root, "documents")
    pdf_files = [
        "EMPLOYEE HANDBOOK.pdf",
        "EMPLOYEE LEAVE OF ABSENCE POLICY.pdf",
        "EMPLOYEE REMOTE WORK POLICY.pdf"
    ]

    print(f"Testing loader with {len(pdf_files)} PDF files from: {documents_dir}")
    print("=" * 60)

    total_pages = 0
    success_count = 0

    for pdf_filename in pdf_files:
        pdf_path = os.path.join(documents_dir, pdf_filename)
        print(f"\nTesting: {pdf_filename}")

        try:
            # 1. Load the PDF
            documents = load_pdf(pdf_path)

            # 2. Verify and print page count
            page_count = len(documents)
            total_pages += page_count
            success_count += 1
            print(f"  ����� ��� ��� � ��� � � ✓ Successfully loaded: {page_count} pages.")

            if page_count > 0:
                first_page = documents[0]

                # 3. Print metadata of the first page (just source and page for brevity)
                metadata_summary = {
                    'source': first_page.metadata.get('source', 'Unknown'),
                    'page': first_page.metadata.get('page', -1),
                    'total_pages': first_page.metadata.get('total_pages', 'Unknown')
                }
                print(f"  ������ ���� ���� �� ���� �� �� 📄 Metadata: {metadata_summary}")

                # 4. Print first 200 characters of the first page
                snippet = first_page.page_content[:200]
                print(f"  ������ ���� ���� �� ���� �� �� 📝 Content preview: {snippet}...")

        except Exception as e:
            print(f"  ����� ��� ��� � ��� � � ✗ Error occurred: {e}")

    print("\n" + "=" * 60)
    print(f"SUMMARY: {success_count}/{len(pdf_files)} PDFs loaded successfully")
    print(f"TOTAL PAGES: {total_pages}")

    if success_count == len(pdf_files):
        print("���������������������🎉 All PDFs loaded successfully!")
        return True
    else:
        print("��������������❌ Some PDFs failed to load.")
        return False

if __name__ == "__main__":
    success = test_all_pdfs_loading()
    sys.exit(0 if success else 1)

"""Tests for POST /api/documents/upload, GET /api/documents, and
DELETE /api/documents/{document_id}."""
import os
import shutil
import sys
import tempfile
import unittest.mock
from pathlib import Path

import chromadb
from fastapi.testclient import TestClient
from langchain_core.documents import Document

import app.vector_store as vector_store_module
from app.main import app
from app.vector_store import CHROMA_DIR, COLLECTION_NAME


MINIMAL_PDF_CONTENT = (
    b"%PDF-1.4\n"
    b"1 0 obj\n<< /Type /Catalog >>\nendobj\n"
    b"trailer\n<< /Root 1 0 R >>\n%%EOF\n"
)
CORRUPT_PDF_CONTENT = b"%PDF-1.4\ncorrupt payload without xref or eof"

UPLOAD_ENDPOINT = "/api/documents/upload"
LIST_ENDPOINT = "/api/documents"
DELETE_ENDPOINT = "/api/documents/{document_id}"
TEMP_FILE_PREFIX = "rag_upload_"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REAL_PDF_NAME = "EMPLOYEE REMOTE WORK POLICY.pdf"
REAL_PDF_PATH = PROJECT_ROOT / "documents" / REAL_PDF_NAME

EXPECTED_DOCUMENT_IDS = {
    "employee_handbook",
    "employee_leave_of_absence_policy",
    "employee_remote_work_policy",
}


def _upload_file(client: TestClient, filename: str, content: bytes,
                 content_type: str = "application/pdf"):
    return client.post(
        UPLOAD_ENDPOINT,
        files={"file": (filename, content, content_type)},
    )


def _temp_leftovers() -> set[str]:
    temp_dir = tempfile.gettempdir()
    return {
        name for name in os.listdir(temp_dir)
        if name.startswith(TEMP_FILE_PREFIX)
    }


def test_valid_pdf_upload_ingests_and_indexes() -> bool:
    """A: A real repository PDF passes validation and is fully indexed."""
    assert REAL_PDF_PATH.exists(), f"Expected real PDF at {REAL_PDF_PATH}."
    client = TestClient(app)
    with open(REAL_PDF_PATH, "rb") as handle:
        pdf_bytes = handle.read()

    response = _upload_file(client, REAL_PDF_NAME, pdf_bytes)

    assert response.status_code == 200, (
        f"Expected HTTP 200, got {response.status_code}: {response.text}"
    )

    body = response.json()
    assert body["filename"] == REAL_PDF_NAME, "Expected original filename echoed."
    assert body["status"] == "indexed", "Expected status 'indexed'."
    assert body["pages"] > 0, "Expected at least one parsed page."
    assert body["chunks"] > 0, "Expected at least one generated chunk."
    assert body["document_id"], "Expected a document_id to be present."

    document_id = body["document_id"]
    assert "/" not in document_id and "\\" not in document_id, (
        "document_id must not leak filesystem paths."
    )
    assert document_id == "employee_remote_work_policy", (
        f"Unexpected deterministic document_id: {document_id}"
    )
    print("POST /api/documents/upload ->", body)
    return True


def test_txt_upload_rejected_without_ingestion() -> bool:
    """B: Non-PDF extensions are rejected with 415 before any processing."""
    client = TestClient(app)
    with unittest.mock.patch("app.api.documents.load_pdf") as spy_loader:
        response = _upload_file(client, "notes.txt", b"plain text content")

    assert response.status_code == 415, (
        f"Expected HTTP 415 for .txt upload, got {response.status_code}: {response.text}"
    )
    spy_loader.assert_not_called(), "Ingestion must not run for invalid extensions."
    return True


def test_missing_file_rejected() -> bool:
    """C1: A request with no file part is rejected by request validation."""
    client = TestClient(app)
    response = client.post(UPLOAD_ENDPOINT)

    assert response.status_code == 422, (
        f"Expected HTTP 422 for missing file, got {response.status_code}"
    )
    return True


def test_blank_filename_rejected() -> bool:
    """C2: A whitespace-only filename is rejected before any processing."""
    client = TestClient(app)
    response = _upload_file(client, "   ", b"%PDF-1.4 fake")

    assert response.status_code == 422, (
        f"Expected HTTP 422 for blank filename, got {response.status_code}: {response.text}"
    )
    return True


def test_empty_file_rejected() -> bool:
    """D: A zero-byte PDF is syntactically addressed but has no content."""
    client = TestClient(app)
    response = _upload_file(client, "empty.pdf", b"")

    assert response.status_code == 400, (
        f"Expected HTTP 400 for empty upload, got {response.status_code}: {response.text}"
    )
    return True


def test_oversized_file_rejected() -> bool:
    """E: Streaming size enforcement aborts uploads beyond the configured limit."""
    previous_limit = os.environ.get("MAX_UPLOAD_SIZE_MB")
    os.environ["MAX_UPLOAD_SIZE_MB"] = "1"
    try:
        client = TestClient(app)
        oversized_content = MINIMAL_PDF_CONTENT + b"A" * (2 * 1024 * 1024)
        response = _upload_file(client, "large.pdf", oversized_content)

        assert response.status_code == 413, (
            f"Expected HTTP 413 for oversized upload, got "
            f"{response.status_code}: {response.text[:200]}"
        )
    finally:
        if previous_limit is None:
            os.environ.pop("MAX_UPLOAD_SIZE_MB", None)
        else:
            os.environ["MAX_UPLOAD_SIZE_MB"] = previous_limit
    return True


def test_pdf_extension_with_fake_content_rejected() -> bool:
    """Signature check: .pdf name with non-PDF bytes must not pass validation."""
    client = TestClient(app)
    response = _upload_file(client, "disguised.pdf", b"definitely not a pdf")

    assert response.status_code == 415, (
        f"Expected HTTP 415 for fake PDF content, got {response.status_code}"
    )
    return True


def test_corrupt_pdf_returns_safe_error_and_cleans_temp() -> bool:
    """F1: Signed-but-corrupt PDF yields a safe error and leaves no temp file."""
    client = TestClient(app)
    before = _temp_leftovers()

    response = _upload_file(client, "corrupt.pdf", CORRUPT_PDF_CONTENT)

    assert response.status_code >= 400, (
        f"Expected an error status for corrupt PDF, got {response.status_code}"
    )
    detail = response.json()["detail"]
    assert detail == "Failed to process the uploaded PDF.", (
        f"Expected a safe generic error detail, got: {detail}"
    )
    assert _temp_leftovers() == before, (
        "Temporary PDF files were left behind after failed ingestion."
    )
    return True


def test_duplicate_upload_is_idempotent() -> bool:
    """F2: Re-uploading the same PDF must not double the ChromaDB records."""
    assert REAL_PDF_PATH.exists(), f"Expected real PDF at {REAL_PDF_PATH}."
    client = TestClient(app)
    with open(REAL_PDF_PATH, "rb") as handle:
        pdf_bytes = handle.read()

    first = _upload_file(client, REAL_PDF_NAME, pdf_bytes)
    assert first.status_code == 200, f"First upload failed: {first.text}"

    collection = chromadb.PersistentClient(path=CHROMA_DIR).get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    count_after_first = collection.count()
    assert count_after_first > 0, "Expected records in the collection after indexing."

    second = _upload_file(client, REAL_PDF_NAME, pdf_bytes)
    assert second.status_code == 200, f"Second upload failed: {second.text}"

    count_after_second = collection.count()
    assert count_after_second == count_after_first, (
        f"Re-upload changed record count from {count_after_first} to "
        f"{count_after_second}; duplicates were created."
    )

    first_body = first.json()
    second_body = second.json()
    assert first_body["document_id"] == second_body["document_id"], (
        "Same file must map to the same deterministic document_id."
    )
    assert first_body["chunks"] == second_body["chunks"], (
        "Same file must produce the same chunk count."
    )
    print(
        f"Duplicate upload idempotent: {count_after_second} records "
        f"(unchanged after re-upload)."
    )
    return True


def test_docs_advertise_upload_endpoint() -> bool:
    """G: /docs and /openapi.json advertise POST /api/documents/upload."""
    client = TestClient(app)
    docs_response = client.get("/docs")
    openapi_response = client.get("/openapi.json")

    assert docs_response.status_code == 200, "Expected /docs to remain available."
    assert openapi_response.status_code == 200, "Expected /openapi.json to work."

    paths = openapi_response.json()["paths"]
    assert "/api/documents/upload" in paths, "Expected /api/documents/upload in OpenAPI paths."
    assert "post" in paths["/api/documents/upload"], "Expected upload endpoint to accept POST."

    schema_props = (
        openapi_response.json()["components"]["schemas"]
        ["UploadIngestionResponse"]["properties"]
    )
    expected_fields = {"document_id", "filename", "pages", "chunks", "status"}
    assert expected_fields.issubset(schema_props), (
        f"Expected typed ingestion fields in schema, got: {sorted(schema_props)}"
    )

    print("OpenAPI paths:", sorted(paths.keys()))
    return True


def test_list_documents_returns_summaries() -> bool:
    """H: GET /api/documents returns typed, path-free, sorted summaries."""
    client = TestClient(app)

    # N: the listing endpoint must never reach OpenRouter/LLM code.
    with unittest.mock.patch("app.api.chat.answer_question") as llm_guard:
        response = client.get(LIST_ENDPOINT)
        llm_guard.assert_not_called()

    assert response.status_code == 200, (
        f"Expected HTTP 200 from GET {LIST_ENDPOINT}, got "
        f"{response.status_code}: {response.text}"
    )

    body = response.json()
    for key in ("documents", "total_documents", "total_chunks"):
        assert key in body, f"Expected '{key}' in response body."

    documents = body["documents"]
    assert isinstance(documents, list) and documents, (
        "Expected at least one indexed document."
    )

    document_ids: set[str] = set()
    filenames: list[str] = []
    total_chunks = 0
    for document in documents:
        for field in ("document_id", "filename", "pages", "chunks"):
            assert field in document, f"Expected '{field}' on each document."
        filename = document["filename"]
        document_id = document["document_id"]

        # D: no absolute paths may leak into filenames.
        assert "D:\\" not in filename and "D:/" not in filename, (
            f"Filename leaks an absolute path: {filename}"
        )
        assert str(PROJECT_ROOT) not in filename, (
            f"Filename leaks the workspace path: {filename}"
        )
        assert "/" not in filename and "\\" not in filename, (
            f"Filename must be a bare name, got: {filename}"
        )

        # E: IDs are normalized and path-free.
        assert "/" not in document_id and "\\" not in document_id, (
            f"document_id must be path-free: {document_id}"
        )
        assert document_id == document_id.lower(), (
            f"document_id must be lowercase: {document_id}"
        )
        allowed = set("abcdefghijklmnopqrstuvwxyz0123456789_")
        assert set(document_id) <= allowed and not document_id.startswith("_"), (
            f"document_id is not normalized: {document_id}"
        )
        assert document["pages"] > 0, f"Expected pages >= 1 for {document_id}."
        assert document["chunks"] > 0, f"Expected chunks >= 1 for {document_id}."

        # H: no duplicate logical documents.
        assert document_id not in document_ids, (
            f"Duplicate document_id returned: {document_id}"
        )
        document_ids.add(document_id)
        filenames.append(filename)
        total_chunks += document["chunks"]

    # F: deterministic ordering by filename.
    assert filenames == sorted(filenames), (
        f"Documents are not sorted by filename: {filenames}"
    )

    # I/J: totals are derived from the per-document summaries.
    assert body["total_documents"] == len(documents), (
        "total_documents must equal len(documents)."
    )
    assert body["total_chunks"] == total_chunks, (
        "total_chunks must equal the sum of per-document chunk counts."
    )

    # Current fixture: all three policy PDFs are indexed.
    missing = EXPECTED_DOCUMENT_IDS - document_ids
    assert not missing, f"Expected indexed documents missing: {sorted(missing)}"

    print(f"GET {LIST_ENDPOINT} ->", {
        "total_documents": body["total_documents"],
        "total_chunks": body["total_chunks"],
    })
    return True


def test_legacy_and_new_records_merge_into_one_document() -> bool:
    """I: Legacy chunks (absolute source, no document_id) merge with new
    uploads (bare filename + document_id) into ONE logical document."""
    legacy_source = str(PROJECT_ROOT / "documents" / REAL_PDF_NAME)
    fake_metadatas = [
        {"source": legacy_source, "page": page}
        for page in (0, 2, 4)
    ] + [
        {"source": REAL_PDF_NAME, "page": page,
         "document_id": "employee_remote_work_policy"}
        for page in (5, 9)
    ]
    mocked_collection = unittest.mock.MagicMock()
    mocked_collection.get.return_value = {"metadatas": fake_metadatas}

    with unittest.mock.patch.object(
        vector_store_module, "_collection", mocked_collection
    ):
        documents = vector_store_module.list_documents()

    remote_work = [
        doc for doc in documents
        if doc["document_id"] == "employee_remote_work_policy"
    ]
    assert len(remote_work) == 1, (
        f"Legacy and new records did not merge into one document: {remote_work}"
    )
    merged = remote_work[0]
    assert merged["filename"] == REAL_PDF_NAME, (
        f"Expected the bare filename, got: {merged['filename']}"
    )
    # Unique-page counting: pages {0,2,4} ∪ {5,9} -> 5 distinct pages,
    # while max(page)+1 would wrongly report 10 due to gaps.
    assert merged["pages"] == 5, (
        f"Expected 5 unique pages, got: {merged['pages']}"
    )
    assert merged["chunks"] == 5, (
        f"Expected 5 merged chunk records, got: {merged['chunks']}"
    )
    mocked_collection.get.assert_called_once_with(include=["metadatas"])
    print("Legacy + new records merged:", merged)
    return True


def test_docs_advertise_list_endpoint() -> bool:
    """J: /openapi.json advertises GET /api/documents."""
    client = TestClient(app)
    openapi_response = client.get("/openapi.json")

    assert openapi_response.status_code == 200, "Expected /openapi.json to work."
    paths = openapi_response.json()["paths"]
    assert "/api/documents" in paths, "Expected /api/documents in OpenAPI paths."
    assert "get" in paths["/api/documents"], (
        "Expected GET /api/documents to be advertised."
    )
    return True


# ---------------------------------------------------------------------------
# DELETE /api/documents/{document_id}
#
# Deletion tests run against isolated, temporary ChromaDB collections
# (patching vector_store_module._collection) so the real chroma_db/ — and
# its historical employee_remote_work_policy duplicates — is never mutated.
# ---------------------------------------------------------------------------

DELETE_TARGET_ID = "employee_remote_work_policy"
DELETE_OTHER_ID = "employee_handbook"
OTHER_PDF_NAME = "EMPLOYEE HANDBOOK.pdf"
TARGET_CHUNK_COUNT = 9  # 4 legacy-style + 5 new-style records for the target
SCENARIO_CHUNK_COUNT = TARGET_CHUNK_COUNT + 2  # plus 2 chunks of the other doc


def _scenario_chunk(source: str, page: int,
                    document_id: str | None = None) -> Document:
    """Builds one chunk record, optionally stamped as a new-style upload."""
    metadata = {"source": source, "page": page}
    if document_id:
        metadata["document_id"] = document_id
    return Document(page_content=f"Scenario text {source} p{page}", metadata=metadata)


def _dummy_embedding() -> list[float]:
    return [0.5] * 384


def _new_isolated_collection(prefix: str):
    """Creates a throwaway disk-backed collection for deletion tests."""
    temp_dir = tempfile.mkdtemp(prefix=prefix)
    client = chromadb.PersistentClient(path=temp_dir)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    return temp_dir, collection


def _seed_deletion_scenario(collection) -> None:
    """
    Indexes two logical documents into an isolated collection:

    - Target: 4 legacy chunks (absolute-path source, NO document_id)
      + 5 new chunks (bare filename + document_id) -> 9 records that all
      resolve to employee_remote_work_policy.
    - Other: 1 legacy + 1 new chunk -> 2 records for employee_handbook.
    """
    legacy_target_source = str(PROJECT_ROOT / "documents" / REAL_PDF_NAME)
    target_chunks = (
        [_scenario_chunk(legacy_target_source, page) for page in range(4)]
        + [
            _scenario_chunk(REAL_PDF_NAME, page, document_id=DELETE_TARGET_ID)
            for page in range(4, 9)
        ]
    )
    other_chunks = [
        _scenario_chunk(str(PROJECT_ROOT / "documents" / OTHER_PDF_NAME), 0),
        _scenario_chunk(OTHER_PDF_NAME, 1, document_id=DELETE_OTHER_ID),
    ]
    for chunk in target_chunks + other_chunks:
        vector_store_module.index_chunks([chunk], [_dummy_embedding()])


def test_delete_existing_document_removes_every_chunk() -> bool:
    """K1-K5: DELETE returns a typed summary, removes ALL chunks (legacy +
    new) for the target only, vanishes from listing and raw ChromaDB,
    leaves the unrelated document intact, and repeats yield 404."""
    temp_dir, collection = _new_isolated_collection("rag_delete_scenario_")
    try:
        api_client = TestClient(app)
        with unittest.mock.patch.object(vector_store_module, "_collection", collection):
            # Seeding MUST run while patched so index_chunks writes to the
            # isolated collection, never to the real chroma_db/.
            _seed_deletion_scenario(collection)
            assert collection.count() == SCENARIO_CHUNK_COUNT, (
                f"Expected {SCENARIO_CHUNK_COUNT} seeded records, "
                f"got {collection.count()}."
            )

            # K1: deleting an existing document succeeds with a summary.
            response = api_client.delete(f"/api/documents/{DELETE_TARGET_ID}")
            assert response.status_code == 200, (
                f"Expected HTTP 200, got {response.status_code}: {response.text}"
            )
            body = response.json()
            assert body["document_id"] == DELETE_TARGET_ID
            assert body["status"] == "deleted", "Expected status 'deleted'."
            assert body["deleted_chunks"] > 0, "Expected deleted_chunks > 0."
            assert body["deleted_chunks"] == TARGET_CHUNK_COUNT, (
                f"Expected all {TARGET_CHUNK_COUNT} target chunks deleted, "
                f"got {body['deleted_chunks']}."
            )

            # K2: the deleted document no longer appears in the listing.
            listed = api_client.get(LIST_ENDPOINT)
            assert listed.status_code == 200, f"Listing failed: {listed.text}"
            listed_ids = {
                doc["document_id"] for doc in listed.json()["documents"]
            }
            assert DELETE_TARGET_ID not in listed_ids, (
                "Deleted document still appears in GET /api/documents."
            )

            # K4: deleting the same document again must 404.
            repeat = api_client.delete(f"/api/documents/{DELETE_TARGET_ID}")
            assert repeat.status_code == 404, (
                f"Expected HTTP 404 on re-delete, got {repeat.status_code}."
            )

        # K3: raw ChromaDB scan — nothing resolves to the deleted ID anymore.
        remaining = collection.get(include=["metadatas"])
        remaining_metas = remaining.get("metadatas") or []
        surviving_ids = {
            vector_store_module._resolve_record_document_id(meta or {})
            for meta in remaining_metas
        }
        assert DELETE_TARGET_ID not in surviving_ids, (
            "Chunk metadata still resolves to the deleted document_id."
        )
        assert collection.count() == SCENARIO_CHUNK_COUNT - TARGET_CHUNK_COUNT

        # K5/K7: ONLY the unrelated document remains, fully intact.
        assert surviving_ids == {DELETE_OTHER_ID}, (
            f"Unexpected surviving documents: {surviving_ids}"
        )
        other_pages = sorted(
            meta["page"] for meta in remaining_metas
            if vector_store_module._resolve_record_document_id(meta) == DELETE_OTHER_ID
        )
        assert other_pages == [0, 1], f"Other doc chunks altered: {other_pages}"

        print("DELETE existing ->", body)
        print(f"ChromaDB after delete: {collection.count()} records remain "
              f"(other document untouched).")
        return True
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_delete_missing_document_returns_404() -> bool:
    """K6: Deleting a document_id with no indexed chunks returns 404."""
    temp_dir, collection = _new_isolated_collection("rag_delete_404_")
    try:
        with unittest.mock.patch.object(vector_store_module, "_collection", collection):
            vector_store_module.index_chunks(
                [_scenario_chunk(OTHER_PDF_NAME, 0, document_id=DELETE_OTHER_ID)],
                [_dummy_embedding()],
            )
            api_client = TestClient(app)
            response = api_client.delete("/api/documents/employee_leave_of_absence_policy")

        assert response.status_code == 404, (
            f"Expected HTTP 404 for missing document, got {response.status_code}: "
            f"{response.text}"
        )
        detail = response.json()["detail"]
        assert isinstance(detail, str) and detail, "Expected a safe error detail."
        print("DELETE missing -> 404:", detail)
        return True
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_delete_identifies_legacy_and_new_records_via_mock() -> bool:
    """K7: With mocked Chroma metadata mixing legacy records (absolute
    source, no document_id) and new records (with document_id), deletion
    must select BOTH sets and never touch another document."""
    legacy_source = str(PROJECT_ROOT / "documents" / REAL_PDF_NAME)
    record_ids = ["legacy#p0", "legacy#p1", "new#p2", "new#p3", "other#p0"]
    fake_metadatas = [
        {"source": legacy_source, "page": 0},
        {"source": legacy_source, "page": 1},
        {"source": REAL_PDF_NAME, "page": 2, "document_id": DELETE_TARGET_ID},
        {"source": REAL_PDF_NAME, "page": 3, "document_id": DELETE_TARGET_ID},
        {"source": "OTHER POLICY.pdf", "page": 0, "document_id": "other_policy"},
    ]
    mocked_collection = unittest.mock.MagicMock()
    mocked_collection.get.return_value = {
        "ids": record_ids,
        "metadatas": fake_metadatas,
    }

    with unittest.mock.patch.object(
        vector_store_module, "_collection", mocked_collection
    ):
        deleted = vector_store_module.delete_document(DELETE_TARGET_ID)

    assert deleted == 4, f"Expected 4 chunks deleted (2 legacy + 2 new), got {deleted}."
    mocked_collection.delete.assert_called_once()
    removed_ids = mocked_collection.delete.call_args.kwargs["ids"]
    assert sorted(removed_ids) == sorted(record_ids[:4]), (
        f"Deletion selected the wrong chunk IDs: {removed_ids}"
    )
    assert "other#p0" not in removed_ids, "Unrelated document was selected for deletion."
    print("Legacy+new mock delete removed IDs:", sorted(removed_ids))
    return True


def test_delete_invalid_document_id_rejected_with_422() -> bool:
    """K8: Unsafe/invalid path input fails request validation with 422."""
    client = TestClient(app)
    invalid_ids = ["Employee Remote Work Policy", "UPPER_CASE_ID", "id-with-dash"]
    for bad_id in invalid_ids:
        response = client.delete(f"/api/documents/{bad_id}")
        assert response.status_code == 422, (
            f"Expected HTTP 422 for unsafe id {bad_id!r}, got "
            f"{response.status_code}: {response.text}"
        )
    print(f"Unsafe ids rejected with 422: {invalid_ids}")
    return True


def test_delete_unexpected_failure_returns_safe_500() -> bool:
    """K9: A ChromaDB outage surfaces as 500 with a safe generic message."""
    broken_collection = unittest.mock.MagicMock()
    broken_collection.get.side_effect = RuntimeError("simulated chroma outage")

    with unittest.mock.patch.object(
        vector_store_module, "_collection", broken_collection
    ):
        response = TestClient(app).delete(f"/api/documents/{DELETE_OTHER_ID}")

    assert response.status_code == 500, (
        f"Expected HTTP 500 on unexpected failure, got {response.status_code}"
    )
    detail = response.json()["detail"]
    assert detail == "Failed to delete the document.", (
        f"Expected a safe generic detail, got: {detail}"
    )
    assert "simulated chroma outage" not in response.text, "Internal error leaked."
    assert "RuntimeError" not in response.text, "Exception class name leaked."
    return True


def test_docs_advertise_delete_endpoint() -> bool:
    """K10: /openapi.json advertises DELETE /api/documents/{document_id}."""
    client = TestClient(app)
    openapi_response = client.get("/openapi.json")

    assert openapi_response.status_code == 200, "Expected /openapi.json to work."
    paths = openapi_response.json()["paths"]
    assert "/api/documents/{document_id}" in paths, (
        "Expected DELETE path in OpenAPI paths."
    )
    assert "delete" in paths["/api/documents/{document_id}"], (
        "Expected DELETE method to be advertised."
    )

    schema_props = (
        openapi_response.json()["components"]["schemas"]
        ["DocumentDeleteResponse"]["properties"]
    )
    expected_fields = {"document_id", "deleted_chunks", "status"}
    assert set(schema_props) == expected_fields, (
        f"Expected typed delete fields in schema, got: {sorted(schema_props)}"
    )
    status_schema = schema_props["status"]
    # Pydantic v2 renders Literal["deleted"] as {"const": "deleted"};
    # accept either that or the equivalent single-value enum form.
    status_value = status_schema.get("const", (status_schema.get("enum") or [None])[0])
    assert status_value == "deleted", (
        f"Expected Literal['deleted'] in schema, got: {status_schema}"
    )

    print("OpenAPI delete path advertised:", "/api/documents/{document_id}")
    return True


if __name__ == "__main__":
    tests = [
        test_valid_pdf_upload_ingests_and_indexes,
        test_txt_upload_rejected_without_ingestion,
        test_missing_file_rejected,
        test_blank_filename_rejected,
        test_empty_file_rejected,
        test_oversized_file_rejected,
        test_pdf_extension_with_fake_content_rejected,
        test_corrupt_pdf_returns_safe_error_and_cleans_temp,
        test_duplicate_upload_is_idempotent,
        test_docs_advertise_upload_endpoint,
        test_list_documents_returns_summaries,
        test_legacy_and_new_records_merge_into_one_document,
        test_docs_advertise_list_endpoint,
        test_delete_existing_document_removes_every_chunk,
        test_delete_missing_document_returns_404,
        test_delete_identifies_legacy_and_new_records_via_mock,
        test_delete_invalid_document_id_rejected_with_422,
        test_delete_unexpected_failure_returns_safe_500,
        test_docs_advertise_delete_endpoint,
    ]
    success = all(test() for test in tests)
    print("\nDocuments API tests complete." if success else "\nDocuments API tests FAILED.")
    sys.exit(0 if success else 1)

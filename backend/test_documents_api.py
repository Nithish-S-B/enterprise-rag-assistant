"""Tests for POST /api/documents/upload (with replacement semantics),
GET /api/documents, and DELETE /api/documents/{document_id}."""
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
from app.vector_store import COLLECTION_NAME


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
    temp_dir, collection = _new_isolated_collection("rag_upload_new_")
    try:
        client = TestClient(app)
        with open(REAL_PDF_PATH, "rb") as handle:
            pdf_bytes = handle.read()

        # Patched BEFORE the upload so ingestion never touches chroma_db/.
        with unittest.mock.patch.object(vector_store_module, "_collection", collection):
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
        assert collection.count() == body["chunks"], (
            "Indexed record count must match the reported chunk count."
        )
        print("POST /api/documents/upload ->", body)
        return True
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_txt_upload_rejected_without_ingestion() -> bool:
    """B: Non-PDF extensions are rejected with 415 before any processing."""
    client = TestClient(app)
    with unittest.mock.patch("app.api.documents.load_pdf") as spy_loader:
        response = _upload_file(client, "notes.txt", b"plain text content")

    assert response.status_code == 415, (
        f"Expected HTTP 415 for .txt upload, got {response.status_code}: {response.text}"
    )
    body = response.json()
    assert body["error_type"] == "unsupported_media_type", (
        f"Expected error_type 'unsupported_media_type', got: {body.get('error_type')}"
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
    body = response.json()
    assert body["error_type"] == "validation_error", (
        f"Expected error_type 'validation_error', got: {body.get('error_type')}"
    )
    return True


def test_blank_filename_rejected() -> bool:
    """C2: A whitespace-only filename is rejected before any processing."""
    client = TestClient(app)
    response = _upload_file(client, "   ", b"%PDF-1.4 fake")

    assert response.status_code == 422, (
        f"Expected HTTP 422 for blank filename, got {response.status_code}: {response.text}"
    )
    body = response.json()
    assert body["error_type"] == "validation_error", (
        f"Expected error_type 'validation_error', got: {body.get('error_type')}"
    )
    return True


def test_empty_file_rejected() -> bool:
    """D: A zero-byte PDF is syntactically addressed but has no content."""
    client = TestClient(app)
    response = _upload_file(client, "empty.pdf", b"")

    assert response.status_code == 400, (
        f"Expected HTTP 400 for empty upload, got {response.status_code}: {response.text}"
    )
    body = response.json()
    assert body["error_type"] == "bad_request", (
        f"Expected error_type 'bad_request', got: {body.get('error_type')}"
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
        body = response.json()
        assert body["error_type"] == "bad_request", (
            f"Expected error_type 'bad_request', got: {body.get('error_type')}"
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
    body = response.json()
    assert body["error_type"] == "unsupported_media_type", (
        f"Expected error_type 'unsupported_media_type', got: {body.get('error_type')}"
    )
    return True


def test_corrupt_pdf_returns_safe_error_and_cleans_temp() -> bool:
    """F1: Signed-but-corrupt PDF yields a safe error and leaves no temp file."""
    client = TestClient(app, raise_server_exceptions=False)
    before = _temp_leftovers()

    response = _upload_file(client, "corrupt.pdf", CORRUPT_PDF_CONTENT)

    assert response.status_code >= 400, (
        f"Expected an error status for corrupt PDF, got {response.status_code}"
    )
    body = response.json()
    assert body["error_type"] == "internal_error", (
        f"Expected error_type 'internal_error', got: {body.get('error_type')}"
    )
    assert body["message"] == "An internal server error occurred.", (
        f"Expected a safe generic error message, got: {body.get('message')}"
    )
    assert body.get("request_id") is None, "request_id should be null."
    assert _temp_leftovers() == before, (
        "Temporary PDF files were left behind after failed ingestion."
    )
    return True


def test_duplicate_upload_is_idempotent() -> bool:
    """F2: Re-uploading the same PDF must not double the ChromaDB records."""
    assert REAL_PDF_PATH.exists(), f"Expected real PDF at {REAL_PDF_PATH}."
    temp_dir, collection = _new_isolated_collection("rag_upload_dup_")
    try:
        client = TestClient(app)
        with open(REAL_PDF_PATH, "rb") as handle:
            pdf_bytes = handle.read()

        with unittest.mock.patch.object(vector_store_module, "_collection", collection):
            first = _upload_file(client, REAL_PDF_NAME, pdf_bytes)
            assert first.status_code == 200, f"First upload failed: {first.text}"

            count_after_first = collection.count()
            assert count_after_first > 0, (
                "Expected records in the collection after indexing."
            )

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
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


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


def _scenario_chunk(source: str, page: int, document_id: str | None = None,
                    text: str | None = None) -> Document:
    """Builds one chunk record, optionally stamped as a new-style upload."""
    metadata = {"source": source, "page": page}
    if document_id:
        metadata["document_id"] = document_id
    return Document(
        page_content=text or f"Scenario text {source} p{page}",
        metadata=metadata,
    )


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
        body = response.json()
        assert body["error_type"] == "not_found", (
            f"Expected error_type 'not_found', got: {body.get('error_type')}"
        )
        assert body["message"] == "Document not found.", (
            f"Expected safe 'Document not found.' message, got: {body.get('message')}"
        )
        assert body.get("request_id") is None, "request_id should be null."
        assert "employee_leave_of_absence_policy" not in response.text, (
            "User-supplied document_id must not be reflected in the response."
        )
        print("DELETE missing -> 404:", body["message"])
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
        body = response.json()
        assert body["error_type"] == "validation_error", (
            f"Expected error_type 'validation_error', got: {body.get('error_type')}"
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
            response = TestClient(app, raise_server_exceptions=False).delete(f"/api/documents/{DELETE_OTHER_ID}")

    assert response.status_code == 500, (
        f"Expected HTTP 500 on unexpected failure, got {response.status_code}"
    )
    body = response.json()
    assert body["error_type"] == "internal_error", (
        f"Expected error_type 'internal_error', got: {body.get('error_type')}"
    )
    assert body["message"] == "An internal server error occurred.", (
        f"Expected safe generic message, got: {body.get('message')}"
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


# ---------------------------------------------------------------------------
# POST /api/documents/upload — replacement semantics (Phase 7.9)
#
# Every test here patches app.vector_store._collection BEFORE seeding or
# uploading anything, so ALL writes (seeds, ingests, deletions) hit an
# isolated temporary collection. The real backend/chroma_db is never
# touched by destructive tests.
# ---------------------------------------------------------------------------

LEAVE_PDF_NAME = "EMPLOYEE LEAVE OF ABSENCE POLICY.pdf"
LEAVE_PDF_ID = "employee_leave_of_absence_policy"
STALE_PAGE_BASE = 1000  # real PDFs have < 20 pages, so these IDs never collide
STALE_TEXT_MARKER = "STALE VERSION CHUNK"


def _seed_stale_version(collection, document_id: str, filename: str,
                        count: int) -> None:
    """
    Indexes `count` obsolete v1 chunks for a document into an ISOLATED
    collection, alternating legacy-style (absolute source, no document_id)
    and new-style records. Page numbers start at STALE_PAGE_BASE so their
    chunk IDs can never collide with a genuine upload of the real PDF.
    """
    absolute_source = str(PROJECT_ROOT / "documents" / filename)
    for index in range(count):
        page = STALE_PAGE_BASE + index
        marker_text = f"{STALE_TEXT_MARKER} {document_id} #{index}"
        if index % 2 == 0:
            chunk = _scenario_chunk(absolute_source, page, text=marker_text)
        else:
            chunk = _scenario_chunk(
                filename, page, document_id=document_id, text=marker_text,
            )
        vector_store_module.index_chunks([chunk], [_dummy_embedding()])


def _collection_snapshot(collection) -> tuple[list, list, list]:
    """Reads (ids, documents, metadatas) from the isolated collection."""
    results = collection.get(include=["metadatas", "documents"])
    return (
        list(results.get("ids") or []),
        list(results.get("documents") or []),
        list(results.get("metadatas") or []),
    )


def _resolved_document_ids(metadatas: list) -> set:
    return {
        vector_store_module._resolve_record_document_id(meta or {})
        for meta in metadatas
    }


def _read_pdf_bytes(pdf_name: str) -> bytes:
    pdf_path = PROJECT_ROOT / "documents" / pdf_name
    assert pdf_path.exists(), f"Expected fixture PDF at {pdf_path}."
    with open(pdf_path, "rb") as handle:
        return handle.read()


def test_upload_new_document_indexes_fresh() -> bool:
    """L1/A: Uploading a document_id that does not exist yet simply indexes
    it; previously indexed documents remain untouched."""
    temp_dir, collection = _new_isolated_collection("rag_replace_new_")
    try:
        api_client = TestClient(app)
        leave_bytes = _read_pdf_bytes(LEAVE_PDF_NAME)

        with unittest.mock.patch.object(vector_store_module, "_collection", collection):
            # Seeded while patched: writes stay in the isolated collection.
            vector_store_module.index_chunks(
                [
                    _scenario_chunk(str(PROJECT_ROOT / "documents" / OTHER_PDF_NAME), 0),
                    _scenario_chunk(OTHER_PDF_NAME, 1, document_id=DELETE_OTHER_ID),
                ],
                [_dummy_embedding(), _dummy_embedding()],
            )
            assert collection.count() == 2, "Expected 2 seeded unrelated records."

            with unittest.mock.patch("app.api.chat.answer_question") as llm_guard:
                response = _upload_file(api_client, LEAVE_PDF_NAME, leave_bytes)
                llm_guard.assert_not_called()

            assert response.status_code == 200, (
                f"Expected HTTP 200, got {response.status_code}: {response.text}"
            )
            body = response.json()
            assert body["document_id"] == LEAVE_PDF_ID
            assert body["status"] == "indexed"
            assert body["pages"] > 0 and body["chunks"] > 0

            _, _, metas = _collection_snapshot(collection)
            assert _resolved_document_ids(metas) == {
                DELETE_OTHER_ID, LEAVE_PDF_ID,
            }, f"Unexpected document set after fresh upload: {_resolved_document_ids(metas)}"
            handbook_count = sum(
                1 for meta in metas
                if vector_store_module._resolve_record_document_id(meta) == DELETE_OTHER_ID
            )
            leave_count = sum(
                1 for meta in metas
                if vector_store_module._resolve_record_document_id(meta) == LEAVE_PDF_ID
            )
            assert handbook_count == 2, "Unrelated document was modified."
            assert leave_count == body["chunks"], (
                f"Indexed {leave_count} records but API reported "
                f"{body['chunks']} chunks."
            )

        print("Fresh upload ->", {
            "document_id": body["document_id"],
            "chunks": leave_count,
            "handbook_intact": handbook_count == 2,
        })
        return True
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_replacement_removes_stale_chunks_when_new_version_smaller() -> bool:
    """L2/B (most important): a v1 seeded with MORE chunk records than the
    replacement produces must end up with ONLY the new representation —
    every unmatched stale ID removed."""
    temp_dir, collection = _new_isolated_collection("rag_replace_smaller_")
    try:
        api_client = TestClient(app)
        handbook_bytes = _read_pdf_bytes(OTHER_PDF_NAME)
        stale_count = 45

        with unittest.mock.patch.object(vector_store_module, "_collection", collection):
            _seed_stale_version(collection, DELETE_OTHER_ID, OTHER_PDF_NAME, stale_count)
            assert collection.count() == stale_count

            with unittest.mock.patch("app.api.chat.answer_question") as llm_guard:
                response = _upload_file(api_client, OTHER_PDF_NAME, handbook_bytes)
                llm_guard.assert_not_called()

            assert response.status_code == 200, (
                f"Replacement upload failed: {response.status_code}: {response.text}"
            )
            body = response.json()
            assert body["document_id"] == DELETE_OTHER_ID
            assert body["status"] == "indexed"
            assert body["chunks"] > 0
            assert body["chunks"] < stale_count, (
                f"Fixture broken: new version ({body['chunks']} chunks) is not "
                f"smaller than the seeded stale one ({stale_count})."
            )

            _, docs, metas = _collection_snapshot(collection)
            final_count = collection.count()
            assert final_count == body["chunks"], (
                f"Expected exactly {body['chunks']} records after replacement "
                f"(the new version), got {final_count}; stale chunks survived."
            )
            assert _resolved_document_ids(metas) == {DELETE_OTHER_ID}

            surviving_stale = [d for d in docs if d and STALE_TEXT_MARKER in d]
            assert not surviving_stale, (
                f"{len(surviving_stale)} stale chunk text(s) survived replacement."
            )
            high_pages = [
                meta.get("page") for meta in metas
                if (meta.get("page") or 0) >= STALE_PAGE_BASE
            ]
            assert not high_pages, (
                f"Stale page numbers survived replacement: {high_pages[:5]}"
            )

        print("Replacement (fewer chunks) ->", {
            "stale_records": stale_count,
            "final_records": final_count,
            "reported_chunks": body["chunks"],
        })
        return True
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_replacement_swaps_content_for_same_document_id() -> bool:
    """L3/C: Same document_id, different content -> the final ChromaDB
    records correspond ONLY to the newly uploaded content."""
    temp_dir, collection = _new_isolated_collection("rag_replace_content_")
    try:
        api_client = TestClient(app)
        remote_bytes = _read_pdf_bytes(REAL_PDF_NAME)
        stale_count = 5

        with unittest.mock.patch.object(vector_store_module, "_collection", collection):
            _seed_stale_version(collection, DELETE_TARGET_ID, REAL_PDF_NAME, stale_count)
            assert collection.count() == stale_count

            with unittest.mock.patch("app.api.chat.answer_question") as llm_guard:
                response = _upload_file(api_client, REAL_PDF_NAME, remote_bytes)
                llm_guard.assert_not_called()

            assert response.status_code == 200, (
                f"Replacement upload failed: {response.status_code}: {response.text}"
            )
            body = response.json()
            assert body["document_id"] == DELETE_TARGET_ID
            assert body["status"] == "indexed"
            assert body["chunks"] > 0

            _, docs, metas = _collection_snapshot(collection)
            assert collection.count() == body["chunks"], (
                f"Expected {body['chunks']} records after content swap, "
                f"got {collection.count()}."
            )
            assert not [d for d in docs if d and STALE_TEXT_MARKER in d], (
                "Stale content survived the replacement."
            )
            # Every surviving record was written by THIS upload: bare-filename
            # source and explicit document_id stamped by the pipeline.
            for meta in metas:
                assert vector_store_module._resolve_record_document_id(meta) == DELETE_TARGET_ID
                assert meta.get("source") == REAL_PDF_NAME, (
                    f"Record kept a foreign source: {meta.get('source')!r}"
                )
                assert meta.get("document_id") == DELETE_TARGET_ID
            non_empty_texts = [d for d in docs if d and d.strip()]
            assert len(non_empty_texts) == len(docs), "Empty chunk text found."

        print("Content swap ->", {
            "stale_records": stale_count,
            "final_records": len(docs),
            "all_new_content": True,
        })
        return True
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_processing_failure_preserves_existing_document() -> bool:
    """L4/D: An embedding failure happens BEFORE any deletion, so the old
    document must remain completely untouched."""
    temp_dir, collection = _new_isolated_collection("rag_replace_prefail_")
    try:
        api_client = TestClient(app, raise_server_exceptions=False)
        handbook_bytes = _read_pdf_bytes(OTHER_PDF_NAME)

        with unittest.mock.patch.object(vector_store_module, "_collection", collection):
            old_chunks = [
                _scenario_chunk(
                    OTHER_PDF_NAME, page, document_id=DELETE_OTHER_ID,
                    text=f"OLD-KEPT handbook paragraph {page}",
                )
                for page in range(3)
            ]
            vector_store_module.index_chunks(
                old_chunks, [_dummy_embedding()] * len(old_chunks),
            )
            ids_before, docs_before, _ = _collection_snapshot(collection)
            assert len(ids_before) == 3

            with unittest.mock.patch(
                "app.api.documents.embed_texts",
                side_effect=RuntimeError("embedding model exploded"),
            ):
                with unittest.mock.patch("app.api.chat.answer_question") as llm_guard:
                    response = _upload_file(api_client, OTHER_PDF_NAME, handbook_bytes)
                    llm_guard.assert_not_called()

            assert response.status_code == 500, (
                f"Expected HTTP 500 on processing failure, got {response.status_code}"
            )
            body = response.json()
            assert body["error_type"] == "internal_error", (
                f"Expected error_type 'internal_error', got: {body.get('error_type')}"
            )
            assert body["message"] == "An internal server error occurred.", (
                f"Expected the safe generic message, got: {body.get('message')}"
            )
            assert "embedding model exploded" not in response.text, "Internal error leaked."

            ids_after, docs_after, _ = _collection_snapshot(collection)
            assert sorted(ids_after) == sorted(ids_before), "Chunk IDs changed."
            assert sorted(docs_after) == sorted(docs_before), "Chunk texts changed."

        print("Pre-deletion failure -> old document preserved:", len(ids_after), "records")
        return True
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_index_failure_after_deletion_leaves_safe_error_and_gap() -> bool:
    """L5/E: If index_chunks fails AFTER the old chunks were deleted, the
    API returns a safe 500 and the consistency gap is documented rather
    than hidden: the old representation stays deleted until a retry."""
    temp_dir, collection = _new_isolated_collection("rag_replace_postfail_")
    try:
        api_client = TestClient(app)
        handbook_bytes = _read_pdf_bytes(OTHER_PDF_NAME)

        with unittest.mock.patch.object(vector_store_module, "_collection", collection):
            old_chunks = [
                _scenario_chunk(
                    OTHER_PDF_NAME, page, document_id=DELETE_OTHER_ID,
                    text=f"OLD-GONE handbook paragraph {page}",
                )
                for page in range(4)
            ]
            vector_store_module.index_chunks(
                old_chunks, [_dummy_embedding()] * len(old_chunks),
            )
            assert collection.count() == 4

            # Patch only the pipeline's reference; seeding above used the
            # real vector_store_module.index_chunks and already succeeded.
            with unittest.mock.patch(
                "app.api.documents.index_chunks",
                side_effect=RuntimeError("simulated index explosion"),
            ):
                with unittest.mock.patch("app.api.chat.answer_question") as llm_guard:
                    response = _upload_file(api_client, OTHER_PDF_NAME, handbook_bytes)
                    llm_guard.assert_not_called()

            assert response.status_code == 500, (
                f"Expected HTTP 500 on indexing failure, got {response.status_code}"
            )
            body = response.json()
            assert body["error_type"] == "internal_error", (
                f"Expected error_type 'internal_error', got: {body.get('error_type')}"
            )
            assert body["message"] == (
                "Document replacement failed during final indexing; "
                "please upload the document again."
            ), f"Expected the documented replacement failure message, got: {body.get('message')}"
            assert "simulated index explosion" not in response.text, "Internal error leaked."
            assert "RuntimeError" not in response.text, "Exception class name leaked."

            # DOCUMENTED LIMITATION: delete+index are not atomic across
            # ChromaDB calls. Old chunks were removed before the failure,
            # so the store temporarily holds no representation of this
            # document until a successful retry.
            _, docs_after, metas_after = _collection_snapshot(collection)
            assert collection.count() == 0, (
                f"Expected the documented post-deletion gap (no records); "
                f"found {collection.count()}."
            )
            assert DELETE_OTHER_ID not in _resolved_document_ids(metas_after)

        print("Post-deletion failure -> safe 500 returned; gap documented:",
              "old representation removed, retry required.")
        return True
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


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
        test_upload_new_document_indexes_fresh,
        test_replacement_removes_stale_chunks_when_new_version_smaller,
        test_replacement_swaps_content_for_same_document_id,
        test_processing_failure_preserves_existing_document,
        test_index_failure_after_deletion_leaves_safe_error_and_gap,
    ]
    success = all(test() for test in tests)
    print("\nDocuments API tests complete." if success else "\nDocuments API tests FAILED.")
    sys.exit(0 if success else 1)

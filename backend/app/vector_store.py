"""
Vector storage for the Enterprise RAG Assistant using ChromaDB.

Chunks and their embeddings are indexed into a persistent, disk-backed
collection. Each chunk becomes exactly one record containing a
deterministic ID, the chunk text, its embedding, and its metadata.
"""
import os
import re

import chromadb
from langchain_core.documents import Document

from .embeddings import embed_text

COLLECTION_NAME = "employee_policies"
CHROMA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "chroma_db")


class DocumentNotFoundError(LookupError):
    """
    Raised when no indexed chunk resolves to the requested document_id.

    Signals the API layer to return HTTP 404 without touching ChromaDB.
    """


# Persistent, disk-backed client and collection loaded once at import time.
_client = chromadb.PersistentClient(path=CHROMA_DIR)
_collection = _client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"},
)


def build_chunk_id(chunk: Document, position: int) -> str:
    """
    Builds a deterministic, unique ID for a chunk from its source, page,
    and position in the chunk list.
    """
    source = chunk.metadata.get("source", "unknown")
    safe_source = os.path.splitext(os.path.basename(source))[0].replace(" ", "_")
    page = chunk.metadata.get("page", -1)
    return f"{safe_source}#page{page}#chunk{position}"


def normalize_document_id(filename: str) -> str:
    """
    Derives a stable, filesystem-safe identifier from a document filename.

    Shared by upload (to stamp chunk metadata) and listing (to derive IDs
    for legacy chunks that were stored without one), so both code paths
    always produce identical IDs.

    Example: "Employee Handbook.pdf" -> "employee_handbook"
    """
    stem = os.path.splitext(os.path.basename(filename))[0]
    normalized = re.sub(r"[^a-z0-9]+", "_", stem.lower()).strip("_")
    return normalized or "document"


def _extract_filename(source: str) -> str:
    """Extracts the bare filename from an absolute path or plain filename."""
    return os.path.basename(str(source).replace("\\", "/"))


def _resolve_record_document_id(metadata: dict) -> str:
    """
    Resolves the logical document identity of a single chunk record.

    Records written by newer uploads carry "document_id"; legacy records
    do not, so their ID is re-derived from the source filename using the
    same normalization as upload. Shared by listing and deletion so both
    code paths always agree on which chunks belong to which document.
    """
    document_id = str(metadata.get("document_id") or "").strip()
    if not document_id:
        source = str(metadata.get("source", ""))
        document_id = normalize_document_id(_extract_filename(source))
    return document_id


def list_documents() -> list[dict]:
    """
    Derives per-document summaries from the collection's chunk metadata.

    Read-only: only metadata is read (no documents or embeddings).
    Records written by newer uploads carry "document_id"; legacy records
    do not, so their ID is re-derived from the source filename using the
    same normalization as upload. Both styles therefore merge into the
    same logical document.

    Returns:
        list[dict]: One summary per document, each with "document_id",
            "filename", "pages" (count of distinct parsed pages), and
            "chunks" (number of indexed records), sorted by filename.
    """
    results = _collection.get(include=["metadatas"])
    metadatas = results.get("metadatas") or []

    groups: dict[str, dict] = {}
    for metadata in metadatas:
        filename = _extract_filename(str(metadata.get("source", "")))
        document_id = _resolve_record_document_id(metadata)

        group = groups.setdefault(
            document_id,
            {"filename": filename, "pages": set(), "chunks": 0},
        )
        page = metadata.get("page")
        if page is not None:
            group["pages"].add(page)
        group["chunks"] += 1

    documents = [
        {
            "document_id": document_id,
            "filename": group["filename"],
            "pages": len(group["pages"]),
            "chunks": group["chunks"],
        }
        for document_id, group in groups.items()
    ]
    documents.sort(key=lambda document: document["filename"])
    return documents


def delete_document(document_id: str) -> int:
    """
    Removes every indexed chunk belonging to the requested document.

    Identity resolution matches list_documents(): records with a
    "document_id" metadata value use it directly; legacy records without
    one are resolved from their source filename via
    normalize_document_id(). Only records resolving to exactly the
    requested ID are deleted, so unrelated documents are never touched.

    Args:
        document_id (str): The normalized identifier of the document.

    Returns:
        int: The number of chunk records deleted.

    Raises:
        DocumentNotFoundError: If no indexed chunk resolves to the
            requested document_id (the collection is left unchanged).
    """
    requested_id = str(document_id or "").strip()
    results = _collection.get(include=["metadatas"])
    record_ids = results.get("ids") or []
    metadatas = results.get("metadatas") or []

    matching_ids = [
        record_id
        for record_id, metadata in zip(record_ids, metadatas)
        if _resolve_record_document_id(metadata or {}) == requested_id
    ]
    if not matching_ids:
        raise DocumentNotFoundError(
            f"No indexed document matches document_id '{requested_id}'."
        )

    _collection.delete(ids=matching_ids)
    return len(matching_ids)


def index_chunks(chunks: list[Document], embeddings: list[list[float]]) -> int:
    """
    Stores one ChromaDB record per chunk: deterministic ID, chunk text,
    embedding, and metadata.

    Args:
        chunks (list[Document]): The document chunks to index.
        embeddings (list[list[float]]): One 384-dimensional embedding
            per chunk, in the same order.

    Returns:
        int: The number of records stored.

    Raises:
        ValueError: If the number of chunks does not match the number
            of embeddings.
    """
    if len(chunks) != len(embeddings):
        raise ValueError(
            f"Number of chunks ({len(chunks)}) must match the number of "
            f"embeddings ({len(embeddings)})."
        )

    ids = [build_chunk_id(chunk, i) for i, chunk in enumerate(chunks)]
    documents = [chunk.page_content for chunk in chunks]
    metadatas = [dict(chunk.metadata) for chunk in chunks]

    _collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )
    return len(ids)


def search(query: str, top_k: int = 4) -> list[dict]:
    """
    Retrieves the top_k most relevant chunks for a query from the
    persistent collection. Read-only: the index is not modified.

    Args:
        query (str): The user question to search for.
        top_k (int): Number of results to return.

    Returns:
        list[dict]: Ranked records, each with "id", "text", "metadata",
            and "distance".
    """
    query_embedding = embed_text(query)
    results = _collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    records = []
    for record_id, text, metadata, distance in zip(
        results["ids"][0],
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        records.append({
            "id": record_id,
            "text": text,
            "metadata": metadata,
            "distance": distance,
        })
    return records

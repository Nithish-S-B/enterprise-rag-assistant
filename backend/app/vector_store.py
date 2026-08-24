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
        source = str(metadata.get("source", ""))
        filename = _extract_filename(source)
        document_id = str(metadata.get("document_id") or "").strip()
        if not document_id:
            document_id = normalize_document_id(filename)

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

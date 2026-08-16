"""
Vector storage for the Enterprise RAG Assistant using ChromaDB.

Chunks and their embeddings are indexed into a persistent, disk-backed
collection. Each chunk becomes exactly one record containing a
deterministic ID, the chunk text, its embedding, and its metadata.
"""
import os

import chromadb
from langchain_core.documents import Document

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

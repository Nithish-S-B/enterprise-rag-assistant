"""Document upload endpoint: validate, ingest, embed, and index PDFs."""
import logging
import os
import re
import tempfile
from typing import Literal

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

from ..chunker import chunk_documents
from ..document_loader import load_pdf
from ..embeddings import embed_texts
from ..vector_store import index_chunks

logger = logging.getLogger(__name__)

router = APIRouter(tags=["documents"])

DEFAULT_MAX_UPLOAD_SIZE_MB = 25
BYTES_PER_MB = 1024 * 1024
READ_CHUNK_SIZE_BYTES = 64 * 1024
PDF_SIGNATURE = b"%PDF-"
SUPPORTED_EXTENSION = ".pdf"
TEMP_FILE_PREFIX = "rag_upload_"


class UploadIngestionResponse(BaseModel):
    """Summary of a successfully ingested and indexed PDF."""

    document_id: str
    filename: str
    pages: int
    chunks: int
    status: Literal["indexed"]


def _max_upload_bytes() -> int:
    """Resolve the configured upload ceiling, falling back to the default."""
    raw_value = os.getenv("MAX_UPLOAD_SIZE_MB", str(DEFAULT_MAX_UPLOAD_SIZE_MB))
    try:
        megabytes = int(raw_value)
    except ValueError:
        logger.warning(
            "Invalid MAX_UPLOAD_SIZE_MB=%r; using default of %d MB.",
            raw_value,
            DEFAULT_MAX_UPLOAD_SIZE_MB,
        )
        megabytes = DEFAULT_MAX_UPLOAD_SIZE_MB
    return max(1, megabytes) * BYTES_PER_MB


def _build_document_id(filename: str) -> str:
    """
    Derives a stable, filesystem-safe identifier from the uploaded filename.

    Example: "Employee Handbook.pdf" -> "employee_handbook"
    """
    stem = os.path.splitext(os.path.basename(filename))[0]
    normalized = re.sub(r"[^a-z0-9]+", "_", stem.lower()).strip("_")
    return normalized or "document"


def _ingest_pdf(temp_path: str, filename: str, document_id: str) -> tuple[int, int]:
    """
    Runs the existing load -> chunk -> embed -> index pipeline against a
    temporary PDF file.

    Returns:
        tuple[int, int]: (page_count, chunk_count)

    Raises:
        HTTPException: With a safe, client-facing message when parsing,
            chunking, embedding, or indexing fails.
    """
    try:
        pages = load_pdf(temp_path)
        if not pages:
            raise ValueError("The PDF contains no readable pages.")

        chunks = chunk_documents(pages)
        if not chunks:
            raise ValueError("The PDF contains no extractable text to index.")

        # Index under the original filename so chunk IDs are deterministic
        # across uploads (upserts replace prior records instead of duplicating).
        for chunk in chunks:
            chunk.metadata["source"] = filename
            chunk.metadata["document_id"] = document_id

        texts = [chunk.page_content for chunk in chunks]
        embeddings = embed_texts(texts)
        index_chunks(chunks, embeddings)
    except HTTPException:
        raise
    except Exception as error:
        logger.exception(
            "Ingestion failed for uploaded PDF '%s' (document_id=%s).",
            filename,
            document_id,
        )
        raise HTTPException(
            status_code=500,
            detail="Failed to process the uploaded PDF.",
        ) from error

    return len(pages), len(chunks)


@router.post("/upload", response_model=UploadIngestionResponse)
async def upload_document(file: UploadFile = File(...)) -> UploadIngestionResponse:
    """Validate an uploaded PDF, then ingest, embed, and index it."""
    filename = (file.filename or "").strip()
    if not filename:
        raise HTTPException(
            status_code=422,
            detail="A file with a non-empty filename is required.",
        )

    if not filename.lower().endswith(SUPPORTED_EXTENSION):
        raise HTTPException(
            status_code=415,
            detail="Unsupported file type. Only PDF files (.pdf) are accepted.",
        )

    max_bytes = _max_upload_bytes()
    chunks: list[bytes] = []
    total_size = 0
    try:
        while True:
            chunk = await file.read(READ_CHUNK_SIZE_BYTES)
            if not chunk:
                break
            total_size += len(chunk)
            if total_size > max_bytes:
                raise HTTPException(
                    status_code=413,
                    detail=(
                        "Uploaded file exceeds the maximum allowed size "
                        f"of {max_bytes // BYTES_PER_MB} MB."
                    ),
                )
            chunks.append(chunk)
    finally:
        await file.close()

    if total_size == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    payload = b"".join(chunks)
    if not payload.startswith(PDF_SIGNATURE):
        raise HTTPException(
            status_code=415,
            detail="File content does not look like a valid PDF document.",
        )

    document_id = _build_document_id(filename)
    temp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=TEMP_FILE_PREFIX,
            suffix=SUPPORTED_EXTENSION,
            delete=False,
        ) as handle:
            temp_path = handle.name
            handle.write(payload)

        pages, chunk_count = _ingest_pdf(temp_path, filename, document_id)
    finally:
        if temp_path is not None:
            try:
                os.unlink(temp_path)
            except OSError:
                logger.warning("Could not remove temporary file '%s'.", temp_path)

    logger.info(
        "Indexed PDF '%s' (document_id=%s, pages=%d, chunks=%d).",
        filename,
        document_id,
        pages,
        chunk_count,
    )
    return UploadIngestionResponse(
        document_id=document_id,
        filename=filename,
        pages=pages,
        chunks=chunk_count,
        status="indexed",
    )

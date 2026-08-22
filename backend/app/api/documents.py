"""Document upload endpoint with validation-only handling."""
import logging
import os

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(tags=["documents"])

DEFAULT_MAX_UPLOAD_SIZE_MB = 25
BYTES_PER_MB = 1024 * 1024
READ_CHUNK_SIZE_BYTES = 64 * 1024
PDF_SIGNATURE = b"%PDF-"
SUPPORTED_EXTENSION = ".pdf"


class UploadValidationResponse(BaseModel):
    """Result of validating an uploaded PDF before any processing."""

    filename: str
    size_bytes: int
    content_type: str | None
    message: str


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


@router.post("/upload", response_model=UploadValidationResponse)
async def upload_document(file: UploadFile = File(...)) -> UploadValidationResponse:
    """Validate an uploaded PDF without ingesting or persisting it."""
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

    logger.info("Validated PDF upload '%s' (%d bytes).", filename, total_size)
    return UploadValidationResponse(
        filename=filename,
        size_bytes=total_size,
        content_type=file.content_type,
        message="PDF upload validated successfully.",
    )

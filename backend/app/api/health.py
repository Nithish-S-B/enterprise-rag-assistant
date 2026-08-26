"""Health and readiness check endpoints.

* ``GET /api/health``  — liveness (is the process alive?)
* ``GET /api/ready``   — readiness (can the app serve requests now?)
"""
import os
from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(tags=["health"])

SERVICE_NAME = "enterprise-rag-assistant"


# ---------------------------------------------------------------------------
# Health (liveness)
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: Literal["ok"]
    service: str


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Report service liveness without touching the RAG stack."""
    return HealthResponse(status="ok", service=SERVICE_NAME)


# ---------------------------------------------------------------------------
# Readiness
# ---------------------------------------------------------------------------

class ReadyResponse(BaseModel):
    status: Literal["ready"]


@router.get("/ready", response_model=ReadyResponse)
async def readiness_check() -> ReadyResponse:
    """Verify that local dependencies are initialized and available.

    Checks (read-only, no side-effects):
    1. ChromaDB collection is reachable.
    2. Embedding model is loaded.
    3. Required application configuration is present.

    Does **not** call OpenRouter, generate text, or write to ChromaDB.
    """
    _check_vector_store()
    _check_embedding_model()
    _check_configuration()
    return ReadyResponse(status="ready")


def _check_vector_store() -> None:
    """Verify the ChromaDB collection can be read."""
    try:
        from ..vector_store import _collection
        _collection.count()
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail="Vector store is not available.",
        ) from exc


def _check_embedding_model() -> None:
    """Verify the embedding model is loaded."""
    try:
        from ..embeddings import _model
        if _model is None:
            raise RuntimeError("Model object is None")
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail="Embedding model is not available.",
        ) from exc


def _check_configuration() -> None:
    """Verify required application configuration is present."""
    required = ["OPENROUTER_MODEL"]
    missing = [key for key in required if not os.getenv(key)]
    if missing:
        raise HTTPException(
            status_code=503,
            detail="Required configuration is missing.",
        )

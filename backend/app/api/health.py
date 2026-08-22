"""Health check endpoints."""
from typing import Literal

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(tags=["health"])

SERVICE_NAME = "enterprise-rag-assistant"


class HealthResponse(BaseModel):
    status: Literal["ok"]
    service: str


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Report service liveness without touching the RAG stack."""
    return HealthResponse(status="ok", service=SERVICE_NAME)

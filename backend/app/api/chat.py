"""Chat endpoint exposing the grounded RAG pipeline."""
import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, field_validator

from ..rag import answer_question

logger = logging.getLogger(__name__)

router = APIRouter(tags=["chat"])


class ChatRequest(BaseModel):
    question: str
    final_k: int = Field(default=4, ge=1, le=10)

    @field_validator("question")
    @classmethod
    def require_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("question must contain non-whitespace text.")
        return value


class Citation(BaseModel):
    citation_id: str
    source: str
    page: int | None = None
    page_label: str
    chunk_id: str


class ChatResponse(BaseModel):
    answer: str
    citations: list[Citation]


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    """Answer a question using retrieval, reranking, and grounded generation."""
    try:
        result = answer_question(request.question, final_k=request.final_k)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except RuntimeError as error:
        logger.exception("LLM provider failure while answering question.")
        raise HTTPException(
            status_code=503,
            detail="Language model provider is currently unavailable.",
        ) from error
    except Exception as error:
        logger.exception("Unexpected failure while answering question.")
        raise HTTPException(status_code=500, detail="Internal server error.") from error

    return ChatResponse(answer=result["answer"], citations=result["citations"])

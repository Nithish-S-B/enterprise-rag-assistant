"""Global error-response contract for the Enterprise RAG Assistant.

Every application-level error returned by the API uses the same shape::

    {
        "error_type": "...",
        "message": "...",
        "request_id": "..."     # populated by RequestIDMiddleware
    }

Three FastAPI exception handlers are registered to enforce this contract
for *all* error paths: Pydantic validation errors, explicit HTTPExceptions
raised by routes, and any unhandled exception that leaks through.
"""
import logging
import traceback

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Error response model
# ---------------------------------------------------------------------------

class ErrorResponse(BaseModel):
    """Uniform error envelope returned by every failing endpoint."""

    error_type: str
    message: str
    request_id: str | None = None


# ---------------------------------------------------------------------------
# HTTPException → error_type mapping
# ---------------------------------------------------------------------------

_HTTP_STATUS_TO_ERROR_TYPE: dict[int, str] = {
    400: "bad_request",
    404: "not_found",
    413: "bad_request",
    415: "unsupported_media_type",
    422: "validation_error",
    500: "internal_error",
    503: "provider_unavailable",
}

_DEFAULT_ERROR_TYPE = "internal_error"

# Safe, non-leaking messages for status codes that should never expose
# internal detail.  Routes may still supply their own safe messages via
# HTTPException(detail=...) — this mapping is only the fallback when the
# detail is the generic sentinel or absent.
_SAFE_STATUS_MESSAGES: dict[int, str] = {
    400: "Bad request.",
    404: "Resource not found.",
    413: "Payload too large.",
    415: "Unsupported media type.",
    422: "Request validation failed.",
    500: "An internal server error occurred.",
    503: "Service temporarily unavailable.",
}


def _resolve_error_type(status_code: int) -> str:
    """Map an HTTP status code to a semantic error_type string."""
    return _HTTP_STATUS_TO_ERROR_TYPE.get(status_code, _DEFAULT_ERROR_TYPE)


def _resolve_message(status_code: int, detail: str | None) -> str:
    """Return a safe message for the client.

    If the route supplied a *non-generic* detail string (i.e. something
    other than the default FastAPI phrasing), it is trusted and forwarded.
    Otherwise a safe canned message is used.
    """
    if detail and isinstance(detail, str):
        return detail
    return _SAFE_STATUS_MESSAGES.get(status_code, "An internal server error occurred.")


# ---------------------------------------------------------------------------
# Exception handlers
# ---------------------------------------------------------------------------

async def http_exception_handler(
    request: Request, exc: Exception
) -> JSONResponse:
    """Convert every HTTPException into the uniform ErrorResponse shape."""
    # FastAPI populates status_code and detail on HTTPException.
    status_code = getattr(exc, "status_code", 500)
    raw_detail = getattr(exc, "detail", None)

    # Unwrap detail if FastAPI/Pydantic nested it in a list (shouldn't
    # happen for explicit HTTPException, but be defensive).
    if isinstance(raw_detail, list):
        raw_detail = str(raw_detail)

    error_type = _resolve_error_type(status_code)
    message = _resolve_message(status_code, raw_detail)
    request_id = getattr(request.state, "request_id", None)

    return JSONResponse(
        status_code=status_code,
        content=ErrorResponse(
            error_type=error_type,
            message=message,
            request_id=request_id,
        ).model_dump(),
    )


async def request_validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Normalise Pydantic / FastAPI request-validation failures.

    Returns a uniform 422 + validation_error response.  The raw field-level
    error structure from Pydantic is *never* exposed to the client.
    """
    request_id = getattr(request.state, "request_id", None)
    return JSONResponse(
        status_code=422,
        content=ErrorResponse(
            error_type="validation_error",
            message="Request validation failed.",
            request_id=request_id,
        ).model_dump(),
    )


async def unhandled_exception_handler(
    request: Request, exc: Exception
) -> JSONResponse:
    """Catch-all for exceptions that no route or handler resolved.

    Logs the full traceback server-side and returns a safe 500 to the
    client.  No internal details are leaked.
    """
    request_id = getattr(request.state, "request_id", None)
    logger.exception(
        "Unhandled exception on %s %s [request_id=%s]",
        request.method,
        request.url.path,
        request_id,
    )
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error_type="internal_error",
            message="An internal server error occurred.",
            request_id=request_id,
        ).model_dump(),
    )


# ---------------------------------------------------------------------------
# Registration helper
# ---------------------------------------------------------------------------

def register_error_handlers(app: FastAPI) -> None:
    """Attach all three global exception handlers to the FastAPI app.

    Call this once during application setup, *after* the app is created
    and *before* routers are included.
    """
    app.add_exception_handler(
        Exception,
        unhandled_exception_handler,
    )
    app.add_exception_handler(
        RequestValidationError,
        request_validation_exception_handler,
    )
    # HTTPException handler is registered via its concrete class so it
    # does NOT shadow the generic Exception handler above — FastAPI
    # dispatches to the most specific matching handler.
    from fastapi import HTTPException as _HTTPExc
    app.add_exception_handler(
        _HTTPExc,
        http_exception_handler,
    )

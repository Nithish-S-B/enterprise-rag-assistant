"""Request logging middleware for the Enterprise RAG Assistant.

Logs a single line for every completed HTTP request using the existing
``request_id`` assigned by :class:`RequestIDMiddleware`.

Log format::

    METHOD PATH -> STATUS | duration_ms=... | request_id=...

The middleware:
* Captures a high-resolution start time before calling downstream.
* Reads ``request.state.request_id`` (set by RequestIDMiddleware) with
  a safe ``"-"`` fallback.
* Never modifies the response body, status code, or headers.
* Re-raises any exception so downstream error handlers can deal with it.
"""
import logging
import time

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware that logs method, path, status, duration, and request_id."""

    async def dispatch(self, request: Request, call_next) -> Response:
        start = time.perf_counter()

        response = await call_next(request)

        elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
        request_id = getattr(request.state, "request_id", None) or "-"
        method = request.method
        path = request.url.path
        status = response.status_code

        logger.info(
            "%s %s -> %d | duration_ms=%.2f | request_id=%s",
            method,
            path,
            status,
            elapsed_ms,
            request_id,
        )

        return response

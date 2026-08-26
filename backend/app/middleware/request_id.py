"""Request-ID middleware for the Enterprise RAG Assistant.

Assigns a unique ``request_id`` to every inbound request so that
clients and operators can correlate logs, errors, and retries.

Behaviour:
* If the client supplies a valid ``X-Request-ID`` header (UUID4 format),
  it is reused for traceability across services.
* Otherwise a fresh UUID4 is generated server-side.
* The ID is stored in ``request.state.request_id`` so that route
  handlers and global error handlers can read it without importing the
  middleware.
* On every response the ID is echoed back via the ``X-Request-ID``
  response header, and included in the JSON body of error responses
  by the error handlers in ``errors.py``.
"""
import logging
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger(__name__)

# Header name used for both request and response.
_REQUEST_ID_HEADER = "X-Request-ID"


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Middleware that injects a ``request_id`` into every request."""

    async def dispatch(self, request: Request, call_next) -> Response:
        # --- Resolve request ID -------------------------------------------------
        raw = request.headers.get(_REQUEST_ID_HEADER, "").strip()
        if raw:
            # Validate: must be a valid UUID4.
            try:
                request_id = str(uuid.UUID(raw, version=4))
            except ValueError:
                # Client sent a non-UUID value; ignore and generate our own.
                logger.debug(
                    "Invalid X-Request-ID header %r - generating new ID", raw
                )
                request_id = str(uuid.uuid4())
        else:
            request_id = str(uuid.uuid4())

        # Make available to downstream handlers (routes + error handlers).
        request.state.request_id = request_id

        # --- Forward request ----------------------------------------------------
        response = await call_next(request)

        # --- Set response header -----------------------------------------------
        # Only set the header when Starlette has not already committed a
        # streaming/error response where headers cannot be modified.
        try:
            response.headers[_REQUEST_ID_HEADER] = request_id
        except Exception:
            # Response already sent (streaming, WebSocket, etc.).
            pass

        return response

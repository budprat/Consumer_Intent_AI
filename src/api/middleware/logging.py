# ABOUTME: Request/response logging middleware with structured logging
# ABOUTME: Logs all API requests with timing, status codes, and error tracking

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from typing import Callable
import time
import logging

from ..config import settings

logger = logging.getLogger(__name__)


class LoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware for structured request/response logging.

    Logs:
    - Request method, path, client IP
    - Response status code, processing time
    - Request/response body (in debug mode)
    - API key prefix (for tracking)
    """

    async def dispatch(self, request: Request, call_next: Callable):
        """
        Log request and response details.

        Args:
            request: FastAPI request object
            call_next: Next middleware/route handler

        Returns:
            Response from next handler
        """
        # Start timer
        start_time = time.time()

        # Extract request metadata
        client_ip = request.client.host if request.client else "unknown"
        method = request.method
        path = request.url.path
        query_params = dict(request.query_params)

        # Get API key prefix (if available after auth middleware)
        api_key = getattr(request.state, "api_key", None)
        api_key_prefix = api_key[:8] + "..." if api_key else "none"

        # Log request
        logger.info(
            f"Request started: {method} {path}",
            extra={
                "event": "request_started",
                "method": method,
                "path": path,
                "query_params": query_params,
                "client_ip": client_ip,
                "api_key_prefix": api_key_prefix,
                "user_agent": request.headers.get("user-agent", "unknown"),
            },
        )

        # In debug mode, log request body
        if settings.DEBUG and method in ["POST", "PUT", "PATCH"]:
            try:
                body = await request.body()
                logger.debug(
                    "Request body",
                    extra={
                        "path": path,
                        "body_size": len(body),
                        "body_preview": body[:1000].decode("utf-8", errors="ignore"),
                    },
                )
            except Exception as e:
                logger.warning(f"Failed to read request body: {e}")

        # Process request
        try:
            response = await call_next(request)
            status_code = response.status_code
            error = None
        except Exception as e:
            status_code = 500
            error = str(e)
            logger.error(
                f"Request failed with exception: {e}",
                exc_info=True,
                extra={
                    "event": "request_exception",
                    "method": method,
                    "path": path,
                    "client_ip": client_ip,
                    "api_key_prefix": api_key_prefix,
                },
            )
            raise

        # Calculate processing time
        processing_time = time.time() - start_time

        # Determine log level based on status code
        if status_code >= 500:
            log_level = logging.ERROR
            event = "request_server_error"
        elif status_code >= 400:
            log_level = logging.WARNING
            event = "request_client_error"
        else:
            log_level = logging.INFO
            event = "request_completed"

        # Log response
        logger.log(
            log_level,
            f"Request completed: {method} {path} - {status_code} ({processing_time:.3f}s)",
            extra={
                "event": event,
                "method": method,
                "path": path,
                "status_code": status_code,
                "processing_time_ms": round(processing_time * 1000, 2),
                "client_ip": client_ip,
                "api_key_prefix": api_key_prefix,
                "error": error,
            },
        )

        # Warn if request took too long
        if processing_time > 5.0:
            logger.warning(
                f"Slow request detected: {method} {path} took {processing_time:.3f}s",
                extra={
                    "event": "slow_request",
                    "method": method,
                    "path": path,
                    "processing_time_ms": round(processing_time * 1000, 2),
                    "threshold_ms": 5000,
                },
            )

        return response

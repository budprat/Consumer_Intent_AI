# ABOUTME: API key authentication middleware
# ABOUTME: Validates API keys for all protected endpoints

from fastapi import Request, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from typing import Callable
import logging

from ..config import settings

logger = logging.getLogger(__name__)


class APIKeyMiddleware(BaseHTTPMiddleware):
    """
    Middleware for API key authentication.

    Validates API keys from request headers against configured keys.
    Exempt paths (health, metrics, docs) bypass authentication.
    """

    # Paths that don't require authentication
    EXEMPT_PATHS = [
        "/health",
        "/metrics",
        "/docs",
        "/redoc",
        "/openapi.json",
        "/",
    ]

    async def dispatch(self, request: Request, call_next: Callable):
        """
        Process request and validate API key.

        Args:
            request: FastAPI request object
            call_next: Next middleware/route handler

        Returns:
            Response from next handler or 401/403 error

        Raises:
            HTTPException: If API key is missing or invalid
        """
        # Skip authentication for OPTIONS requests (CORS preflight)
        if request.method == "OPTIONS":
            return await call_next(request)

        # Skip authentication for exempt paths
        if self._is_exempt(request.url.path):
            return await call_next(request)

        # Extract API key from header
        api_key = request.headers.get(settings.API_KEY_HEADER)

        if not api_key:
            logger.warning(
                "Request without API key",
                extra={
                    "path": request.url.path,
                    "client_ip": request.client.host if request.client else None,
                },
            )
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={
                    "detail": f"Missing API key in header: {settings.API_KEY_HEADER}",
                    "hint": "Include your API key in the request header",
                },
            )

        # Validate API key
        if not settings.validate_api_key(api_key):
            logger.warning(
                "Request with invalid API key",
                extra={
                    "path": request.url.path,
                    "client_ip": request.client.host if request.client else None,
                    "api_key_prefix": api_key[:8] + "...",
                },
            )
            return JSONResponse(
                status_code=status.HTTP_403_FORBIDDEN,
                content={
                    "detail": "Invalid API key",
                    "hint": "Check your API key or contact support",
                },
            )

        # Store API key in request state for downstream use
        request.state.api_key = api_key

        logger.debug(
            "API key validated",
            extra={
                "path": request.url.path,
                "api_key_prefix": api_key[:8] + "...",
            },
        )

        # Proceed to next middleware/handler
        return await call_next(request)

    def _is_exempt(self, path: str) -> bool:
        """
        Check if path is exempt from authentication.

        Args:
            path: Request URL path

        Returns:
            True if exempt, False otherwise
        """
        return any(path.startswith(exempt_path) for exempt_path in self.EXEMPT_PATHS)

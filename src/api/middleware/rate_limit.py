# ABOUTME: Rate limiting middleware with sliding window algorithm
# ABOUTME: Enforces per-API-key rate limits (minute, hour, day) and usage quotas

from fastapi import Request, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from typing import Callable, Dict, Tuple
from collections import defaultdict, deque
import time
import logging

from ..config import settings

logger = logging.getLogger(__name__)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Middleware for rate limiting using sliding window algorithm.

    Tracks requests per API key across three time windows:
    - Per minute (60 requests default)
    - Per hour (1000 requests default)
    - Per day (10000 requests default)

    Uses in-memory storage (production should use Redis).
    """

    def __init__(self, app):
        """
        Initialize rate limiter with empty request tracking.

        Args:
            app: FastAPI application instance
        """
        super().__init__(app)

        # Request timestamps per API key
        # Format: {api_key: [(timestamp, endpoint), ...]}
        self._request_log: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=settings.RATE_LIMIT_REQUESTS_PER_DAY)
        )

        # Time windows in seconds
        self._windows = {
            "minute": (60, settings.RATE_LIMIT_REQUESTS_PER_MINUTE),
            "hour": (3600, settings.RATE_LIMIT_REQUESTS_PER_HOUR),
            "day": (86400, settings.RATE_LIMIT_REQUESTS_PER_DAY),
        }

        logger.info(
            "Rate limiting initialized",
            extra={
                "limits": {
                    "per_minute": settings.RATE_LIMIT_REQUESTS_PER_MINUTE,
                    "per_hour": settings.RATE_LIMIT_REQUESTS_PER_HOUR,
                    "per_day": settings.RATE_LIMIT_REQUESTS_PER_DAY,
                }
            },
        )

    async def dispatch(self, request: Request, call_next: Callable):
        """
        Check rate limits before processing request.

        Args:
            request: FastAPI request object
            call_next: Next middleware/route handler

        Returns:
            Response from next handler or 429 rate limit error
        """
        # Skip rate limiting for exempt paths
        if self._is_exempt(request.url.path):
            return await call_next(request)

        # Get API key from request state (set by auth middleware)
        api_key = getattr(request.state, "api_key", None)
        if not api_key:
            # If no API key, skip rate limiting (will be caught by auth middleware)
            return await call_next(request)

        # Check rate limits
        is_allowed, limit_info = self._check_rate_limit(api_key, request.url.path)

        if not is_allowed:
            logger.warning(
                "Rate limit exceeded",
                extra={
                    "api_key_prefix": api_key[:8] + "...",
                    "path": request.url.path,
                    "window": limit_info["window"],
                    "limit": limit_info["limit"],
                    "current": limit_info["current"],
                },
            )

            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "detail": "Rate limit exceeded",
                    "window": limit_info["window"],
                    "limit": limit_info["limit"],
                    "current": limit_info["current"],
                    "retry_after": limit_info["retry_after"],
                },
                headers={
                    "Retry-After": str(limit_info["retry_after"]),
                    "X-RateLimit-Limit": str(limit_info["limit"]),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(
                        int(time.time()) + limit_info["retry_after"]
                    ),
                },
            )

        # Record request
        self._record_request(api_key, request.url.path)

        # Add rate limit headers to response
        response = await call_next(request)

        # Calculate remaining requests
        remaining = self._get_remaining_requests(api_key)
        response.headers["X-RateLimit-Limit-Minute"] = str(
            settings.RATE_LIMIT_REQUESTS_PER_MINUTE
        )
        response.headers["X-RateLimit-Limit-Hour"] = str(
            settings.RATE_LIMIT_REQUESTS_PER_HOUR
        )
        response.headers["X-RateLimit-Limit-Day"] = str(
            settings.RATE_LIMIT_REQUESTS_PER_DAY
        )
        response.headers["X-RateLimit-Remaining-Minute"] = str(remaining["minute"])
        response.headers["X-RateLimit-Remaining-Hour"] = str(remaining["hour"])
        response.headers["X-RateLimit-Remaining-Day"] = str(remaining["day"])

        return response

    def _check_rate_limit(
        self, api_key: str, endpoint: str
    ) -> Tuple[bool, Dict[str, any]]:
        """
        Check if request is within rate limits.

        Args:
            api_key: API key making the request
            endpoint: Request endpoint path

        Returns:
            Tuple of (is_allowed, limit_info_dict)
        """
        current_time = time.time()
        request_log = self._request_log[api_key]

        # Check each time window
        for window_name, (window_seconds, limit) in self._windows.items():
            # Count requests within window
            window_start = current_time - window_seconds
            count = sum(1 for timestamp, _ in request_log if timestamp >= window_start)

            if count >= limit:
                # Calculate retry after (time until oldest request expires)
                oldest_in_window = next(
                    (t for t, _ in request_log if t >= window_start), None
                )
                retry_after = (
                    int(window_seconds - (current_time - oldest_in_window) + 1)
                    if oldest_in_window
                    else window_seconds
                )

                return False, {
                    "window": window_name,
                    "limit": limit,
                    "current": count,
                    "retry_after": retry_after,
                }

        return True, {}

    def _record_request(self, api_key: str, endpoint: str) -> None:
        """
        Record request timestamp for rate limiting.

        Args:
            api_key: API key making the request
            endpoint: Request endpoint path
        """
        current_time = time.time()
        self._request_log[api_key].append((current_time, endpoint))

        # Clean up old entries (beyond day window)
        window_start = current_time - self._windows["day"][0]
        self._request_log[api_key] = deque(
            (t, e) for t, e in self._request_log[api_key] if t >= window_start
        )

    def _get_remaining_requests(self, api_key: str) -> Dict[str, int]:
        """
        Calculate remaining requests for each time window.

        Args:
            api_key: API key to check

        Returns:
            Dictionary with remaining requests per window
        """
        current_time = time.time()
        request_log = self._request_log[api_key]
        remaining = {}

        for window_name, (window_seconds, limit) in self._windows.items():
            window_start = current_time - window_seconds
            count = sum(1 for timestamp, _ in request_log if timestamp >= window_start)
            remaining[window_name] = max(0, limit - count)

        return remaining

    def _is_exempt(self, path: str) -> bool:
        """
        Check if path is exempt from rate limiting.

        Args:
            path: Request URL path

        Returns:
            True if exempt, False otherwise
        """
        exempt_paths = ["/health", "/metrics", "/docs", "/redoc", "/openapi.json", "/"]
        return any(path.startswith(exempt_path) for exempt_path in exempt_paths)

    def get_usage_stats(self, api_key: str) -> Dict[str, any]:
        """
        Get detailed usage statistics for an API key.

        Args:
            api_key: API key to analyze

        Returns:
            Dictionary with usage statistics across all windows
        """
        current_time = time.time()
        request_log = self._request_log[api_key]
        stats = {
            "total_requests_tracked": len(request_log),
            "windows": {},
        }

        for window_name, (window_seconds, limit) in self._windows.items():
            window_start = current_time - window_seconds
            requests_in_window = [(t, e) for t, e in request_log if t >= window_start]

            stats["windows"][window_name] = {
                "limit": limit,
                "used": len(requests_in_window),
                "remaining": max(0, limit - len(requests_in_window)),
                "percentage": (len(requests_in_window) / limit * 100)
                if limit > 0
                else 0,
            }

        return stats

# ABOUTME: FastAPI application entry point with middleware configuration
# ABOUTME: Bootstraps the server, routes, middleware, and background task processing

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from contextlib import asynccontextmanager
import logging
from typing import AsyncGenerator

from .config import settings
from .middleware.auth import APIKeyMiddleware
from .middleware.rate_limit import RateLimitMiddleware
from .middleware.logging import LoggingMiddleware
from .routes import surveys, health, metrics

# Initialize logging
logging.basicConfig(level=settings.LOG_LEVEL)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """
    Application lifespan context manager.

    Handles startup and shutdown logic:
    - Startup: Initialize connections, load reference sets, start task workers
    - Shutdown: Clean up connections, flush logs, stop task workers
    """
    # ============================================================================
    # Startup
    # ============================================================================
    logger.info(
        "Starting Synthetic Consumer SSR API",
        extra={
            "version": settings.APP_VERSION,
            "environment": settings.ENV,
            "host": settings.HOST,
            "port": settings.PORT,
        },
    )

    # Initialize OpenAI client (if API key provided)
    if settings.OPENAI_API_KEY:
        logger.info("OpenAI API key detected - LLM and embeddings enabled")
    else:
        logger.warning(
            "No OpenAI API key found - API will run in demo mode without actual LLM calls"
        )

    # Initialize task queue backend
    if settings.TASK_QUEUE_BACKEND == "redis" and settings.REDIS_URL:
        logger.info(f"Initializing Redis task queue: {settings.REDIS_URL}")
        # TODO: Initialize Redis connection pool
    elif settings.TASK_QUEUE_BACKEND == "celery":
        logger.info("Initializing Celery task queue")
        # TODO: Initialize Celery worker
    else:
        logger.info("Using in-memory task queue (development mode)")

    # Load default reference sets
    logger.info("Loading default reference statement sets")
    # TODO: Pre-load reference sets into memory cache

    # Initialize metrics collector (if enabled)
    if settings.METRICS_ENABLED:
        logger.info(f"Starting Prometheus metrics on port {settings.METRICS_PORT}")
        # TODO: Initialize Prometheus client

    # Initialize Sentry (if enabled in production)
    if settings.SENTRY_DSN and settings.is_production():
        import sentry_sdk

        sentry_sdk.init(
            dsn=settings.SENTRY_DSN,
            traces_sample_rate=settings.SENTRY_TRACES_SAMPLE_RATE,
            environment=settings.ENV,
        )
        logger.info("Sentry error tracking initialized")

    logger.info("API startup complete")

    yield

    # ============================================================================
    # Shutdown
    # ============================================================================
    logger.info("Shutting down API")

    # Close database connections
    # TODO: Close Redis/Celery connections

    # Flush logs
    logging.shutdown()

    logger.info("API shutdown complete")


# Initialize FastAPI application
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="""
# Synthetic Consumer SSR API

Production API for generating synthetic consumer survey responses using
Semantic Similarity Rating (SSR) methodology.

## Key Features

- **High Reliability**: Achieves 90% of human test-retest reliability (ρ ≥ 0.90)
- **Demographic Conditioning**: Integrates US Census 2020 distributions
- **Multi-Model Support**: GPT-4o and Gemini-2.0-flash
- **Advanced Averaging**: 5 averaging strategies for robustness
- **Bias Detection**: Automatic demographic bias analysis
- **Quality Metrics**: Reference statement quality assessment

## Authentication

All endpoints require API key authentication via `X-API-Key` header.

## Rate Limits

- 60 requests/minute
- 1,000 requests/hour
- 10,000 requests/day

## Quotas

- 1,000 surveys/month (default)
- 100,000 synthetic responses/month (default)

Contact your account manager for quota increases.
    """,
    docs_url="/docs" if not settings.is_production() else None,
    redoc_url="/redoc" if not settings.is_production() else None,
    openapi_url="/openapi.json" if not settings.is_production() else None,
    lifespan=lifespan,
)

# ============================================================================
# Middleware Configuration
# ============================================================================

# CORS Middleware (must be first)
app.add_middleware(
    CORSMiddleware,
    **settings.get_cors_config(),
)

# GZip Compression
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Custom Middleware (order matters)
app.add_middleware(LoggingMiddleware)  # Log all requests

if settings.RATE_LIMIT_ENABLED:
    app.add_middleware(RateLimitMiddleware)  # Rate limiting

app.add_middleware(APIKeyMiddleware)  # Authentication (should be after rate limiting)


# ============================================================================
# Exception Handlers
# ============================================================================


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """
    Handle request validation errors with detailed messages.

    Args:
        request: FastAPI request object
        exc: Validation error exception

    Returns:
        JSON response with validation error details
    """
    errors = []
    for error in exc.errors():
        errors.append(
            {
                "loc": list(error["loc"]),
                "msg": error["msg"],
                "type": error["type"],
            }
        )

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "detail": "Request validation failed",
            "errors": errors,
        },
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Handle uncaught exceptions with logging and user-friendly messages.

    Args:
        request: FastAPI request object
        exc: Exception object

    Returns:
        JSON response with error details
    """
    logger.error(
        f"Unhandled exception: {exc}",
        exc_info=True,
        extra={
            "path": request.url.path,
            "method": request.method,
            "client_ip": request.client.host if request.client else None,
        },
    )

    # In production, don't expose internal error details
    if settings.is_production():
        message = "An internal server error occurred. Please contact support."
    else:
        message = str(exc)

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": message,
            "type": type(exc).__name__,
        },
    )


# ============================================================================
# Route Registration
# ============================================================================

# Health check and metrics (no auth required)
app.include_router(health.router, tags=["Health"])
app.include_router(metrics.router, tags=["Metrics"])

# API routes (auth required)
app.include_router(
    surveys.router,
    prefix="/api/v1",
    tags=["Surveys"],
)


# ============================================================================
# Root Endpoint
# ============================================================================


@app.get("/", include_in_schema=False)
async def root() -> dict:
    """
    Root endpoint with API information.

    Returns:
        API metadata and status
    """
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "operational",
        "docs_url": "/docs" if not settings.is_production() else None,
        "health_url": settings.HEALTH_CHECK_PATH,
        "environment": settings.ENV,
    }


# ============================================================================
# Application Entry Point
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower(),
        access_log=True,
    )

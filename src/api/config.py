# ABOUTME: API configuration settings with environment variable support
# ABOUTME: Centralizes all API-level configuration including auth, rate limiting, and monitoring

from typing import Optional, List
from pydantic import Field
from pydantic_settings import BaseSettings
from pathlib import Path


class APISettings(BaseSettings):
    """
    API configuration settings loaded from environment variables.

    Provides centralized configuration for:
    - Server settings (host, port, debug mode)
    - Security (API keys, CORS, JWT)
    - Rate limiting and quotas
    - Monitoring and observability
    - Async processing (task queue)
    """

    # ============================================================================
    # Server Configuration
    # ============================================================================

    APP_NAME: str = Field(
        default="Synthetic Consumer SSR API",
        description="Application name displayed in OpenAPI docs",
    )

    APP_VERSION: str = Field(
        default="1.0.0", description="API version for OpenAPI spec"
    )

    HOST: str = Field(default="0.0.0.0", description="Server host address")

    PORT: int = Field(default=8000, description="Server port number")

    DEBUG: bool = Field(
        default=False, description="Enable debug mode (auto-reload, detailed errors)"
    )

    ENV: str = Field(
        default="development",
        description="Environment: development, staging, production",
    )

    # ============================================================================
    # Security Configuration
    # ============================================================================

    API_KEY_HEADER: str = Field(
        default="X-API-Key", description="Header name for API key authentication"
    )

    API_KEYS: List[str] = Field(
        default_factory=list,
        description="Valid API keys for authentication (comma-separated in env)",
    )

    JWT_SECRET_KEY: Optional[str] = Field(
        default=None,
        description="Secret key for JWT token signing (future enhancement)",
    )

    JWT_ALGORITHM: str = Field(default="HS256", description="JWT signing algorithm")

    JWT_EXPIRATION_MINUTES: int = Field(
        default=60, description="JWT token expiration time in minutes"
    )

    # CORS Configuration
    CORS_ORIGINS: List[str] = Field(
        default=["*"], description="Allowed CORS origins (comma-separated in env)"
    )

    CORS_ALLOW_CREDENTIALS: bool = Field(
        default=True, description="Allow credentials in CORS requests"
    )

    CORS_ALLOW_METHODS: List[str] = Field(
        default=["*"], description="Allowed HTTP methods for CORS"
    )

    CORS_ALLOW_HEADERS: List[str] = Field(
        default=["*"], description="Allowed headers for CORS"
    )

    # ============================================================================
    # Rate Limiting Configuration
    # ============================================================================

    RATE_LIMIT_ENABLED: bool = Field(
        default=True, description="Enable rate limiting middleware"
    )

    RATE_LIMIT_REQUESTS_PER_MINUTE: int = Field(
        default=60, description="Maximum requests per minute per API key"
    )

    RATE_LIMIT_REQUESTS_PER_HOUR: int = Field(
        default=1000, description="Maximum requests per hour per API key"
    )

    RATE_LIMIT_REQUESTS_PER_DAY: int = Field(
        default=10000, description="Maximum requests per day per API key"
    )

    # Usage Quotas
    QUOTA_ENABLED: bool = Field(default=True, description="Enable usage quota tracking")

    QUOTA_DEFAULT_MONTHLY_SURVEYS: int = Field(
        default=1000, description="Default monthly survey quota per API key"
    )

    QUOTA_DEFAULT_MONTHLY_RESPONSES: int = Field(
        default=100000,
        description="Default monthly synthetic response quota per API key",
    )

    # ============================================================================
    # Async Processing Configuration
    # ============================================================================

    TASK_QUEUE_BACKEND: str = Field(
        default="memory", description="Task queue backend: memory, redis, celery"
    )

    REDIS_URL: Optional[str] = Field(
        default=None,
        description="Redis URL for task queue and caching (redis://localhost:6379/0)",
    )

    CELERY_BROKER_URL: Optional[str] = Field(
        default=None, description="Celery broker URL for distributed task processing"
    )

    CELERY_RESULT_BACKEND: Optional[str] = Field(
        default=None, description="Celery result backend URL"
    )

    MAX_CONCURRENT_TASKS: int = Field(
        default=10, description="Maximum concurrent background tasks"
    )

    TASK_TIMEOUT_SECONDS: int = Field(
        default=3600, description="Task execution timeout (1 hour default)"
    )

    # ============================================================================
    # Monitoring & Observability
    # ============================================================================

    LOG_LEVEL: str = Field(
        default="INFO",
        description="Logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL",
    )

    LOG_FORMAT: str = Field(default="json", description="Log format: json, text")

    LOG_FILE: Optional[Path] = Field(
        default=None, description="Log file path (None for stdout only)"
    )

    METRICS_ENABLED: bool = Field(
        default=True, description="Enable Prometheus metrics endpoint"
    )

    METRICS_PORT: int = Field(
        default=9090, description="Prometheus metrics exporter port"
    )

    HEALTH_CHECK_PATH: str = Field(
        default="/health", description="Health check endpoint path"
    )

    SENTRY_DSN: Optional[str] = Field(
        default=None, description="Sentry DSN for error tracking (production)"
    )

    SENTRY_TRACES_SAMPLE_RATE: float = Field(
        default=0.1, description="Sentry transaction sampling rate (0.0-1.0)"
    )

    # ============================================================================
    # OpenAI Configuration (from existing system)
    # ============================================================================

    OPENAI_API_KEY: Optional[str] = Field(
        default=None, description="OpenAI API key for embeddings and LLM calls"
    )

    OPENAI_ORG_ID: Optional[str] = Field(
        default=None, description="OpenAI organization ID (optional)"
    )

    # ============================================================================
    # Data Storage Configuration
    # ============================================================================

    DATA_DIR: Path = Field(
        default=Path("./data"), description="Base directory for data storage"
    )

    CACHE_DIR: Path = Field(
        default=Path("./data/cache"),
        description="Directory for caching embeddings and results",
    )

    REFERENCE_SETS_DIR: Path = Field(
        default=Path("./data/reference_sets"),
        description="Directory for reference statement sets",
    )

    RESULTS_DIR: Path = Field(
        default=Path("./data/results"),
        description="Directory for survey results and reports",
    )

    # ============================================================================
    # Feature Flags
    # ============================================================================

    ENABLE_DEMOGRAPHICS: bool = Field(
        default=True,
        description="Enable demographic conditioning (CRITICAL for 90% reliability)",
    )

    ENABLE_MULTI_SET_AVERAGING: bool = Field(
        default=True, description="Enable multi-reference set averaging for robustness"
    )

    ENABLE_BIAS_DETECTION: bool = Field(
        default=True, description="Enable demographic bias detection and reporting"
    )

    ENABLE_QUALITY_METRICS: bool = Field(
        default=True, description="Enable reference statement quality analysis"
    )

    # ============================================================================
    # Configuration Methods
    # ============================================================================

    class Config:
        """Pydantic settings configuration."""

        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
        # Support comma-separated lists in environment variables
        env_nested_delimiter = "__"

        @classmethod
        def parse_env_var(cls, field_name: str, raw_val: str):
            """Parse environment variables with special handling for lists."""
            if field_name in [
                "API_KEYS",
                "CORS_ORIGINS",
                "CORS_ALLOW_METHODS",
                "CORS_ALLOW_HEADERS",
            ]:
                return [item.strip() for item in raw_val.split(",") if item.strip()]
            return raw_val

    def __init__(self, **kwargs):
        """Initialize settings and create required directories."""
        super().__init__(**kwargs)
        self._ensure_directories()

    def _ensure_directories(self) -> None:
        """Create required directories if they don't exist."""
        for dir_path in [
            self.DATA_DIR,
            self.CACHE_DIR,
            self.REFERENCE_SETS_DIR,
            self.RESULTS_DIR,
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)

    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.ENV.lower() == "production"

    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.ENV.lower() == "development"

    def get_redis_config(self) -> dict:
        """Get Redis configuration dictionary."""
        if not self.REDIS_URL:
            return {"host": "localhost", "port": 6379, "db": 0}

        # Parse Redis URL
        from urllib.parse import urlparse

        parsed = urlparse(self.REDIS_URL)
        return {
            "host": parsed.hostname or "localhost",
            "port": parsed.port or 6379,
            "db": int(parsed.path.lstrip("/")) if parsed.path else 0,
            "password": parsed.password,
        }

    def get_cors_config(self) -> dict:
        """Get CORS configuration dictionary for FastAPI."""
        return {
            "allow_origins": self.CORS_ORIGINS,
            "allow_credentials": self.CORS_ALLOW_CREDENTIALS,
            "allow_methods": self.CORS_ALLOW_METHODS,
            "allow_headers": self.CORS_ALLOW_HEADERS,
        }

    def validate_api_key(self, api_key: str) -> bool:
        """
        Validate an API key against configured keys.

        Args:
            api_key: API key to validate

        Returns:
            True if valid, False otherwise
        """
        if not self.API_KEYS:
            # If no API keys configured, allow all (development mode)
            return self.is_development()

        return api_key in self.API_KEYS

    def get_log_config(self) -> dict:
        """
        Get logging configuration dictionary.

        Returns:
            Logging configuration for uvicorn/python logging
        """
        config = {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                },
                "json": {
                    "()": "pythonjsonlogger.jsonlogger.JsonFormatter",
                    "format": "%(asctime)s %(name)s %(levelname)s %(message)s",
                },
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "formatter": "json" if self.LOG_FORMAT == "json" else "default",
                    "stream": "ext://sys.stdout",
                },
            },
            "root": {
                "level": self.LOG_LEVEL,
                "handlers": ["console"],
            },
        }

        if self.LOG_FILE:
            config["handlers"]["file"] = {
                "class": "logging.handlers.RotatingFileHandler",
                "formatter": "json" if self.LOG_FORMAT == "json" else "default",
                "filename": str(self.LOG_FILE),
                "maxBytes": 10485760,  # 10MB
                "backupCount": 5,
            }
            config["root"]["handlers"].append("file")

        return config


# Global settings instance
settings = APISettings()

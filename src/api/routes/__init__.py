# ABOUTME: API routes package initialization
# ABOUTME: Exports all route routers for FastAPI registration

from . import health, metrics, surveys

__all__ = ["health", "metrics", "surveys"]

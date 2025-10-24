# ABOUTME: Health check endpoints for monitoring and load balancers
# ABOUTME: Provides liveness and readiness probes for orchestration

from fastapi import APIRouter, status
from fastapi.responses import JSONResponse
from typing import Dict, Any
import time
import psutil
import os

from ..config import settings

router = APIRouter()


@router.get("/health", status_code=status.HTTP_200_OK, tags=["Health"])
async def health_check() -> Dict[str, Any]:
    """
    Basic health check endpoint.

    Returns:
        Health status and basic system info
    """
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "environment": settings.ENV,
        "version": settings.APP_VERSION,
    }


@router.get("/health/live", status_code=status.HTTP_200_OK, tags=["Health"])
async def liveness_probe() -> Dict[str, str]:
    """
    Kubernetes liveness probe.

    Indicates if the application is running.

    Returns:
        Liveness status
    """
    return {"status": "alive"}


@router.get("/health/ready", status_code=status.HTTP_200_OK, tags=["Health"])
async def readiness_probe() -> Dict[str, Any]:
    """
    Kubernetes readiness probe.

    Indicates if the application is ready to serve traffic.
    Checks:
    - OpenAI API key configured
    - Required directories exist
    - System resources available

    Returns:
        Readiness status and checks
    """
    checks = {}

    # Check OpenAI API key
    checks["openai_api_key"] = {
        "configured": settings.OPENAI_API_KEY is not None,
        "status": "ready" if settings.OPENAI_API_KEY else "warning",
        "message": "API key configured"
        if settings.OPENAI_API_KEY
        else "No API key - demo mode only",
    }

    # Check directories
    required_dirs = [
        settings.DATA_DIR,
        settings.CACHE_DIR,
        settings.REFERENCE_SETS_DIR,
        settings.RESULTS_DIR,
    ]
    dirs_exist = all(d.exists() for d in required_dirs)
    checks["directories"] = {
        "status": "ready" if dirs_exist else "not_ready",
        "message": "All directories exist"
        if dirs_exist
        else "Some directories missing",
    }

    # Check system resources
    try:
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage("/")

        checks["system_resources"] = {
            "status": "ready"
            if memory.percent < 90 and disk.percent < 90
            else "warning",
            "cpu_percent": cpu_percent,
            "memory_percent": memory.percent,
            "disk_percent": disk.percent,
        }
    except Exception as e:
        checks["system_resources"] = {
            "status": "warning",
            "message": f"Failed to check resources: {e}",
        }

    # Overall readiness
    all_ready = all(
        check["status"] in ["ready", "warning"] for check in checks.values()
    )

    overall_status = (
        status.HTTP_200_OK if all_ready else status.HTTP_503_SERVICE_UNAVAILABLE
    )

    return JSONResponse(
        status_code=overall_status,
        content={
            "status": "ready" if all_ready else "not_ready",
            "checks": checks,
            "timestamp": time.time(),
        },
    )


@router.get("/health/detailed", status_code=status.HTTP_200_OK, tags=["Health"])
async def detailed_health() -> Dict[str, Any]:
    """
    Detailed health and system information.

    Returns:
        Comprehensive system status and metrics
    """
    # System info
    system_info = {
        "platform": os.uname().sysname,
        "platform_version": os.uname().release,
        "python_version": os.sys.version.split()[0],
        "process_id": os.getpid(),
    }

    # Resource usage
    try:
        process = psutil.Process(os.getpid())
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage("/")

        resources = {
            "cpu": {
                "system_percent": cpu_percent,
                "process_percent": process.cpu_percent(),
            },
            "memory": {
                "total_gb": round(memory.total / (1024**3), 2),
                "available_gb": round(memory.available / (1024**3), 2),
                "used_percent": memory.percent,
                "process_mb": round(process.memory_info().rss / (1024**2), 2),
            },
            "disk": {
                "total_gb": round(disk.total / (1024**3), 2),
                "free_gb": round(disk.free / (1024**3), 2),
                "used_percent": disk.percent,
            },
        }
    except Exception as e:
        resources = {"error": str(e)}

    # Configuration status
    config_status = {
        "environment": settings.ENV,
        "debug_mode": settings.DEBUG,
        "rate_limiting_enabled": settings.RATE_LIMIT_ENABLED,
        "quota_tracking_enabled": settings.QUOTA_ENABLED,
        "metrics_enabled": settings.METRICS_ENABLED,
        "openai_configured": settings.OPENAI_API_KEY is not None,
        "redis_configured": settings.REDIS_URL is not None,
    }

    # Feature flags
    features = {
        "demographics": settings.ENABLE_DEMOGRAPHICS,
        "multi_set_averaging": settings.ENABLE_MULTI_SET_AVERAGING,
        "bias_detection": settings.ENABLE_BIAS_DETECTION,
        "quality_metrics": settings.ENABLE_QUALITY_METRICS,
    }

    return {
        "status": "healthy",
        "timestamp": time.time(),
        "version": settings.APP_VERSION,
        "system": system_info,
        "resources": resources,
        "configuration": config_status,
        "features": features,
    }

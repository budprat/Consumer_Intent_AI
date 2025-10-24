# ABOUTME: Prometheus metrics endpoint for monitoring and observability
# ABOUTME: Exposes request counts, latencies, and custom business metrics

from fastapi import APIRouter, Response
from typing import Dict, Any
import time

from ..config import settings

router = APIRouter()

# In-memory metrics storage (production should use Prometheus client library)
_metrics = {
    "requests_total": 0,
    "requests_by_status": {},
    "requests_by_endpoint": {},
    "processing_time_seconds": [],
    "surveys_created": 0,
    "surveys_completed": 0,
    "surveys_failed": 0,
    "synthetic_responses_generated": 0,
}


@router.get("/metrics", tags=["Metrics"])
async def prometheus_metrics() -> Response:
    """
    Prometheus-compatible metrics endpoint.

    Returns:
        Metrics in Prometheus text format
    """
    if not settings.METRICS_ENABLED:
        return Response(
            content="# Metrics disabled\n",
            media_type="text/plain; version=0.0.4",
        )

    metrics_text = _generate_prometheus_format()

    return Response(
        content=metrics_text,
        media_type="text/plain; version=0.0.4",
    )


@router.get("/metrics/json", tags=["Metrics"])
async def json_metrics() -> Dict[str, Any]:
    """
    JSON-formatted metrics for easy consumption.

    Returns:
        Metrics as JSON dictionary
    """
    if not settings.METRICS_ENABLED:
        return {"enabled": False, "message": "Metrics disabled"}

    return {
        "enabled": True,
        "timestamp": time.time(),
        "metrics": _metrics.copy(),
        "system": {
            "environment": settings.ENV,
            "version": settings.APP_VERSION,
        },
    }


def _generate_prometheus_format() -> str:
    """
    Generate Prometheus text format from metrics.

    Returns:
        Metrics string in Prometheus exposition format
    """
    lines = []

    # Request totals
    lines.append("# HELP http_requests_total Total HTTP requests")
    lines.append("# TYPE http_requests_total counter")
    lines.append(f"http_requests_total {_metrics['requests_total']}")
    lines.append("")

    # Requests by status code
    lines.append("# HELP http_requests_by_status HTTP requests by status code")
    lines.append("# TYPE http_requests_by_status counter")
    for status_code, count in _metrics["requests_by_status"].items():
        lines.append(f'http_requests_by_status{{status_code="{status_code}"}} {count}')
    lines.append("")

    # Requests by endpoint
    lines.append("# HELP http_requests_by_endpoint HTTP requests by endpoint")
    lines.append("# TYPE http_requests_by_endpoint counter")
    for endpoint, count in _metrics["requests_by_endpoint"].items():
        lines.append(f'http_requests_by_endpoint{{endpoint="{endpoint}"}} {count}')
    lines.append("")

    # Processing time histogram (simplified)
    if _metrics["processing_time_seconds"]:
        avg_time = sum(_metrics["processing_time_seconds"]) / len(
            _metrics["processing_time_seconds"]
        )
        max_time = max(_metrics["processing_time_seconds"])
        min_time = min(_metrics["processing_time_seconds"])

        lines.append("# HELP http_request_duration_seconds HTTP request duration")
        lines.append("# TYPE http_request_duration_seconds summary")
        lines.append(
            f"http_request_duration_seconds{{quantile=\"0.5\"}} {avg_time:.6f}"
        )
        lines.append(
            f"http_request_duration_seconds{{quantile=\"0.9\"}} {max_time:.6f}"
        )
        lines.append(
            f"http_request_duration_seconds{{quantile=\"0.99\"}} {max_time:.6f}"
        )
        lines.append(
            f"http_request_duration_seconds_sum {sum(_metrics['processing_time_seconds']):.6f}"
        )
        lines.append(
            f"http_request_duration_seconds_count {len(_metrics['processing_time_seconds'])}"
        )
        lines.append("")

    # Business metrics - Surveys
    lines.append("# HELP surveys_created_total Total surveys created")
    lines.append("# TYPE surveys_created_total counter")
    lines.append(f"surveys_created_total {_metrics['surveys_created']}")
    lines.append("")

    lines.append("# HELP surveys_completed_total Total surveys completed")
    lines.append("# TYPE surveys_completed_total counter")
    lines.append(f"surveys_completed_total {_metrics['surveys_completed']}")
    lines.append("")

    lines.append("# HELP surveys_failed_total Total surveys failed")
    lines.append("# TYPE surveys_failed_total counter")
    lines.append(f"surveys_failed_total {_metrics['surveys_failed']}")
    lines.append("")

    # Business metrics - Synthetic responses
    lines.append(
        "# HELP synthetic_responses_generated_total Total synthetic responses generated"
    )
    lines.append("# TYPE synthetic_responses_generated_total counter")
    lines.append(
        f"synthetic_responses_generated_total {_metrics['synthetic_responses_generated']}"
    )
    lines.append("")

    return "\n".join(lines)


# Helper functions to record metrics (called from middleware and routes)


def record_request(endpoint: str, status_code: int, processing_time: float) -> None:
    """
    Record a completed request.

    Args:
        endpoint: Request endpoint path
        status_code: HTTP status code
        processing_time: Request processing time in seconds
    """
    _metrics["requests_total"] += 1

    # By status code
    status_key = str(status_code)
    _metrics["requests_by_status"][status_key] = (
        _metrics["requests_by_status"].get(status_key, 0) + 1
    )

    # By endpoint
    _metrics["requests_by_endpoint"][endpoint] = (
        _metrics["requests_by_endpoint"].get(endpoint, 0) + 1
    )

    # Processing time
    _metrics["processing_time_seconds"].append(processing_time)

    # Keep only last 1000 processing times
    if len(_metrics["processing_time_seconds"]) > 1000:
        _metrics["processing_time_seconds"] = _metrics["processing_time_seconds"][-1000:]


def record_survey_created() -> None:
    """Record a survey creation event."""
    _metrics["surveys_created"] += 1


def record_survey_completed() -> None:
    """Record a survey completion event."""
    _metrics["surveys_completed"] += 1


def record_survey_failed() -> None:
    """Record a survey failure event."""
    _metrics["surveys_failed"] += 1


def record_responses_generated(count: int) -> None:
    """
    Record synthetic response generation.

    Args:
        count: Number of responses generated
    """
    _metrics["synthetic_responses_generated"] += count

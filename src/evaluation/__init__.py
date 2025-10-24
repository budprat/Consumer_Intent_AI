# ABOUTME: Evaluation module for SSR system validation and benchmarking
# ABOUTME: Implements core metrics, test-retest reliability, and human data comparison

from .metrics import (
    MetricsCalculator,
    KSSimilarity,
    CorrelationMetrics,
    ErrorMetrics,
    MetricsReport,
)
from .reliability import (
    ReliabilitySimulator,
    ReliabilityResult,
    ReliabilityReport,
)
from .benchmarking import (
    HumanDataLoader,
    HumanSurveyResponse,
    HumanDataset,
    BenchmarkComparator,
    BenchmarkComparison,
    BenchmarkReport,
)

__all__ = [
    "MetricsCalculator",
    "KSSimilarity",
    "CorrelationMetrics",
    "ErrorMetrics",
    "MetricsReport",
    "ReliabilitySimulator",
    "ReliabilityResult",
    "ReliabilityReport",
    "HumanDataLoader",
    "HumanSurveyResponse",
    "HumanDataset",
    "BenchmarkComparator",
    "BenchmarkComparison",
    "BenchmarkReport",
]

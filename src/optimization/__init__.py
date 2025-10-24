# ABOUTME: Optimization module for SSR system performance enhancement
# ABOUTME: Implements advanced averaging strategies and reference statement optimization

from .averaging import (
    AveragingStrategy,
    AveragingConfig,
    AdvancedAverager,
    AveragingResult,
    PerformanceWeights,
)
from .quality_metrics import (
    StatementQuality,
    SetQuality,
    QualityReport,
    QualityAnalyzer,
)
from .custom_sets import (
    DomainCategory,
    ReferenceStatement,
    CustomReferenceSet,
    ValidationResult,
    CustomSetBuilder,
)

__all__ = [
    "AveragingStrategy",
    "AveragingConfig",
    "AdvancedAverager",
    "AveragingResult",
    "PerformanceWeights",
    "StatementQuality",
    "SetQuality",
    "QualityReport",
    "QualityAnalyzer",
    "DomainCategory",
    "ReferenceStatement",
    "CustomReferenceSet",
    "ValidationResult",
    "CustomSetBuilder",
]

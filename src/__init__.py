# ABOUTME: Package initialization for synthetic consumer SSR system
# ABOUTME: Provides top-level imports for main system components

__version__ = "0.1.0"
__author__ = "ClaudeOS Team"

from .core.ssr_engine import SSREngine
from .core.similarity import SimilarityCalculator
from .core.distribution import DistributionConstructor

__all__ = [
    "SSREngine",
    "SimilarityCalculator",
    "DistributionConstructor",
]

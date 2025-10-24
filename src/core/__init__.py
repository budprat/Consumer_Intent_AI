# ABOUTME: Core SSR engine module providing semantic similarity rating functionality
# ABOUTME: Contains text elicitation, embedding retrieval, similarity calculation, and distribution construction

from .ssr_engine import SSREngine
from .similarity import SimilarityCalculator
from .distribution import DistributionConstructor
from .embedding import EmbeddingRetriever
from .reference_statements import ReferenceStatementManager

__all__ = [
    "SSREngine",
    "SimilarityCalculator",
    "DistributionConstructor",
    "EmbeddingRetriever",
    "ReferenceStatementManager",
]

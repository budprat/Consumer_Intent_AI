"""
ABOUTME: Advanced multi-set averaging strategies for SSR distribution optimization
ABOUTME: Implements weighted, adaptive, and performance-based averaging methods

This module extends the basic multi-set averaging from Phase 1 with advanced
strategies that can improve distribution accuracy and reduce sensitivity to
specific reference statement phrasing.

Averaging Strategies:
1. UNIFORM: Equal weights for all sets (baseline from paper)
2. WEIGHTED: User-defined weights based on domain knowledge
3. ADAPTIVE: Automatic weight adjustment based on response characteristics
4. PERFORMANCE_BASED: Weights optimized from historical performance data
5. BEST_SUBSET: Select top-k performing sets dynamically

The paper uses uniform averaging across 6 sets for robustness. These advanced
strategies aim to improve upon that baseline while maintaining interpretability.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
import numpy as np
from scipy import stats
import warnings


class AveragingStrategy(Enum):
    """Enumeration of available averaging strategies"""

    UNIFORM = "uniform"  # Equal weights (paper baseline)
    WEIGHTED = "weighted"  # User-defined weights
    ADAPTIVE = "adaptive"  # Automatic weight adjustment
    PERFORMANCE_BASED = "performance"  # Optimized from historical data
    BEST_SUBSET = "best_subset"  # Select top-k sets only


@dataclass
class PerformanceWeights:
    """
    Performance-based weights for reference statement sets

    Attributes:
        set_ids: Identifier for each reference set
        weights: Weight values (normalized to sum to 1.0)
        performance_scores: Historical performance scores (0.0-1.0)
        sample_sizes: Number of samples used to calculate each score
        confidence_intervals: 95% confidence intervals for each weight
    """

    set_ids: List[str]
    weights: np.ndarray
    performance_scores: np.ndarray
    sample_sizes: np.ndarray
    confidence_intervals: Optional[np.ndarray] = None

    def __post_init__(self):
        """Validate and normalize weights"""
        # Ensure weights sum to 1.0
        weight_sum = np.sum(self.weights)
        if not np.isclose(weight_sum, 1.0):
            self.weights = self.weights / weight_sum

        # Calculate confidence intervals if not provided
        if self.confidence_intervals is None and len(self.performance_scores) > 0:
            # Use Wilson score interval for binomial proportions
            self.confidence_intervals = self._calculate_confidence_intervals()

    def _calculate_confidence_intervals(self, confidence: float = 0.95) -> np.ndarray:
        """
        Calculate confidence intervals for performance scores

        Args:
            confidence: Confidence level (default: 0.95 for 95% CI)

        Returns:
            Array of shape (n_sets, 2) with [lower, upper] bounds
        """
        z = stats.norm.ppf((1 + confidence) / 2)  # 1.96 for 95% CI
        intervals = []

        for score, n in zip(self.performance_scores, self.sample_sizes):
            if n > 0:
                # Wilson score interval
                center = (score + z**2 / (2 * n)) / (1 + z**2 / n)
                margin = (
                    z
                    * np.sqrt(score * (1 - score) / n + z**2 / (4 * n**2))
                    / (1 + z**2 / n)
                )
                lower = max(0.0, center - margin)
                upper = min(1.0, center + margin)
            else:
                lower = upper = score

            intervals.append([lower, upper])

        return np.array(intervals)


@dataclass
class AveragingConfig:
    """
    Configuration for advanced averaging

    Attributes:
        strategy: Averaging strategy to use
        weights: User-defined weights (for WEIGHTED strategy)
        k_best: Number of best sets to use (for BEST_SUBSET strategy)
        adaptive_threshold: Sensitivity threshold for adaptive weighting
        performance_weights: Pre-computed performance weights (for PERFORMANCE_BASED)
        min_weight: Minimum weight for any set (prevents zero weights)
        normalize: Whether to normalize weights to sum to 1.0
    """

    strategy: AveragingStrategy = AveragingStrategy.UNIFORM
    weights: Optional[np.ndarray] = None
    k_best: int = 4
    adaptive_threshold: float = 0.1
    performance_weights: Optional[PerformanceWeights] = None
    min_weight: float = 0.05
    normalize: bool = True


@dataclass
class AveragingResult:
    """
    Result of advanced averaging operation

    Attributes:
        averaged_distribution: Final averaged probability distribution
        individual_distributions: Distributions from each reference set
        weights_used: Actual weights applied to each set
        strategy: Strategy used for averaging
        set_ids: Identifiers for reference sets
        performance_metrics: Optional performance metrics for each set
        variance: Variance across individual distributions
        entropy: Entropy of averaged distribution
        confidence_score: Confidence in averaged result (0.0-1.0)
    """

    averaged_distribution: np.ndarray
    individual_distributions: List[np.ndarray]
    weights_used: np.ndarray
    strategy: AveragingStrategy
    set_ids: List[str]
    performance_metrics: Optional[Dict[str, float]] = None
    variance: float = 0.0
    entropy: float = 0.0
    confidence_score: float = 1.0

    def __post_init__(self):
        """Calculate derived metrics"""
        if len(self.individual_distributions) > 0:
            # Calculate variance across sets
            stacked = np.stack(self.individual_distributions)
            self.variance = np.mean(np.var(stacked, axis=0))

            # Calculate entropy of averaged distribution
            self.entropy = self._calculate_entropy(self.averaged_distribution)

            # Calculate confidence score based on agreement between sets
            self.confidence_score = self._calculate_confidence()

    def _calculate_entropy(self, distribution: np.ndarray) -> float:
        """Calculate Shannon entropy of distribution"""
        # Avoid log(0) by adding small epsilon
        epsilon = 1e-10
        safe_dist = distribution + epsilon
        safe_dist = safe_dist / safe_dist.sum()  # Renormalize
        return -np.sum(safe_dist * np.log2(safe_dist))

    def _calculate_confidence(self) -> float:
        """
        Calculate confidence score based on inter-set agreement

        Higher confidence when distributions are similar across sets.
        Lower confidence when distributions diverge significantly.
        """
        if len(self.individual_distributions) < 2:
            return 1.0

        # Calculate pairwise KS statistics
        ks_stats = []
        for i in range(len(self.individual_distributions)):
            for j in range(i + 1, len(self.individual_distributions)):
                # Convert to CDFs
                cdf_i = np.cumsum(self.individual_distributions[i])
                cdf_j = np.cumsum(self.individual_distributions[j])
                # KS statistic
                ks_stat = np.max(np.abs(cdf_i - cdf_j))
                ks_stats.append(ks_stat)

        # Average KS statistic (0 = perfect agreement, 1 = max divergence)
        avg_ks = np.mean(ks_stats)

        # Convert to confidence score (1 = high confidence, 0 = low confidence)
        confidence = 1.0 - avg_ks

        return confidence


class AdvancedAverager:
    """
    Advanced multi-set averaging engine

    Features:
    - Multiple averaging strategies
    - Automatic weight optimization
    - Performance-based weighting
    - Sensitivity analysis
    - Subset selection
    """

    def __init__(self, config: Optional[AveragingConfig] = None):
        """
        Initialize advanced averager

        Args:
            config: Averaging configuration (defaults to UNIFORM strategy)
        """
        self.config = config or AveragingConfig()

    def average(
        self,
        distributions: List[np.ndarray],
        set_ids: Optional[List[str]] = None,
        response_characteristics: Optional[Dict[str, Any]] = None,
    ) -> AveragingResult:
        """
        Average distributions using configured strategy

        Args:
            distributions: List of probability distributions to average
            set_ids: Optional identifiers for each distribution
            response_characteristics: Optional response features for adaptive weighting

        Returns:
            AveragingResult with averaged distribution and metadata
        """
        if len(distributions) == 0:
            raise ValueError("Cannot average empty list of distributions")

        # Default set IDs
        if set_ids is None:
            set_ids = [f"set_{i}" for i in range(len(distributions))]

        # Validate distributions
        for i, dist in enumerate(distributions):
            if not np.isclose(np.sum(dist), 1.0):
                raise ValueError(
                    f"Distribution {i} does not sum to 1.0: {np.sum(dist)}"
                )

        # Calculate weights based on strategy
        weights = self._calculate_weights(
            distributions, set_ids, response_characteristics
        )

        # Apply weights
        averaged = self._apply_weights(distributions, weights)

        # Create result
        result = AveragingResult(
            averaged_distribution=averaged,
            individual_distributions=distributions,
            weights_used=weights,
            strategy=self.config.strategy,
            set_ids=set_ids,
        )

        return result

    def _calculate_weights(
        self,
        distributions: List[np.ndarray],
        set_ids: List[str],
        response_characteristics: Optional[Dict[str, Any]],
    ) -> np.ndarray:
        """
        Calculate weights based on configured strategy

        Args:
            distributions: List of probability distributions
            set_ids: Identifiers for each distribution
            response_characteristics: Optional response features

        Returns:
            Normalized weight array
        """
        n_sets = len(distributions)

        if self.config.strategy == AveragingStrategy.UNIFORM:
            # Equal weights (paper baseline)
            weights = np.ones(n_sets) / n_sets

        elif self.config.strategy == AveragingStrategy.WEIGHTED:
            # User-defined weights
            if self.config.weights is None:
                raise ValueError("WEIGHTED strategy requires weights in config")
            if len(self.config.weights) != n_sets:
                raise ValueError(
                    f"Weights length {len(self.config.weights)} != {n_sets} sets"
                )
            weights = self.config.weights.copy()

        elif self.config.strategy == AveragingStrategy.ADAPTIVE:
            # Adaptive weighting based on response characteristics
            weights = self._adaptive_weights(
                distributions, set_ids, response_characteristics
            )

        elif self.config.strategy == AveragingStrategy.PERFORMANCE_BASED:
            # Performance-based weighting
            if self.config.performance_weights is None:
                raise ValueError(
                    "PERFORMANCE_BASED strategy requires performance_weights in config"
                )
            weights = self._performance_weights(set_ids)

        elif self.config.strategy == AveragingStrategy.BEST_SUBSET:
            # Select best k sets
            weights = self._best_subset_weights(distributions, set_ids)

        else:
            raise ValueError(f"Unknown strategy: {self.config.strategy}")

        # Apply minimum weight constraint
        weights = np.maximum(weights, self.config.min_weight)

        # Normalize if requested
        if self.config.normalize:
            weights = weights / np.sum(weights)

        return weights

    def _adaptive_weights(
        self,
        distributions: List[np.ndarray],
        set_ids: List[str],
        response_characteristics: Optional[Dict[str, Any]],
    ) -> np.ndarray:
        """
        Calculate adaptive weights based on distribution characteristics

        Strategy: Weight sets higher if they show:
        1. Lower variance (more confident)
        2. Higher discriminative power (not flat)
        3. Consistency with response sentiment (if available)

        Args:
            distributions: List of probability distributions
            set_ids: Identifiers for each distribution
            response_characteristics: Optional response features

        Returns:
            Adaptive weight array
        """
        n_sets = len(distributions)
        scores = np.zeros(n_sets)

        for i, dist in enumerate(distributions):
            # Score 1: Entropy (lower is more discriminative)
            # Normalized entropy: H / H_max where H_max = log2(n_classes)
            epsilon = 1e-10
            safe_dist = dist + epsilon
            safe_dist = safe_dist / safe_dist.sum()
            entropy = -np.sum(safe_dist * np.log2(safe_dist))
            max_entropy = np.log2(len(dist))
            normalized_entropy = entropy / max_entropy

            # Lower entropy = higher score
            entropy_score = 1.0 - normalized_entropy

            # Score 2: Peakedness (higher is more confident)
            # How much probability mass is in the mode
            peakedness_score = np.max(dist)

            # Score 3: Sentiment consistency (if available)
            sentiment_score = 0.5  # Neutral default
            if response_characteristics and "sentiment" in response_characteristics:
                # Expected: sentiment in [-1, 1] where -1=negative, 0=neutral, 1=positive
                sentiment = response_characteristics["sentiment"]

                # Calculate expected rating from distribution
                # Assume ratings are [1, 2, 3, 4, 5]
                ratings = np.arange(1, len(dist) + 1)
                expected_rating = np.sum(dist * ratings)

                # Map expected rating to [-1, 1] scale
                # 1-5 scale: 1=negative, 3=neutral, 5=positive
                expected_sentiment = (expected_rating - 3) / 2

                # Sentiment consistency: how well does distribution match sentiment?
                sentiment_consistency = 1.0 - abs(sentiment - expected_sentiment) / 2
                sentiment_score = sentiment_consistency

            # Combine scores (equal weighting)
            combined_score = (entropy_score + peakedness_score + sentiment_score) / 3
            scores[i] = combined_score

        # Convert scores to weights
        # Add threshold to avoid extreme weights
        scores = scores + self.config.adaptive_threshold

        # Normalize to weights
        weights = scores / np.sum(scores)

        return weights

    def _performance_weights(self, set_ids: List[str]) -> np.ndarray:
        """
        Get performance-based weights from historical data

        Args:
            set_ids: Identifiers for reference sets

        Returns:
            Performance-optimized weights
        """
        perf_weights = self.config.performance_weights

        # Map set IDs to performance weights
        weights = np.ones(len(set_ids))

        for i, set_id in enumerate(set_ids):
            if set_id in perf_weights.set_ids:
                idx = perf_weights.set_ids.index(set_id)
                weights[i] = perf_weights.weights[idx]
            else:
                # Unknown set gets uniform weight
                warnings.warn(
                    f"Set '{set_id}' not found in performance weights, using uniform"
                )

        return weights

    def _best_subset_weights(
        self, distributions: List[np.ndarray], set_ids: List[str]
    ) -> np.ndarray:
        """
        Select best k sets based on discriminative power

        Strategy: Choose sets with highest discriminative power (lowest entropy)
        and apply uniform weights to selected sets, zero to others.

        Args:
            distributions: List of probability distributions
            set_ids: Identifiers for each distribution

        Returns:
            Binary weights (1 for selected, 0 for not selected, normalized)
        """
        n_sets = len(distributions)
        k = min(self.config.k_best, n_sets)

        # Calculate discriminative power (inverse of entropy)
        discriminative_scores = []
        for dist in distributions:
            epsilon = 1e-10
            safe_dist = dist + epsilon
            safe_dist = safe_dist / safe_dist.sum()
            entropy = -np.sum(safe_dist * np.log2(safe_dist))
            # Lower entropy = higher discriminative power
            discriminative_scores.append(-entropy)

        # Select top k
        top_k_indices = np.argsort(discriminative_scores)[-k:]

        # Create binary weights
        weights = np.zeros(n_sets)
        weights[top_k_indices] = 1.0 / k

        return weights

    def _apply_weights(
        self, distributions: List[np.ndarray], weights: np.ndarray
    ) -> np.ndarray:
        """
        Apply weights to distributions and compute weighted average

        Args:
            distributions: List of probability distributions
            weights: Weight for each distribution

        Returns:
            Weighted average distribution
        """
        # Stack distributions
        stacked = np.stack(distributions)

        # Weight and sum
        weighted = stacked * weights[:, np.newaxis]
        averaged = np.sum(weighted, axis=0)

        # Ensure normalization (should already be normalized, but verify)
        averaged = averaged / np.sum(averaged)

        return averaged

    def analyze_sensitivity(
        self,
        distributions: List[np.ndarray],
        set_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Analyze sensitivity to different set combinations

        Tests all possible subsets of size k and reports variance in results.

        Args:
            distributions: List of probability distributions
            set_ids: Optional identifiers for each distribution

        Returns:
            Dictionary with sensitivity analysis results
        """
        from itertools import combinations

        if set_ids is None:
            set_ids = [f"set_{i}" for i in range(len(distributions))]

        n_sets = len(distributions)
        k = min(self.config.k_best, n_sets)

        # Test all k-combinations
        all_averages = []
        all_combinations = []

        for combo_indices in combinations(range(n_sets), k):
            # Create subset
            subset_dists = [distributions[i] for i in combo_indices]
            subset_ids = [set_ids[i] for i in combo_indices]

            # Average with uniform weights
            temp_config = AveragingConfig(strategy=AveragingStrategy.UNIFORM)
            temp_averager = AdvancedAverager(config=temp_config)
            result = temp_averager.average(subset_dists, subset_ids)

            all_averages.append(result.averaged_distribution)
            all_combinations.append(subset_ids)

        # Calculate statistics
        all_averages_arr = np.stack(all_averages)

        mean_distribution = np.mean(all_averages_arr, axis=0)
        std_distribution = np.std(all_averages_arr, axis=0)
        max_variance = np.max(np.var(all_averages_arr, axis=0))

        # Find most stable and most variable combinations
        variances = [np.var(avg) for avg in all_averages]
        most_stable_idx = np.argmin(variances)
        most_variable_idx = np.argmax(variances)

        return {
            "n_combinations_tested": len(all_averages),
            "subset_size": k,
            "mean_distribution": mean_distribution,
            "std_distribution": std_distribution,
            "max_variance": max_variance,
            "most_stable_combination": all_combinations[most_stable_idx],
            "most_variable_combination": all_combinations[most_variable_idx],
            "stability_score": 1.0 - max_variance,  # 1 = stable, 0 = highly variable
        }


# Example usage and testing
if __name__ == "__main__":
    print("Advanced Averaging Testing")
    print("=" * 70)

    # Test 1: Uniform averaging (baseline)
    print("\n1. Uniform Averaging (Paper Baseline):")

    np.random.seed(42)

    # Simulate 6 reference sets with similar but not identical distributions
    base_dist = np.array([0.1, 0.2, 0.4, 0.2, 0.1])
    distributions = []
    for i in range(6):
        # Add small noise
        noise = np.random.normal(0, 0.02, 5)
        dist = base_dist + noise
        dist = np.maximum(dist, 0)  # No negative probabilities
        dist = dist / dist.sum()  # Renormalize
        distributions.append(dist)

    config = AveragingConfig(strategy=AveragingStrategy.UNIFORM)
    averager = AdvancedAverager(config)
    result = averager.average(distributions)

    print(f"Strategy: {result.strategy.value}")
    print(f"Weights used: {result.weights_used}")
    print(f"Averaged distribution: {result.averaged_distribution}")
    print(f"Variance across sets: {result.variance:.4f}")
    print(f"Entropy: {result.entropy:.3f}")
    print(f"Confidence score: {result.confidence_score:.3f}")

    # Test 2: Adaptive averaging
    print("\n" + "=" * 70)
    print("2. Adaptive Averaging:")

    # Create distributions with different characteristics
    # Set 0: Very confident (peaked)
    distributions[0] = np.array([0.05, 0.05, 0.8, 0.05, 0.05])
    # Set 1: Uncertain (flat)
    distributions[1] = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    # Others: moderate
    for i in range(2, 6):
        distributions[i] = base_dist + np.random.normal(0, 0.05, 5)
        distributions[i] = np.maximum(distributions[i], 0)
        distributions[i] = distributions[i] / distributions[i].sum()

    config = AveragingConfig(
        strategy=AveragingStrategy.ADAPTIVE, adaptive_threshold=0.1
    )
    averager = AdvancedAverager(config)
    result = averager.average(
        distributions, response_characteristics={"sentiment": 0.5}
    )

    print(f"Strategy: {result.strategy.value}")
    print(f"Weights used: {result.weights_used}")
    print(f"Averaged distribution: {result.averaged_distribution}")
    print(f"Confidence score: {result.confidence_score:.3f}")
    print("\nNote: Set 0 (peaked) should have higher weight than Set 1 (flat)")

    # Test 3: Best subset selection
    print("\n" + "=" * 70)
    print("3. Best Subset Selection (k=4):")

    config = AveragingConfig(strategy=AveragingStrategy.BEST_SUBSET, k_best=4)
    averager = AdvancedAverager(config)
    result = averager.average(distributions)

    print(f"Strategy: {result.strategy.value}")
    print(f"Weights used: {result.weights_used}")
    print(f"Selected sets: {[i for i, w in enumerate(result.weights_used) if w > 0]}")
    print(f"Averaged distribution: {result.averaged_distribution}")

    # Test 4: Performance-based weighting
    print("\n" + "=" * 70)
    print("4. Performance-Based Weighting:")

    # Simulate historical performance data
    perf_weights = PerformanceWeights(
        set_ids=[f"set_{i}" for i in range(6)],
        weights=np.array([0.12, 0.08, 0.22, 0.18, 0.25, 0.15]),  # Optimized weights
        performance_scores=np.array([0.85, 0.78, 0.92, 0.88, 0.95, 0.82]),
        sample_sizes=np.array([100, 100, 100, 100, 100, 100]),
    )

    config = AveragingConfig(
        strategy=AveragingStrategy.PERFORMANCE_BASED, performance_weights=perf_weights
    )
    averager = AdvancedAverager(config)
    result = averager.average(distributions, set_ids=[f"set_{i}" for i in range(6)])

    print(f"Strategy: {result.strategy.value}")
    print(f"Performance scores: {perf_weights.performance_scores}")
    print(f"Optimized weights: {result.weights_used}")
    print(f"Averaged distribution: {result.averaged_distribution}")

    # Test 5: Sensitivity analysis
    print("\n" + "=" * 70)
    print("5. Sensitivity Analysis:")

    config = AveragingConfig(k_best=4)
    averager = AdvancedAverager(config)
    sensitivity = averager.analyze_sensitivity(
        distributions, set_ids=[f"set_{i}" for i in range(6)]
    )

    print(f"Combinations tested: {sensitivity['n_combinations_tested']}")
    print(f"Subset size: {sensitivity['subset_size']}")
    print(f"Max variance: {sensitivity['max_variance']:.4f}")
    print(f"Stability score: {sensitivity['stability_score']:.3f}")
    print(f"Most stable combination: {sensitivity['most_stable_combination']}")
    print(f"Most variable combination: {sensitivity['most_variable_combination']}")

    print("\n" + "=" * 70)
    print("Advanced averaging testing complete")
    print("\nKey Insights:")
    print("- UNIFORM: Simple baseline, equal treatment of all sets")
    print("- ADAPTIVE: Weights confident/discriminative sets higher")
    print("- BEST_SUBSET: Selects top performers, ignores poor sets")
    print("- PERFORMANCE: Uses historical data for optimal weighting")
    print("- Sensitivity analysis identifies robust set combinations")

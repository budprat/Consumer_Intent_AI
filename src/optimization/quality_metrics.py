"""
ABOUTME: Reference statement quality metrics and discriminative power analysis
ABOUTME: Evaluates effectiveness of individual reference statements and sets

This module implements quality assessment for reference statements used in the SSR system.
Good reference statements should:
1. Have high discriminative power (not produce flat distributions)
2. Be consistent with other statements in their set
3. Correlate well with human judgments
4. Generalize across different product categories

Quality Metrics:
- Discriminative Power: How well a statement differentiates between ratings
- Inter-Set Consistency: Agreement between statements within a set
- Statement Effectiveness: Correlation with ground truth ratings
- Redundancy Analysis: Identification of similar/redundant statements
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np
from scipy import stats
from scipy.spatial.distance import cosine as cosine_distance
import warnings


@dataclass
class StatementQuality:
    """
    Quality metrics for a single reference statement

    Attributes:
        statement_id: Identifier for the statement
        statement_text: The actual text of the statement
        discriminative_power: How well it differentiates ratings (0.0-1.0)
        entropy_score: Distribution entropy (lower = more discriminative)
        peakedness: Maximum probability mass (higher = more confident)
        consistency_score: Agreement with other statements in set (0.0-1.0)
        correlation_with_truth: Correlation with ground truth (if available)
        sample_size: Number of responses analyzed
        mean_distribution: Average distribution produced by this statement
        std_distribution: Standard deviation across responses
    """

    statement_id: str
    statement_text: str
    discriminative_power: float
    entropy_score: float
    peakedness: float
    consistency_score: float
    correlation_with_truth: Optional[float] = None
    sample_size: int = 0
    mean_distribution: Optional[np.ndarray] = None
    std_distribution: Optional[np.ndarray] = None

    def get_overall_quality(self) -> float:
        """
        Calculate overall quality score (0.0-1.0)

        Combines multiple metrics into single score.
        Higher is better.
        """
        # Weight components
        weights = {
            "discriminative_power": 0.4,
            "consistency": 0.3,
            "peakedness": 0.2,
            "correlation": 0.1,
        }

        score = (
            weights["discriminative_power"] * self.discriminative_power
            + weights["consistency"] * self.consistency_score
            + weights["peakedness"] * self.peakedness
        )

        # Add correlation if available
        if self.correlation_with_truth is not None:
            # Normalize correlation from [-1, 1] to [0, 1]
            normalized_corr = (self.correlation_with_truth + 1) / 2
            score += weights["correlation"] * normalized_corr
        else:
            # Redistribute correlation weight
            remaining_weight = 1.0 - weights["correlation"]
            score = score / remaining_weight

        return score


@dataclass
class SetQuality:
    """
    Quality metrics for a reference statement set

    Attributes:
        set_id: Identifier for the set
        statements: List of StatementQuality objects
        mean_discriminative_power: Average discriminative power across statements
        inter_statement_consistency: How similar statements are within set
        set_diversity: How diverse the statements are (complement of consistency)
        redundancy_pairs: Pairs of statements that may be redundant
        overall_quality: Combined quality score for the set
    """

    set_id: str
    statements: List[StatementQuality]
    mean_discriminative_power: float
    inter_statement_consistency: float
    set_diversity: float
    redundancy_pairs: List[Tuple[str, str, float]] = field(default_factory=list)
    overall_quality: float = 0.0

    def __post_init__(self):
        """Calculate derived metrics"""
        if len(self.statements) > 0:
            self.overall_quality = np.mean(
                [s.get_overall_quality() for s in self.statements]
            )


@dataclass
class QualityReport:
    """
    Comprehensive quality report for all reference statements

    Attributes:
        set_qualities: Quality metrics for each set
        statement_rankings: Statements ranked by overall quality
        recommended_weights: Recommended weights based on quality
        improvement_suggestions: Actionable suggestions for improvement
    """

    set_qualities: List[SetQuality]
    statement_rankings: List[Tuple[str, float]]  # (statement_id, quality_score)
    recommended_weights: Dict[str, float]
    improvement_suggestions: List[str] = field(default_factory=list)

    def generate_summary(self) -> str:
        """Generate human-readable quality report"""
        lines = [
            "Reference Statement Quality Report",
            "=" * 70,
        ]

        lines.append(f"\n📊 Sets Analyzed: {len(self.set_qualities)}")

        for set_quality in self.set_qualities:
            lines.append(f"\n{set_quality.set_id}:")
            lines.append(f"  Statements: {len(set_quality.statements)}")
            lines.append(
                f"  Mean Discriminative Power: {set_quality.mean_discriminative_power:.3f}"
            )
            lines.append(
                f"  Inter-Statement Consistency: {set_quality.inter_statement_consistency:.3f}"
            )
            lines.append(f"  Set Diversity: {set_quality.set_diversity:.3f}")
            lines.append(f"  Overall Quality: {set_quality.overall_quality:.3f}")

            if set_quality.redundancy_pairs:
                lines.append(
                    f"  ⚠️  Redundancy detected: {len(set_quality.redundancy_pairs)} pairs"
                )

        lines.append("\n🏆 Top 5 Statements by Quality:")
        for i, (stmt_id, quality) in enumerate(self.statement_rankings[:5], 1):
            lines.append(f"  {i}. {stmt_id}: {quality:.3f}")

        if self.improvement_suggestions:
            lines.append("\n💡 Improvement Suggestions:")
            for suggestion in self.improvement_suggestions:
                lines.append(f"  • {suggestion}")

        return "\n".join(lines)


class QualityAnalyzer:
    """
    Analyzer for reference statement quality and effectiveness

    Features:
    - Discriminative power measurement
    - Inter-statement consistency analysis
    - Redundancy detection
    - Quality-based weight recommendations
    - Improvement suggestions
    """

    def __init__(
        self,
        redundancy_threshold: float = 0.9,
        min_discriminative_power: float = 0.5,
    ):
        """
        Initialize quality analyzer

        Args:
            redundancy_threshold: Cosine similarity threshold for redundancy (default: 0.9)
            min_discriminative_power: Minimum acceptable discriminative power (default: 0.5)
        """
        self.redundancy_threshold = redundancy_threshold
        self.min_discriminative_power = min_discriminative_power

    def analyze_statement(
        self,
        statement_id: str,
        statement_text: str,
        distributions: List[np.ndarray],
        ground_truth_ratings: Optional[List[float]] = None,
    ) -> StatementQuality:
        """
        Analyze quality of a single reference statement

        Args:
            statement_id: Identifier for the statement
            statement_text: Text of the statement
            distributions: List of distributions produced by this statement
            ground_truth_ratings: Optional ground truth ratings for correlation

        Returns:
            StatementQuality with computed metrics
        """
        if len(distributions) == 0:
            raise ValueError("Cannot analyze statement with no distributions")

        # Calculate discriminative power
        discriminative_power = self._calculate_discriminative_power(distributions)

        # Calculate entropy (average across all distributions)
        entropies = [self._calculate_entropy(dist) for dist in distributions]
        mean_entropy = np.mean(entropies)

        # Calculate peakedness (average max probability)
        peakedness_values = [np.max(dist) for dist in distributions]
        mean_peakedness = np.mean(peakedness_values)

        # Calculate consistency (how similar are distributions to each other?)
        consistency_score = self._calculate_consistency(distributions)

        # Calculate correlation with ground truth (if available)
        correlation = None
        if ground_truth_ratings is not None and len(ground_truth_ratings) == len(
            distributions
        ):
            correlation = self._calculate_correlation_with_truth(
                distributions, ground_truth_ratings
            )

        # Calculate mean and std distributions
        mean_dist = np.mean(distributions, axis=0)
        std_dist = np.std(distributions, axis=0)

        return StatementQuality(
            statement_id=statement_id,
            statement_text=statement_text,
            discriminative_power=discriminative_power,
            entropy_score=mean_entropy,
            peakedness=mean_peakedness,
            consistency_score=consistency_score,
            correlation_with_truth=correlation,
            sample_size=len(distributions),
            mean_distribution=mean_dist,
            std_distribution=std_dist,
        )

    def analyze_set(
        self,
        set_id: str,
        statement_qualities: List[StatementQuality],
        statement_embeddings: Optional[List[np.ndarray]] = None,
    ) -> SetQuality:
        """
        Analyze quality of a reference statement set

        Args:
            set_id: Identifier for the set
            statement_qualities: Quality metrics for each statement in set
            statement_embeddings: Optional embeddings for redundancy analysis

        Returns:
            SetQuality with set-level metrics
        """
        if len(statement_qualities) == 0:
            raise ValueError("Cannot analyze empty statement set")

        # Mean discriminative power
        mean_disc_power = np.mean(
            [sq.discriminative_power for sq in statement_qualities]
        )

        # Inter-statement consistency (average consistency across all statements)
        mean_consistency = np.mean([sq.consistency_score for sq in statement_qualities])

        # Set diversity (complement of consistency)
        set_diversity = 1.0 - mean_consistency

        # Detect redundancy using embeddings
        redundancy_pairs = []
        if statement_embeddings is not None and len(statement_embeddings) == len(
            statement_qualities
        ):
            redundancy_pairs = self._detect_redundancy(
                statement_qualities, statement_embeddings
            )

        return SetQuality(
            set_id=set_id,
            statements=statement_qualities,
            mean_discriminative_power=mean_disc_power,
            inter_statement_consistency=mean_consistency,
            set_diversity=set_diversity,
            redundancy_pairs=redundancy_pairs,
        )

    def generate_report(
        self,
        set_qualities: List[SetQuality],
        generate_suggestions: bool = True,
    ) -> QualityReport:
        """
        Generate comprehensive quality report

        Args:
            set_qualities: Quality metrics for all sets
            generate_suggestions: Whether to generate improvement suggestions

        Returns:
            QualityReport with recommendations
        """
        # Rank all statements by quality
        all_statements = []
        for set_quality in set_qualities:
            for stmt in set_quality.statements:
                all_statements.append((stmt.statement_id, stmt.get_overall_quality()))

        # Sort by quality (descending)
        statement_rankings = sorted(all_statements, key=lambda x: x[1], reverse=True)

        # Generate quality-based weights for sets
        recommended_weights = self._calculate_recommended_weights(set_qualities)

        # Generate improvement suggestions
        suggestions = []
        if generate_suggestions:
            suggestions = self._generate_suggestions(set_qualities)

        return QualityReport(
            set_qualities=set_qualities,
            statement_rankings=statement_rankings,
            recommended_weights=recommended_weights,
            improvement_suggestions=suggestions,
        )

    def _calculate_discriminative_power(self, distributions: List[np.ndarray]) -> float:
        """
        Calculate discriminative power of a statement

        High discriminative power = produces peaked distributions (not flat)

        Args:
            distributions: List of probability distributions

        Returns:
            Discriminative power score (0.0-1.0)
        """
        # Average entropy across all distributions
        entropies = [self._calculate_entropy(dist) for dist in distributions]
        mean_entropy = np.mean(entropies)

        # Normalize by max possible entropy
        max_entropy = np.log2(len(distributions[0]))  # Uniform distribution
        normalized_entropy = mean_entropy / max_entropy

        # Discriminative power is inverse of normalized entropy
        discriminative_power = 1.0 - normalized_entropy

        return discriminative_power

    def _calculate_entropy(self, distribution: np.ndarray) -> float:
        """Calculate Shannon entropy of distribution"""
        epsilon = 1e-10
        safe_dist = distribution + epsilon
        safe_dist = safe_dist / safe_dist.sum()
        return -np.sum(safe_dist * np.log2(safe_dist))

    def _calculate_consistency(self, distributions: List[np.ndarray]) -> float:
        """
        Calculate consistency across distributions

        High consistency = distributions are similar to each other

        Args:
            distributions: List of probability distributions

        Returns:
            Consistency score (0.0-1.0)
        """
        if len(distributions) < 2:
            return 1.0

        # Calculate pairwise KS similarities
        similarities = []
        for i in range(len(distributions)):
            for j in range(i + 1, len(distributions)):
                # KS similarity
                cdf_i = np.cumsum(distributions[i])
                cdf_j = np.cumsum(distributions[j])
                ks_stat = np.max(np.abs(cdf_i - cdf_j))
                ks_similarity = 1.0 - ks_stat
                similarities.append(ks_similarity)

        # Average similarity
        mean_similarity = np.mean(similarities)

        return mean_similarity

    def _calculate_correlation_with_truth(
        self, distributions: List[np.ndarray], ground_truth: List[float]
    ) -> float:
        """
        Calculate correlation between predicted ratings and ground truth

        Args:
            distributions: List of probability distributions
            ground_truth: True ratings

        Returns:
            Pearson correlation coefficient
        """
        # Convert distributions to expected ratings
        # Assume ratings are [1, 2, 3, 4, 5]
        ratings = np.arange(1, len(distributions[0]) + 1)
        predicted = [np.sum(dist * ratings) for dist in distributions]

        # Calculate correlation
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            correlation, _ = stats.pearsonr(predicted, ground_truth)

        if np.isnan(correlation):
            correlation = 0.0

        return correlation

    def _detect_redundancy(
        self,
        statement_qualities: List[StatementQuality],
        statement_embeddings: List[np.ndarray],
    ) -> List[Tuple[str, str, float]]:
        """
        Detect redundant statement pairs using embedding similarity

        Args:
            statement_qualities: Quality metrics for statements
            statement_embeddings: Embeddings for each statement

        Returns:
            List of (statement_id_1, statement_id_2, similarity) tuples
        """
        redundancy_pairs = []

        for i in range(len(statement_qualities)):
            for j in range(i + 1, len(statement_qualities)):
                # Cosine similarity
                similarity = 1.0 - cosine_distance(
                    statement_embeddings[i], statement_embeddings[j]
                )

                if similarity >= self.redundancy_threshold:
                    redundancy_pairs.append(
                        (
                            statement_qualities[i].statement_id,
                            statement_qualities[j].statement_id,
                            similarity,
                        )
                    )

        return redundancy_pairs

    def _calculate_recommended_weights(
        self, set_qualities: List[SetQuality]
    ) -> Dict[str, float]:
        """
        Calculate recommended weights based on set quality

        Args:
            set_qualities: Quality metrics for all sets

        Returns:
            Dictionary mapping set_id to recommended weight
        """
        # Use overall quality scores as basis for weights
        quality_scores = np.array([sq.overall_quality for sq in set_qualities])

        # Normalize to weights
        weights = quality_scores / np.sum(quality_scores)

        # Create mapping
        recommended = {}
        for i, sq in enumerate(set_qualities):
            recommended[sq.set_id] = weights[i]

        return recommended

    def _generate_suggestions(self, set_qualities: List[SetQuality]) -> List[str]:
        """
        Generate actionable improvement suggestions

        Args:
            set_qualities: Quality metrics for all sets

        Returns:
            List of suggestion strings
        """
        suggestions = []

        for sq in set_qualities:
            # Check for low discriminative power
            if sq.mean_discriminative_power < self.min_discriminative_power:
                suggestions.append(
                    f"{sq.set_id}: Low discriminative power ({sq.mean_discriminative_power:.2f}). "
                    f"Consider revising statements to be more specific and opinionated."
                )

            # Check for redundancy
            if sq.redundancy_pairs:
                suggestions.append(
                    f"{sq.set_id}: {len(sq.redundancy_pairs)} redundant statement pairs detected. "
                    f"Consider removing or revising similar statements."
                )

            # Check for low diversity
            if sq.set_diversity < 0.2:
                suggestions.append(
                    f"{sq.set_id}: Low diversity ({sq.set_diversity:.2f}). "
                    f"Statements are too similar. Add more varied perspectives."
                )

            # Check individual statement quality
            low_quality_statements = [
                s for s in sq.statements if s.get_overall_quality() < 0.5
            ]
            if low_quality_statements:
                suggestions.append(
                    f"{sq.set_id}: {len(low_quality_statements)} low-quality statements. "
                    f"Review: {', '.join([s.statement_id for s in low_quality_statements[:3]])}"
                )

        return suggestions


# Example usage and testing
if __name__ == "__main__":
    print("Reference Statement Quality Analysis Testing")
    print("=" * 70)

    np.random.seed(42)

    # Test 1: Analyze individual statement
    print("\n1. Individual Statement Analysis:")

    # Simulate distributions from a good statement (peaked, consistent)
    good_statement_dists = []
    for _ in range(50):
        # Base: mostly 4s and 5s (positive sentiment)
        base = np.array([0.05, 0.1, 0.15, 0.4, 0.3])
        noise = np.random.normal(0, 0.05, 5)
        dist = base + noise
        dist = np.maximum(dist, 0)
        dist = dist / dist.sum()
        good_statement_dists.append(dist)

    # Simulate ground truth ratings
    ground_truth = [4.0 + np.random.normal(0, 0.5) for _ in range(50)]
    ground_truth = [np.clip(g, 1, 5) for g in ground_truth]

    analyzer = QualityAnalyzer()
    good_quality = analyzer.analyze_statement(
        statement_id="stmt_good",
        statement_text="I would definitely purchase this product",
        distributions=good_statement_dists,
        ground_truth_ratings=ground_truth,
    )

    print(f"Statement: {good_quality.statement_text}")
    print(f"Discriminative Power: {good_quality.discriminative_power:.3f}")
    print(f"Entropy Score: {good_quality.entropy_score:.3f}")
    print(f"Peakedness: {good_quality.peakedness:.3f}")
    print(f"Consistency Score: {good_quality.consistency_score:.3f}")
    if good_quality.correlation_with_truth:
        print(f"Correlation with Truth: {good_quality.correlation_with_truth:.3f}")
    print(f"Overall Quality: {good_quality.get_overall_quality():.3f}")

    # Test 2: Compare good vs poor statement
    print("\n" + "=" * 70)
    print("2. Good vs Poor Statement Comparison:")

    # Simulate distributions from a poor statement (flat, inconsistent)
    poor_statement_dists = []
    for _ in range(50):
        # Mostly flat distribution
        base = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        noise = np.random.normal(0, 0.1, 5)  # Higher noise
        dist = base + noise
        dist = np.maximum(dist, 0)
        dist = dist / dist.sum()
        poor_statement_dists.append(dist)

    poor_quality = analyzer.analyze_statement(
        statement_id="stmt_poor",
        statement_text="This product exists",
        distributions=poor_statement_dists,
        ground_truth_ratings=ground_truth,
    )

    print("\nGood Statement:")
    print(f"  Discriminative Power: {good_quality.discriminative_power:.3f}")
    print(f"  Overall Quality: {good_quality.get_overall_quality():.3f}")

    print("\nPoor Statement:")
    print(f"  Discriminative Power: {poor_quality.discriminative_power:.3f}")
    print(f"  Overall Quality: {poor_quality.get_overall_quality():.3f}")

    # Test 3: Set analysis
    print("\n" + "=" * 70)
    print("3. Reference Set Analysis:")

    # Create a set with mixed quality statements
    statement_qualities = [good_quality, poor_quality]

    # Add a few more moderate quality statements
    for i in range(3):
        moderate_dists = []
        for _ in range(50):
            base = np.array([0.1, 0.15, 0.3, 0.3, 0.15])
            noise = np.random.normal(0, 0.07, 5)
            dist = base + noise
            dist = np.maximum(dist, 0)
            dist = dist / dist.sum()
            moderate_dists.append(dist)

        moderate_quality = analyzer.analyze_statement(
            statement_id=f"stmt_moderate_{i}",
            statement_text=f"Moderate statement {i}",
            distributions=moderate_dists,
        )
        statement_qualities.append(moderate_quality)

    set_quality = analyzer.analyze_set(
        set_id="test_set", statement_qualities=statement_qualities
    )

    print(f"Set ID: {set_quality.set_id}")
    print(f"Number of Statements: {len(set_quality.statements)}")
    print(f"Mean Discriminative Power: {set_quality.mean_discriminative_power:.3f}")
    print(f"Inter-Statement Consistency: {set_quality.inter_statement_consistency:.3f}")
    print(f"Set Diversity: {set_quality.set_diversity:.3f}")
    print(f"Overall Quality: {set_quality.overall_quality:.3f}")

    # Test 4: Generate comprehensive report
    print("\n" + "=" * 70)
    print("4. Comprehensive Quality Report:")

    report = analyzer.generate_report([set_quality])
    print("\n" + report.generate_summary())

    print("\n" + "=" * 70)
    print("Quality analysis testing complete")
    print("\nKey Insights:")
    print("- High discriminative power = peaked distributions (not flat)")
    print("- Consistency = how similar distributions are across responses")
    print("- Quality score combines multiple metrics for holistic assessment")
    print("- Recommendations help identify improvement opportunities")

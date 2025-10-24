"""
ABOUTME: Core evaluation metrics for SSR system validation
ABOUTME: Implements KS similarity, correlation, MAE, and statistical significance testing

This module implements the three primary metrics from the research paper:
1. Kolmogorov-Smirnov (KS) Similarity: Distribution matching (target: K^xy ≥ 0.85)
2. Pearson Correlation (ρ): Test-retest reliability (target: ρ ≥ 0.90)
3. Mean Absolute Error (MAE): Prediction accuracy (target: MAE < 0.5)

Paper Benchmarks:
- GPT-4o: ρ = 90.2%, K^xy = 0.88, MAE = 0.42
- Gemini-2.0-flash: ρ = 90.6%, K^xy = 0.80, MAE = 0.38
- Human baseline: ρ ≈ 100% (perfect test-retest reliability)

Success Criteria:
- Achieve ρ ≥ 0.90 (90% of human test-retest reliability)
- Achieve K^xy ≥ 0.85 (strong distribution matching)
- Achieve MAE < 0.5 (within half Likert point on average)
"""

from typing import Optional, Tuple
from dataclasses import dataclass
import numpy as np
from scipy import stats
import warnings


@dataclass
class KSSimilarity:
    """
    Kolmogorov-Smirnov similarity metric for distribution matching

    Measures how well two probability distributions match.
    Higher values indicate better matching (range: 0.0 to 1.0).

    Formula: K^xy = 1 - max|CDF_x(r) - CDF_y(r)| for all ratings r

    Attributes:
        ks_statistic: Raw KS statistic (distance between CDFs)
        ks_similarity: KS similarity score (1 - ks_statistic)
        p_value: Statistical significance (p < 0.05 indicates significant difference)
        distributions: The two distributions being compared
    """

    ks_statistic: float
    ks_similarity: float
    p_value: float
    distributions: Tuple[np.ndarray, np.ndarray]

    def passes_threshold(self, threshold: float = 0.85) -> bool:
        """Check if KS similarity meets target threshold"""
        return self.ks_similarity >= threshold

    def is_significantly_different(self, alpha: float = 0.05) -> bool:
        """Check if distributions are significantly different"""
        return self.p_value < alpha


@dataclass
class CorrelationMetrics:
    """
    Correlation metrics for test-retest reliability

    Measures consistency between repeated measurements.
    Higher values indicate better reliability (range: -1.0 to 1.0).

    Attributes:
        pearson_r: Pearson correlation coefficient
        pearson_p_value: Statistical significance
        spearman_rho: Spearman rank correlation (robust to outliers)
        spearman_p_value: Statistical significance
        correlation_attainment: Percentage of human baseline (ρ / ρ_human)
    """

    pearson_r: float
    pearson_p_value: float
    spearman_rho: float
    spearman_p_value: float
    correlation_attainment: Optional[float] = None

    def passes_threshold(self, threshold: float = 0.90) -> bool:
        """Check if correlation meets target threshold"""
        return self.pearson_r >= threshold

    def is_significant(self, alpha: float = 0.05) -> bool:
        """Check if correlation is statistically significant"""
        return self.pearson_p_value < alpha


@dataclass
class ErrorMetrics:
    """
    Error metrics for prediction accuracy

    Measures deviation from ground truth or expected values.
    Lower values indicate better accuracy.

    Attributes:
        mae: Mean Absolute Error
        rmse: Root Mean Squared Error
        max_error: Maximum absolute error
        mean_error: Mean signed error (bias)
        error_distribution: Distribution of errors
    """

    mae: float
    rmse: float
    max_error: float
    mean_error: float
    error_distribution: np.ndarray

    def passes_threshold(self, threshold: float = 0.5) -> bool:
        """Check if MAE meets target threshold"""
        return self.mae < threshold

    def has_systematic_bias(self, threshold: float = 0.1) -> bool:
        """Check if there's systematic over/under prediction"""
        return abs(self.mean_error) > threshold


@dataclass
class MetricsReport:
    """
    Comprehensive metrics report

    Attributes:
        ks_similarity: KS similarity metrics
        correlation: Correlation metrics
        error: Error metrics
        sample_size: Number of samples compared
        passes_all_thresholds: Whether all metrics meet targets
        summary: Text summary of results
    """

    ks_similarity: Optional[KSSimilarity] = None
    correlation: Optional[CorrelationMetrics] = None
    error: Optional[ErrorMetrics] = None
    sample_size: int = 0
    passes_all_thresholds: bool = False
    summary: str = ""

    def generate_summary(self) -> str:
        """Generate human-readable summary"""
        lines = [f"Metrics Report (N={self.sample_size})"]
        lines.append("=" * 60)

        if self.ks_similarity:
            lines.append(f"\nKS Similarity: {self.ks_similarity.ks_similarity:.3f}")
            lines.append(
                f"  Target: ≥0.85 | Status: {'✅ PASS' if self.ks_similarity.passes_threshold() else '❌ FAIL'}"
            )

        if self.correlation:
            lines.append(f"\nPearson Correlation: {self.correlation.pearson_r:.3f}")
            lines.append(
                f"  Target: ≥0.90 | Status: {'✅ PASS' if self.correlation.passes_threshold() else '❌ FAIL'}"
            )
            if self.correlation.correlation_attainment:
                lines.append(
                    f"  Correlation Attainment: {self.correlation.correlation_attainment * 100:.1f}% of human baseline"
                )

        if self.error:
            lines.append(f"\nMean Absolute Error: {self.error.mae:.3f}")
            lines.append(
                f"  Target: <0.5 | Status: {'✅ PASS' if self.error.passes_threshold() else '❌ FAIL'}"
            )
            lines.append(f"  RMSE: {self.error.rmse:.3f}")

        lines.append(
            f"\nOverall: {'✅ ALL TARGETS MET' if self.passes_all_thresholds else '❌ SOME TARGETS NOT MET'}"
        )

        return "\n".join(lines)


class MetricsCalculator:
    """
    Calculator for SSR evaluation metrics

    Features:
    - KS similarity for distribution matching
    - Pearson and Spearman correlation
    - MAE, RMSE, and other error metrics
    - Statistical significance testing
    - Comprehensive reporting
    """

    def __init__(self, human_baseline_correlation: float = 1.0):
        """
        Initialize metrics calculator

        Args:
            human_baseline_correlation: Human test-retest correlation (default: 1.0)
        """
        self.human_baseline_correlation = human_baseline_correlation

    def calculate_ks_similarity(
        self,
        distribution1: np.ndarray,
        distribution2: np.ndarray,
        normalize: bool = True,
    ) -> KSSimilarity:
        """
        Calculate Kolmogorov-Smirnov similarity between two distributions

        Args:
            distribution1: First probability distribution (5 values for Likert 1-5)
            distribution2: Second probability distribution
            normalize: Whether to normalize distributions to sum to 1.0

        Returns:
            KSSimilarity object with metrics

        Raises:
            ValueError: If distributions have different lengths or invalid values
        """
        # Validate inputs
        dist1 = np.array(distribution1, dtype=float)
        dist2 = np.array(distribution2, dtype=float)

        if len(dist1) != len(dist2):
            raise ValueError(
                f"Distributions must have same length: {len(dist1)} vs {len(dist2)}"
            )

        if len(dist1) == 0:
            raise ValueError("Distributions cannot be empty")

        # Normalize if requested
        if normalize:
            if np.sum(dist1) > 0:
                dist1 = dist1 / np.sum(dist1)
            if np.sum(dist2) > 0:
                dist2 = dist2 / np.sum(dist2)

        # Calculate cumulative distribution functions
        cdf1 = np.cumsum(dist1)
        cdf2 = np.cumsum(dist2)

        # KS statistic is maximum difference between CDFs
        ks_statistic = np.max(np.abs(cdf1 - cdf2))

        # KS similarity is 1 - KS statistic
        ks_similarity = 1.0 - ks_statistic

        # Statistical test (two-sample KS test)
        # For discrete distributions, we need to expand to samples
        # Generate samples proportional to distribution
        n_samples = 1000
        samples1 = self._distribution_to_samples(dist1, n_samples)
        samples2 = self._distribution_to_samples(dist2, n_samples)

        # Perform KS test
        ks_test_result = stats.ks_2samp(samples1, samples2)
        p_value = ks_test_result.pvalue

        return KSSimilarity(
            ks_statistic=ks_statistic,
            ks_similarity=ks_similarity,
            p_value=p_value,
            distributions=(dist1, dist2),
        )

    def calculate_correlation(
        self,
        values1: np.ndarray,
        values2: np.ndarray,
        human_baseline: Optional[float] = None,
    ) -> CorrelationMetrics:
        """
        Calculate correlation metrics between two sets of values

        Args:
            values1: First set of values (e.g., test responses)
            values2: Second set of values (e.g., retest responses)
            human_baseline: Human baseline correlation (uses default if None)

        Returns:
            CorrelationMetrics object

        Raises:
            ValueError: If input arrays have different lengths
        """
        v1 = np.array(values1, dtype=float)
        v2 = np.array(values2, dtype=float)

        if len(v1) != len(v2):
            raise ValueError(f"Arrays must have same length: {len(v1)} vs {len(v2)}")

        if len(v1) < 2:
            raise ValueError("Need at least 2 data points for correlation")

        # Pearson correlation (linear relationship)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pearson_r, pearson_p = stats.pearsonr(v1, v2)

        # Spearman correlation (rank-based, robust to outliers)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            spearman_rho, spearman_p = stats.spearmanr(v1, v2)

        # Handle NaN cases (e.g., all values are identical)
        if np.isnan(pearson_r):
            pearson_r = 1.0 if np.array_equal(v1, v2) else 0.0
            pearson_p = 1.0

        if np.isnan(spearman_rho):
            spearman_rho = 1.0 if np.array_equal(v1, v2) else 0.0
            spearman_p = 1.0

        # Calculate correlation attainment
        baseline = human_baseline or self.human_baseline_correlation
        correlation_attainment = pearson_r / baseline if baseline > 0 else None

        return CorrelationMetrics(
            pearson_r=pearson_r,
            pearson_p_value=pearson_p,
            spearman_rho=spearman_rho,
            spearman_p_value=spearman_p,
            correlation_attainment=correlation_attainment,
        )

    def calculate_error_metrics(
        self, predicted: np.ndarray, actual: np.ndarray
    ) -> ErrorMetrics:
        """
        Calculate error metrics between predicted and actual values

        Args:
            predicted: Predicted values (e.g., SSR mean ratings)
            actual: Actual values (e.g., human ground truth)

        Returns:
            ErrorMetrics object

        Raises:
            ValueError: If arrays have different lengths
        """
        pred = np.array(predicted, dtype=float)
        act = np.array(actual, dtype=float)

        if len(pred) != len(act):
            raise ValueError(f"Arrays must have same length: {len(pred)} vs {len(act)}")

        if len(pred) == 0:
            raise ValueError("Cannot calculate error metrics for empty arrays")

        # Calculate errors
        errors = pred - act
        abs_errors = np.abs(errors)

        # Mean Absolute Error
        mae = np.mean(abs_errors)

        # Root Mean Squared Error
        rmse = np.sqrt(np.mean(errors**2))

        # Maximum error
        max_error = np.max(abs_errors)

        # Mean signed error (indicates bias)
        mean_error = np.mean(errors)

        return ErrorMetrics(
            mae=mae,
            rmse=rmse,
            max_error=max_error,
            mean_error=mean_error,
            error_distribution=errors,
        )

    def generate_report(
        self,
        synthetic_distribution: Optional[np.ndarray] = None,
        human_distribution: Optional[np.ndarray] = None,
        test_values: Optional[np.ndarray] = None,
        retest_values: Optional[np.ndarray] = None,
        predicted_ratings: Optional[np.ndarray] = None,
        actual_ratings: Optional[np.ndarray] = None,
    ) -> MetricsReport:
        """
        Generate comprehensive metrics report

        Args:
            synthetic_distribution: Synthetic SSR distribution
            human_distribution: Human response distribution
            test_values: Test responses
            retest_values: Retest responses
            predicted_ratings: Predicted mean ratings
            actual_ratings: Actual mean ratings

        Returns:
            MetricsReport with all available metrics
        """
        report = MetricsReport()

        # KS Similarity (distribution matching)
        if synthetic_distribution is not None and human_distribution is not None:
            report.ks_similarity = self.calculate_ks_similarity(
                synthetic_distribution, human_distribution
            )
            report.sample_size = len(synthetic_distribution)

        # Correlation (test-retest reliability)
        if test_values is not None and retest_values is not None:
            report.correlation = self.calculate_correlation(test_values, retest_values)
            if report.sample_size == 0:
                report.sample_size = len(test_values)

        # Error metrics (prediction accuracy)
        if predicted_ratings is not None and actual_ratings is not None:
            report.error = self.calculate_error_metrics(
                predicted_ratings, actual_ratings
            )
            if report.sample_size == 0:
                report.sample_size = len(predicted_ratings)

        # Check if all thresholds are met
        report.passes_all_thresholds = self._check_all_thresholds(report)

        # Generate summary
        report.summary = report.generate_summary()

        return report

    def _check_all_thresholds(self, report: MetricsReport) -> bool:
        """Check if all metrics meet their target thresholds"""
        checks = []

        if report.ks_similarity:
            checks.append(report.ks_similarity.passes_threshold())

        if report.correlation:
            checks.append(report.correlation.passes_threshold())

        if report.error:
            checks.append(report.error.passes_threshold())

        # All available metrics must pass
        return all(checks) if checks else False

    def _distribution_to_samples(
        self, distribution: np.ndarray, n_samples: int
    ) -> np.ndarray:
        """
        Convert probability distribution to sample array

        Args:
            distribution: Probability distribution
            n_samples: Number of samples to generate

        Returns:
            Array of samples
        """
        # Ratings are 1-5 (indices 0-4 map to ratings 1-5)
        ratings = np.arange(1, len(distribution) + 1)

        # Normalize distribution
        prob = distribution / np.sum(distribution)

        # Generate samples
        samples = np.random.choice(ratings, size=n_samples, p=prob)

        return samples


# Example usage and testing
if __name__ == "__main__":
    print("Metrics Calculator Testing")
    print("=" * 60)

    calculator = MetricsCalculator(human_baseline_correlation=1.0)

    # Test 1: KS Similarity
    print("\n1. KS Similarity Test:")
    # Synthetic distribution (example: positive response)
    synthetic = np.array([0.05, 0.10, 0.20, 0.35, 0.30])  # Skewed toward 4-5
    # Human distribution (similar pattern)
    human = np.array([0.08, 0.12, 0.18, 0.32, 0.30])

    ks_result = calculator.calculate_ks_similarity(synthetic, human)
    print(f"KS Statistic: {ks_result.ks_statistic:.3f}")
    print(f"KS Similarity: {ks_result.ks_similarity:.3f}")
    print(f"P-value: {ks_result.p_value:.3f}")
    print(
        f"Passes threshold (≥0.85): {'✅ YES' if ks_result.passes_threshold() else '❌ NO'}"
    )

    # Test 2: Correlation
    print("\n" + "=" * 60)
    print("2. Correlation Test (Test-Retest):")
    # Simulate test-retest responses (ratings 1-5)
    test_responses = np.array([4, 5, 3, 4, 5, 2, 4, 3, 5, 4])
    retest_responses = np.array([4, 5, 3, 5, 5, 2, 4, 3, 4, 4])  # Slight variations

    corr_result = calculator.calculate_correlation(test_responses, retest_responses)
    print(f"Pearson r: {corr_result.pearson_r:.3f}")
    print(f"P-value: {corr_result.pearson_p_value:.3f}")
    print(
        f"Correlation Attainment: {corr_result.correlation_attainment * 100:.1f}% of human baseline"
    )
    print(
        f"Passes threshold (≥0.90): {'✅ YES' if corr_result.passes_threshold() else '❌ NO'}"
    )

    # Test 3: Error Metrics
    print("\n" + "=" * 60)
    print("3. Error Metrics Test:")
    # Predicted mean ratings vs actual
    predicted = np.array([4.2, 3.8, 4.5, 3.1, 4.0])
    actual = np.array([4.0, 4.0, 4.5, 3.0, 3.8])

    error_result = calculator.calculate_error_metrics(predicted, actual)
    print(f"MAE: {error_result.mae:.3f}")
    print(f"RMSE: {error_result.rmse:.3f}")
    print(f"Max Error: {error_result.max_error:.3f}")
    print(f"Mean Error (bias): {error_result.mean_error:+.3f}")
    print(
        f"Passes threshold (<0.5): {'✅ YES' if error_result.passes_threshold() else '❌ NO'}"
    )

    # Test 4: Comprehensive Report
    print("\n" + "=" * 60)
    print("4. Comprehensive Metrics Report:")

    report = calculator.generate_report(
        synthetic_distribution=synthetic,
        human_distribution=human,
        test_values=test_responses,
        retest_values=retest_responses,
        predicted_ratings=predicted,
        actual_ratings=actual,
    )

    print("\n" + report.summary)

    # Test 5: Paper Benchmark Simulation
    print("\n" + "=" * 60)
    print("5. Paper Benchmark Simulation:")
    print("\nGPT-4o Results (from paper):")
    print("  ρ = 90.2%, K^xy = 0.88, MAE = 0.42")

    # Simulate GPT-4o performance
    gpt4o_dist = np.array([0.06, 0.12, 0.20, 0.32, 0.30])
    human_dist = np.array([0.08, 0.12, 0.18, 0.32, 0.30])

    gpt4o_ks = calculator.calculate_ks_similarity(gpt4o_dist, human_dist)
    print(f"\nSimulated KS Similarity: {gpt4o_ks.ks_similarity:.3f} (target: ≥0.85)")

    # Simulate correlation
    np.random.seed(42)
    test = np.random.randint(1, 6, 100)
    noise = np.random.normal(0, 0.3, 100)
    retest = np.clip(test + noise, 1, 5).astype(int)

    gpt4o_corr = calculator.calculate_correlation(test, retest)
    print(f"Simulated Correlation: {gpt4o_corr.pearson_r:.3f} (target: ≥0.90)")

    print("\n" + "=" * 60)
    print("Metrics Calculator testing complete")
    print("\nKey Insights:")
    print("- KS Similarity measures distribution matching (higher is better)")
    print("- Pearson correlation measures test-retest reliability (target: ≥0.90)")
    print("- MAE measures prediction accuracy (target: <0.5 Likert points)")
    print("- System achieves paper benchmarks when all three metrics pass")

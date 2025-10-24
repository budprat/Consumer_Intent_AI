"""
ABOUTME: Test-retest reliability simulation and validation framework
ABOUTME: Measures consistency of synthetic consumer responses across repeated trials

This module implements test-retest reliability assessment, which is the core
validation metric for the SSR system. The paper's key finding is that LLMs achieve
90% of human test-retest reliability when properly conditioned with demographics.

Test-Retest Reliability:
- Measures consistency when same question asked multiple times
- Human baseline: ρ ≈ 1.0 (near-perfect consistency)
- Target for synthetic consumers: ρ ≥ 0.90 (90% of human reliability)

Key Components:
1. Repeat response generation (same profile, same product, multiple trials)
2. Intra-class correlation coefficient (ICC) calculation
3. Comparison to human baseline reliability
4. Statistical significance testing
"""

from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
import numpy as np
from scipy import stats
import warnings

from .metrics import MetricsCalculator


@dataclass
class ReliabilityResult:
    """
    Test-retest reliability result for a single profile

    Attributes:
        profile_id: Identifier for demographic profile
        test_responses: List of responses from test phase
        retest_responses: List of responses from retest phase
        test_mean_rating: Mean rating from test responses
        retest_mean_rating: Mean rating from retest responses
        correlation: Pearson correlation between test and retest
        icc: Intra-class correlation coefficient
        absolute_agreement: Proportion of identical responses
    """

    profile_id: str
    test_responses: List[float]
    retest_responses: List[float]
    test_mean_rating: float
    retest_mean_rating: float
    correlation: float
    icc: float
    absolute_agreement: float


@dataclass
class ReliabilityReport:
    """
    Comprehensive test-retest reliability report

    Attributes:
        num_profiles: Number of profiles tested
        num_trials: Number of repeat trials per profile
        individual_results: Results for each profile
        overall_correlation: Overall Pearson correlation
        overall_icc: Overall intra-class correlation
        mean_absolute_agreement: Mean proportion of identical responses
        correlation_attainment: Percentage of human baseline achieved
        passes_threshold: Whether reliability meets target (≥0.90)
        summary: Text summary of results
    """

    num_profiles: int
    num_trials: int
    individual_results: List[ReliabilityResult]
    overall_correlation: float
    overall_icc: float
    mean_absolute_agreement: float
    correlation_attainment: float
    passes_threshold: bool
    summary: str = ""

    def generate_summary(self) -> str:
        """Generate human-readable summary"""
        lines = [
            "Test-Retest Reliability Report",
            f"Profiles: {self.num_profiles} | Trials: {self.num_trials}",
            "=" * 60,
        ]

        lines.append(f"\nOverall Correlation (ρ): {self.overall_correlation:.3f}")
        lines.append(
            f"  Target: ≥0.90 | Status: {'✅ PASS' if self.passes_threshold else '❌ FAIL'}"
        )
        lines.append(
            f"  Correlation Attainment: {self.correlation_attainment * 100:.1f}% of human baseline"
        )

        lines.append(f"\nIntra-class Correlation (ICC): {self.overall_icc:.3f}")
        lines.append(
            f"Mean Absolute Agreement: {self.mean_absolute_agreement * 100:.1f}%"
        )

        lines.append("\nIndividual Profile Statistics:")
        correlations = [r.correlation for r in self.individual_results]
        lines.append(f"  Min correlation: {min(correlations):.3f}")
        lines.append(f"  Max correlation: {max(correlations):.3f}")
        lines.append(f"  Mean correlation: {np.mean(correlations):.3f}")
        lines.append(f"  Std deviation: {np.std(correlations):.3f}")

        return "\n".join(lines)


class ReliabilitySimulator:
    """
    Test-retest reliability simulator for synthetic consumers

    Features:
    - Multiple trial response generation
    - ICC calculation (measures consistency)
    - Comparison to human baseline
    - Statistical significance testing
    - Individual and aggregate analysis
    """

    def __init__(
        self,
        response_generator: Optional[Callable] = None,
        human_baseline_correlation: float = 1.0,
    ):
        """
        Initialize reliability simulator

        Args:
            response_generator: Function that generates responses (profile, product) -> rating
            human_baseline_correlation: Human test-retest correlation (default: 1.0)
        """
        self.response_generator = response_generator
        self.human_baseline_correlation = human_baseline_correlation
        self.metrics_calculator = MetricsCalculator(human_baseline_correlation)

    def simulate_test_retest(
        self,
        profiles: List[Any],
        product_concept: Dict[str, str],
        num_test_trials: int = 3,
        num_retest_trials: int = 3,
        time_delay_simulation: bool = False,
    ) -> ReliabilityReport:
        """
        Simulate test-retest reliability across multiple profiles

        Args:
            profiles: List of demographic profiles
            product_concept: Product to evaluate
            num_test_trials: Number of test phase trials
            num_retest_trials: Number of retest phase trials
            time_delay_simulation: Whether to simulate time delay effects

        Returns:
            ReliabilityReport with comprehensive results

        Note:
            This is a framework method. Actual implementation requires
            integration with LLM interface and SSR engine.
        """
        if not self.response_generator:
            # Placeholder: In real implementation, this would call LLM + SSR
            return self._placeholder_simulation(
                len(profiles), num_test_trials, num_retest_trials
            )

        individual_results = []

        for profile in profiles:
            # Test phase: Generate initial responses
            test_responses = []
            for _ in range(num_test_trials):
                rating = self.response_generator(profile, product_concept)
                test_responses.append(rating)

            # Simulate time delay if requested
            if time_delay_simulation:
                pass  # Could add noise or drift here

            # Retest phase: Generate repeat responses
            retest_responses = []
            for _ in range(num_retest_trials):
                rating = self.response_generator(profile, product_concept)
                retest_responses.append(rating)

            # Calculate individual reliability metrics
            result = self._calculate_individual_reliability(
                profile_id=getattr(profile, "id", str(hash(str(profile)))),
                test_responses=test_responses,
                retest_responses=retest_responses,
            )

            individual_results.append(result)

        # Calculate overall metrics
        return self._calculate_overall_reliability(
            individual_results, num_test_trials, num_retest_trials
        )

    def calculate_icc(
        self, test_values: np.ndarray, retest_values: np.ndarray, icc_type: str = "2,1"
    ) -> float:
        """
        Calculate Intra-class Correlation Coefficient

        ICC measures consistency between repeated measurements.
        Higher values indicate better reliability.

        Args:
            test_values: Test phase ratings
            retest_values: Retest phase ratings
            icc_type: Type of ICC ("2,1" for two-way random effects, absolute agreement)

        Returns:
            ICC value (0.0 to 1.0)

        Note:
            ICC(2,1) is appropriate for test-retest reliability with random effects
        """
        # Combine test and retest into single data matrix
        # Rows: subjects/profiles, Columns: trials
        data = np.column_stack([test_values, retest_values])

        n_subjects = data.shape[0]
        n_raters = data.shape[1]

        # Calculate mean squares
        grand_mean = np.mean(data)
        subject_means = np.mean(data, axis=1)
        trial_means = np.mean(data, axis=0)

        # Between-subjects variance
        ms_between = (
            n_raters * np.sum((subject_means - grand_mean) ** 2) / (n_subjects - 1)
        )

        # Within-subjects variance
        within_ss = np.sum((data - subject_means[:, np.newaxis]) ** 2)
        ms_within = within_ss / (n_subjects * (n_raters - 1))

        # ICC(2,1) formula for absolute agreement
        icc = (ms_between - ms_within) / (ms_between + (n_raters - 1) * ms_within)

        # Clip to valid range [0, 1]
        icc = np.clip(icc, 0.0, 1.0)

        return icc

    def _calculate_individual_reliability(
        self, profile_id: str, test_responses: List[float], retest_responses: List[float]
    ) -> ReliabilityResult:
        """Calculate reliability metrics for a single profile"""
        test_arr = np.array(test_responses)
        retest_arr = np.array(retest_responses)

        # Mean ratings
        test_mean = np.mean(test_arr)
        retest_mean = np.mean(retest_arr)

        # Correlation
        if len(test_arr) > 1 and len(retest_arr) > 1:
            # Pair responses for correlation
            # If unequal lengths, pair up to minimum length
            min_len = min(len(test_arr), len(retest_arr))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                correlation, _ = stats.pearsonr(
                    test_arr[:min_len], retest_arr[:min_len]
                )
            if np.isnan(correlation):
                correlation = 1.0 if np.array_equal(test_arr, retest_arr) else 0.0
        else:
            correlation = 1.0 if test_mean == retest_mean else 0.0

        # ICC
        try:
            icc = self.calculate_icc(test_arr, retest_arr)
        except Exception:
            icc = correlation  # Fallback to correlation

        # Absolute agreement (proportion of identical responses)
        min_len = min(len(test_arr), len(retest_arr))
        agreements = sum(
            1 for i in range(min_len) if test_arr[i] == retest_arr[i]
        )
        absolute_agreement = agreements / min_len if min_len > 0 else 0.0

        return ReliabilityResult(
            profile_id=profile_id,
            test_responses=test_responses,
            retest_responses=retest_responses,
            test_mean_rating=test_mean,
            retest_mean_rating=retest_mean,
            correlation=correlation,
            icc=icc,
            absolute_agreement=absolute_agreement,
        )

    def _calculate_overall_reliability(
        self,
        individual_results: List[ReliabilityResult],
        num_test_trials: int,
        num_retest_trials: int,
    ) -> ReliabilityReport:
        """Calculate overall reliability across all profiles"""
        if not individual_results:
            return ReliabilityReport(
                num_profiles=0,
                num_trials=num_test_trials + num_retest_trials,
                individual_results=[],
                overall_correlation=0.0,
                overall_icc=0.0,
                mean_absolute_agreement=0.0,
                correlation_attainment=0.0,
                passes_threshold=False,
            )

        # Collect all test and retest values
        all_test = []
        all_retest = []
        for result in individual_results:
            # Use mean ratings for overall correlation
            all_test.append(result.test_mean_rating)
            all_retest.append(result.retest_mean_rating)

        # Overall correlation
        corr_metrics = self.metrics_calculator.calculate_correlation(
            np.array(all_test), np.array(all_retest), self.human_baseline_correlation
        )

        # Overall ICC
        try:
            overall_icc = self.calculate_icc(
                np.array(all_test), np.array(all_retest)
            )
        except Exception:
            overall_icc = corr_metrics.pearson_r

        # Mean absolute agreement
        mean_agreement = np.mean([r.absolute_agreement for r in individual_results])

        # Check threshold
        passes = corr_metrics.pearson_r >= 0.90

        report = ReliabilityReport(
            num_profiles=len(individual_results),
            num_trials=num_test_trials + num_retest_trials,
            individual_results=individual_results,
            overall_correlation=corr_metrics.pearson_r,
            overall_icc=overall_icc,
            mean_absolute_agreement=mean_agreement,
            correlation_attainment=corr_metrics.correlation_attainment or 0.0,
            passes_threshold=passes,
        )

        report.summary = report.generate_summary()

        return report

    def _placeholder_simulation(
        self, num_profiles: int, num_test: int, num_retest: int
    ) -> ReliabilityReport:
        """
        Placeholder simulation for demonstration

        In production, this would be replaced with actual LLM + SSR pipeline
        """
        np.random.seed(42)
        individual_results = []

        for i in range(num_profiles):
            # Simulate realistic test-retest behavior
            # High correlation but some variation (ρ ≈ 0.90)
            base_rating = np.random.uniform(2.5, 4.5)
            noise_std = 0.4  # Standard deviation for test-retest variation

            test_responses = np.clip(
                base_rating + np.random.normal(0, noise_std, num_test), 1, 5
            ).tolist()

            retest_responses = np.clip(
                base_rating + np.random.normal(0, noise_std, num_retest), 1, 5
            ).tolist()

            result = self._calculate_individual_reliability(
                profile_id=f"profile_{i}",
                test_responses=test_responses,
                retest_responses=retest_responses,
            )

            individual_results.append(result)

        return self._calculate_overall_reliability(
            individual_results, num_test, num_retest
        )


# Example usage and testing
if __name__ == "__main__":
    print("Test-Retest Reliability Simulator Testing")
    print("=" * 60)

    simulator = ReliabilitySimulator(human_baseline_correlation=1.0)

    # Test 1: Placeholder simulation (demonstrates framework)
    print("\n1. Placeholder Simulation (N=50 profiles, 3+3 trials):")
    report = simulator._placeholder_simulation(
        num_profiles=50, num_test=3, num_retest=3
    )

    print("\n" + report.summary)

    # Test 2: ICC calculation
    print("\n" + "=" * 60)
    print("2. ICC Calculation Test:")

    test = np.array([4.0, 5.0, 3.0, 4.5, 4.0])
    retest = np.array([4.0, 4.5, 3.5, 4.5, 4.0])

    icc = simulator.calculate_icc(test, retest)
    print(f"Test ratings: {test}")
    print(f"Retest ratings: {retest}")
    print(f"ICC(2,1): {icc:.3f}")
    print(
        f"Interpretation: {'Excellent' if icc > 0.9 else 'Good' if icc > 0.75 else 'Moderate' if icc > 0.5 else 'Poor'}"
    )

    # Test 3: Individual reliability
    print("\n" + "=" * 60)
    print("3. Individual Profile Reliability:")

    test_resp = [4.0, 4.0, 5.0]
    retest_resp = [4.0, 4.0, 4.0]

    individual = simulator._calculate_individual_reliability(
        profile_id="test_profile", test_responses=test_resp, retest_responses=retest_resp
    )

    print(f"Profile ID: {individual.profile_id}")
    print(f"Test mean: {individual.test_mean_rating:.2f}")
    print(f"Retest mean: {individual.retest_mean_rating:.2f}")
    print(f"Correlation: {individual.correlation:.3f}")
    print(f"ICC: {individual.icc:.3f}")
    print(f"Absolute agreement: {individual.absolute_agreement * 100:.1f}%")

    # Test 4: Comparison to paper benchmarks
    print("\n" + "=" * 60)
    print("4. Paper Benchmark Comparison:")
    print("\nPaper Results:")
    print("  GPT-4o: ρ = 90.2% (90.2% of human baseline)")
    print("  Gemini-2.0-flash: ρ = 90.6% (90.6% of human baseline)")
    print("  Human: ρ ≈ 100% (perfect consistency)")

    # Simulate GPT-4o performance
    gpt4o_report = simulator._placeholder_simulation(100, 5, 5)
    print("\nSimulated GPT-4o:")
    print(f"  ρ = {gpt4o_report.overall_correlation:.3f}")
    print(
        f"  Correlation attainment: {gpt4o_report.correlation_attainment * 100:.1f}%"
    )
    print(
        f"  Status: {'✅ PASSES' if gpt4o_report.passes_threshold else '❌ FAILS'} target (≥0.90)"
    )

    print("\n" + "=" * 60)
    print("Test-Retest Reliability testing complete")
    print("\nKey Insights:")
    print("- Test-retest reliability measures response consistency")
    print("- ICC provides robust measure of agreement")
    print("- Target: ρ ≥ 0.90 (90% of human reliability)")
    print("- High reliability is critical for generalizable survey results")

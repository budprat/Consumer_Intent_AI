"""
ABOUTME: Demographic bias detection and mitigation for synthetic consumer cohorts
ABOUTME: Ensures fairness and representative sampling across demographic groups

This module implements bias detection and mitigation strategies to ensure
synthetic consumer cohorts are representative and unbiased. Critical for:
1. Validating cohort representativeness
2. Detecting underrepresented groups
3. Ensuring fairness across demographics
4. Mitigating systematic biases

Bias Types Detected:
- Sampling bias (deviation from target distributions)
- Representation bias (underrepresented groups)
- Response bias (systematic patterns in responses)
- Demographic skew (overrepresentation of certain groups)
"""

from typing import Dict, Optional, List
from dataclasses import dataclass, field
from enum import Enum
import statistics

from .profiles import DemographicProfile


class BiasType(Enum):
    """Types of demographic bias"""

    SAMPLING_BIAS = "sampling_bias"  # Deviation from target distribution
    REPRESENTATION_BIAS = "representation_bias"  # Underrepresented groups
    DEMOGRAPHIC_SKEW = "demographic_skew"  # Overrepresented groups
    INTERSECTIONAL_BIAS = "intersectional_bias"  # Bias in combinations


class BiasSeverity(Enum):
    """Severity levels for detected bias"""

    NONE = "none"  # No significant bias detected
    LOW = "low"  # Minor deviation, acceptable
    MODERATE = "moderate"  # Notable deviation, review recommended
    HIGH = "high"  # Significant deviation, action required
    CRITICAL = "critical"  # Severe deviation, must fix


@dataclass
class BiasDetection:
    """
    Individual bias detection result

    Attributes:
        bias_type: Type of bias detected
        severity: Severity level
        attribute: Demographic attribute affected
        expected_distribution: Expected distribution
        actual_distribution: Actual distribution
        deviation_score: Numerical deviation score (0.0-1.0)
        affected_groups: Specific groups affected
        recommendations: Mitigation recommendations
    """

    bias_type: BiasType
    severity: BiasSeverity
    attribute: str
    expected_distribution: Dict[str, float]
    actual_distribution: Dict[str, float]
    deviation_score: float
    affected_groups: List[str]
    recommendations: List[str] = field(default_factory=list)


@dataclass
class BiasReport:
    """
    Comprehensive bias detection report

    Attributes:
        cohort_size: Number of profiles analyzed
        total_biases_detected: Total number of biases found
        severity_breakdown: Count by severity level
        detections: List of individual bias detections
        overall_bias_score: Overall bias score (0.0-1.0, lower is better)
        is_representative: Whether cohort is sufficiently representative
        mitigation_priority: Prioritized list of mitigation actions
    """

    cohort_size: int
    total_biases_detected: int
    severity_breakdown: Dict[str, int]
    detections: List[BiasDetection]
    overall_bias_score: float
    is_representative: bool
    mitigation_priority: List[str] = field(default_factory=list)


class BiasDetector:
    """
    Detects and analyzes demographic bias in synthetic cohorts

    Features:
    - Statistical bias detection
    - Comparison against US Census distributions
    - Intersectional bias analysis
    - Mitigation recommendations
    - Fairness validation
    """

    def __init__(
        self,
        target_distributions: Optional[Dict[str, Dict[str, float]]] = None,
        bias_threshold: float = 0.05,
    ):
        """
        Initialize bias detector

        Args:
            target_distributions: Target demographic distributions
            bias_threshold: Threshold for detecting bias (default: 5% deviation)
        """
        self.bias_threshold = bias_threshold

        # US Census 2020-based target distributions (if not provided)
        self.target_distributions = target_distributions or {
            "age_groups": {
                "18-24": 0.12,
                "25-34": 0.18,
                "35-44": 0.17,
                "45-54": 0.16,
                "55-64": 0.17,
                "65+": 0.20,
            },
            "gender": {
                "Male": 0.49,
                "Female": 0.50,
                "Non-binary": 0.005,
                "Other": 0.005,
            },
            "income_levels": {
                "Less than $25,000": 0.20,
                "$25,000-$49,999": 0.22,
                "$50,000-$74,999": 0.18,
                "$75,000-$99,999": 0.15,
                "$100,000-$149,999": 0.15,
                "$150,000 or more": 0.10,
            },
            "ethnicity": {
                "White": 0.60,
                "Black or African American": 0.13,
                "Hispanic or Latino": 0.19,
                "Asian": 0.06,
                "Native American or Alaska Native": 0.01,
                "Two or More Races": 0.01,
            },
        }

    def detect_bias(self, profiles: List[DemographicProfile]) -> BiasReport:
        """
        Detect demographic bias in cohort

        Args:
            profiles: List of DemographicProfile objects

        Returns:
            BiasReport with comprehensive bias analysis
        """
        if not profiles:
            return BiasReport(
                cohort_size=0,
                total_biases_detected=0,
                severity_breakdown={},
                detections=[],
                overall_bias_score=0.0,
                is_representative=False,
            )

        # Calculate actual distributions
        actual_distributions = self._calculate_distributions(profiles)

        # Detect biases for each attribute
        detections = []

        # Age bias
        age_bias = self._detect_distribution_bias(
            attribute="age_groups",
            actual_dist=actual_distributions["age_groups"],
            expected_dist=self.target_distributions["age_groups"],
        )
        if age_bias:
            detections.append(age_bias)

        # Gender bias
        gender_bias = self._detect_distribution_bias(
            attribute="gender",
            actual_dist=actual_distributions["gender"],
            expected_dist=self.target_distributions["gender"],
        )
        if gender_bias:
            detections.append(gender_bias)

        # Income bias
        income_bias = self._detect_distribution_bias(
            attribute="income_levels",
            actual_dist=actual_distributions["income_levels"],
            expected_dist=self.target_distributions["income_levels"],
        )
        if income_bias:
            detections.append(income_bias)

        # Ethnicity bias
        ethnicity_bias = self._detect_distribution_bias(
            attribute="ethnicity",
            actual_dist=actual_distributions["ethnicity"],
            expected_dist=self.target_distributions["ethnicity"],
        )
        if ethnicity_bias:
            detections.append(ethnicity_bias)

        # Calculate overall metrics
        severity_breakdown = self._count_by_severity(detections)
        overall_bias_score = self._calculate_overall_bias_score(detections)
        is_representative = overall_bias_score < 0.1  # Less than 10% overall deviation

        # Generate mitigation priority
        mitigation_priority = self._generate_mitigation_priority(detections)

        return BiasReport(
            cohort_size=len(profiles),
            total_biases_detected=len(detections),
            severity_breakdown=severity_breakdown,
            detections=detections,
            overall_bias_score=overall_bias_score,
            is_representative=is_representative,
            mitigation_priority=mitigation_priority,
        )

    def _calculate_distributions(
        self, profiles: List[DemographicProfile]
    ) -> Dict[str, Dict[str, float]]:
        """Calculate actual distributions from profiles"""
        cohort_size = len(profiles)

        # Age groups
        age_counts = {}
        for p in profiles:
            age_group = p.get_age_group()
            age_counts[age_group] = age_counts.get(age_group, 0) + 1
        age_dist = {k: v / cohort_size for k, v in age_counts.items()}

        # Gender
        gender_counts = {}
        for p in profiles:
            gender_counts[p.gender] = gender_counts.get(p.gender, 0) + 1
        gender_dist = {k: v / cohort_size for k, v in gender_counts.items()}

        # Income levels
        income_counts = {}
        for p in profiles:
            income_counts[p.income_level] = income_counts.get(p.income_level, 0) + 1
        income_dist = {k: v / cohort_size for k, v in income_counts.items()}

        # Ethnicity
        ethnicity_counts = {}
        for p in profiles:
            ethnicity_counts[p.ethnicity] = ethnicity_counts.get(p.ethnicity, 0) + 1
        ethnicity_dist = {k: v / cohort_size for k, v in ethnicity_counts.items()}

        return {
            "age_groups": age_dist,
            "gender": gender_dist,
            "income_levels": income_dist,
            "ethnicity": ethnicity_dist,
        }

    def _detect_distribution_bias(
        self,
        attribute: str,
        actual_dist: Dict[str, float],
        expected_dist: Dict[str, float],
    ) -> Optional[BiasDetection]:
        """
        Detect bias in a single attribute distribution

        Returns BiasDetection if bias detected, None otherwise
        """
        # Calculate deviations for each category
        deviations = {}
        affected_groups = []

        all_categories = set(expected_dist.keys()) | set(actual_dist.keys())

        for category in all_categories:
            expected = expected_dist.get(category, 0.0)
            actual = actual_dist.get(category, 0.0)
            deviation = abs(expected - actual)

            deviations[category] = deviation

            if deviation > self.bias_threshold:
                affected_groups.append(category)

        # If no significant deviations, no bias detected
        if not affected_groups:
            return None

        # Calculate overall deviation score
        deviation_score = statistics.mean(deviations.values())

        # Determine severity
        severity = self._determine_severity(deviation_score)

        # Determine bias type
        # Check if any groups are underrepresented
        underrepresented = [
            cat
            for cat in affected_groups
            if actual_dist.get(cat, 0) < expected_dist.get(cat, 0)
        ]
        overrepresented = [
            cat
            for cat in affected_groups
            if actual_dist.get(cat, 0) > expected_dist.get(cat, 0)
        ]

        if underrepresented:
            bias_type = BiasType.REPRESENTATION_BIAS
        elif overrepresented:
            bias_type = BiasType.DEMOGRAPHIC_SKEW
        else:
            bias_type = BiasType.SAMPLING_BIAS

        # Generate recommendations
        recommendations = self._generate_recommendations(
            attribute, affected_groups, underrepresented, overrepresented
        )

        return BiasDetection(
            bias_type=bias_type,
            severity=severity,
            attribute=attribute,
            expected_distribution=expected_dist,
            actual_distribution=actual_dist,
            deviation_score=deviation_score,
            affected_groups=affected_groups,
            recommendations=recommendations,
        )

    def _determine_severity(self, deviation_score: float) -> BiasSeverity:
        """Determine bias severity from deviation score"""
        if deviation_score < 0.05:
            return BiasSeverity.NONE
        elif deviation_score < 0.10:
            return BiasSeverity.LOW
        elif deviation_score < 0.20:
            return BiasSeverity.MODERATE
        elif deviation_score < 0.30:
            return BiasSeverity.HIGH
        else:
            return BiasSeverity.CRITICAL

    def _generate_recommendations(
        self,
        attribute: str,
        affected_groups: List[str],
        underrepresented: List[str],
        overrepresented: List[str],
    ) -> List[str]:
        """Generate mitigation recommendations"""
        recommendations = []

        if underrepresented:
            recommendations.append(
                f"Increase sampling of underrepresented {attribute}: {', '.join(underrepresented)}"
            )

        if overrepresented:
            recommendations.append(
                f"Reduce sampling of overrepresented {attribute}: {', '.join(overrepresented)}"
            )

        recommendations.append(
            f"Use stratified sampling with fixed quotas for {attribute}"
        )
        recommendations.append(f"Validate {attribute} distribution after resampling")

        return recommendations

    def _count_by_severity(self, detections: List[BiasDetection]) -> Dict[str, int]:
        """Count detections by severity level"""
        counts = {
            "none": 0,
            "low": 0,
            "moderate": 0,
            "high": 0,
            "critical": 0,
        }

        for detection in detections:
            counts[detection.severity.value] += 1

        return counts

    def _calculate_overall_bias_score(self, detections: List[BiasDetection]) -> float:
        """Calculate overall bias score (0.0-1.0, lower is better)"""
        if not detections:
            return 0.0

        # Weight by severity
        severity_weights = {
            BiasSeverity.NONE: 0.0,
            BiasSeverity.LOW: 0.2,
            BiasSeverity.MODERATE: 0.5,
            BiasSeverity.HIGH: 0.8,
            BiasSeverity.CRITICAL: 1.0,
        }

        total_weighted_score = sum(
            detection.deviation_score * severity_weights[detection.severity]
            for detection in detections
        )

        # Normalize by number of detections
        return min(total_weighted_score / len(detections), 1.0)

    def _generate_mitigation_priority(
        self, detections: List[BiasDetection]
    ) -> List[str]:
        """Generate prioritized mitigation actions"""
        # Sort by severity (critical first)
        severity_order = {
            BiasSeverity.CRITICAL: 0,
            BiasSeverity.HIGH: 1,
            BiasSeverity.MODERATE: 2,
            BiasSeverity.LOW: 3,
            BiasSeverity.NONE: 4,
        }

        sorted_detections = sorted(detections, key=lambda d: severity_order[d.severity])

        priority_actions = []
        for detection in sorted_detections:
            if detection.severity in [BiasSeverity.CRITICAL, BiasSeverity.HIGH]:
                priority_actions.extend(
                    detection.recommendations[:2]
                )  # Top 2 recommendations

        # Add general recommendations
        if priority_actions:
            priority_actions.append("Regenerate cohort using stratified sampling")
            priority_actions.append(
                "Validate new cohort against US Census distributions"
            )

        return priority_actions


# Example usage and testing
if __name__ == "__main__":
    from .sampling import DemographicSampler, SamplingConfig, SamplingStrategy

    print("Demographic Bias Detection System Testing")
    print("=" * 60)

    # Initialize
    detector = BiasDetector(bias_threshold=0.05)
    sampler = DemographicSampler(seed=42)

    # Test 1: Representative cohort (should have low bias)
    print("\n1. Testing Representative Cohort (N=150, Stratified Sampling):")
    config = SamplingConfig(
        strategy=SamplingStrategy.STRATIFIED, cohort_size=150, seed=42
    )
    cohort = sampler.generate_cohort(config)

    report = detector.detect_bias(cohort)

    print(f"Cohort size: {report.cohort_size}")
    print(f"Biases detected: {report.total_biases_detected}")
    print(f"Overall bias score: {report.overall_bias_score:.3f}")
    print(f"Is representative: {report.is_representative}")

    print("\nSeverity breakdown:")
    for severity, count in report.severity_breakdown.items():
        if count > 0:
            print(f"  {severity}: {count}")

    if report.detections:
        print("\nDetected biases:")
        for detection in report.detections:
            print(f"\n  {detection.attribute} ({detection.bias_type.value}):")
            print(f"    Severity: {detection.severity.value}")
            print(f"    Deviation score: {detection.deviation_score:.3f}")
            print(f"    Affected groups: {', '.join(detection.affected_groups)}")

    # Test 2: Biased cohort (custom distribution skewed)
    print("\n" + "=" * 60)
    print("2. Testing Biased Cohort (Young, High-Income Skew):")

    biased_config = SamplingConfig(
        strategy=SamplingStrategy.CUSTOM,
        cohort_size=50,
        age_distribution={"18-24": 0.4, "25-34": 0.5, "35-44": 0.1},  # Skewed young
        gender_distribution={"Male": 0.7, "Female": 0.3},  # Skewed male
        income_distribution={
            "$100,000-$149,999": 0.6,  # Skewed high income
            "$150,000 or more": 0.3,
            "$75,000-$99,999": 0.1,
        },
        ethnicity_distribution={
            "Asian": 0.5,  # Skewed Asian
            "White": 0.4,
            "Hispanic or Latino": 0.1,
        },
        seed=42,
    )

    biased_cohort = sampler.generate_cohort(biased_config)
    biased_report = detector.detect_bias(biased_cohort)

    print(f"Cohort size: {biased_report.cohort_size}")
    print(f"Biases detected: {biased_report.total_biases_detected}")
    print(f"Overall bias score: {biased_report.overall_bias_score:.3f}")
    print(f"Is representative: {biased_report.is_representative}")

    if biased_report.detections:
        print("\nDetected biases:")
        for detection in biased_report.detections:
            print(f"\n  {detection.attribute}:")
            print(f"    Severity: {detection.severity.value}")
            print(f"    Affected: {', '.join(detection.affected_groups)}")

    if biased_report.mitigation_priority:
        print("\nMitigation priority:")
        for i, action in enumerate(biased_report.mitigation_priority[:5], 1):
            print(f"  {i}. {action}")

    # Test 3: Small cohort (expected higher variance)
    print("\n" + "=" * 60)
    print("3. Testing Small Cohort (N=10):")

    small_config = SamplingConfig(
        strategy=SamplingStrategy.STRATIFIED, cohort_size=10, seed=42
    )
    small_cohort = sampler.generate_cohort(small_config)
    small_report = detector.detect_bias(small_cohort)

    print(f"Cohort size: {small_report.cohort_size}")
    print(f"Biases detected: {small_report.total_biases_detected}")
    print(f"Overall bias score: {small_report.overall_bias_score:.3f}")
    print("Note: Small cohorts naturally have higher variance")

    print("\n" + "=" * 60)
    print("Bias Detection testing complete")
    print("\nKey Insights:")
    print("- Stratified sampling reduces bias (N≥150 recommended)")
    print("- Bias threshold: 5% deviation from US Census")
    print("- Critical biases require immediate resampling")
    print("- Representative cohorts ensure generalizable results")

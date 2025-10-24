"""
ABOUTME: Demographic sampling strategies for representative synthetic cohorts
ABOUTME: Implements stratified, quota, and custom sampling based on US Census data

This module ensures synthetic consumer cohorts are representative of target
populations, which is critical for generalizable survey results.

Sampling Strategies:
1. Stratified Sampling: Proportional representation across demographics
2. Quota Sampling: Fixed quotas per category
3. Custom Sampling: User-defined demographic mix

Paper Approach: N=150-400 per survey with demographic matching to human cohorts
"""

import random
from typing import List, Dict, Any, Optional
from enum import Enum
from dataclasses import dataclass

from .profiles import DemographicProfile, Location


class SamplingStrategy(Enum):
    """Sampling strategy types"""

    STRATIFIED = "stratified"  # Proportional representation
    QUOTA = "quota"  # Fixed quotas per category
    CUSTOM = "custom"  # User-defined distribution


@dataclass
class SamplingConfig:
    """
    Configuration for demographic sampling

    Attributes:
        strategy: Sampling strategy to use
        cohort_size: Number of profiles to generate
        age_distribution: Age group distribution (optional)
        gender_distribution: Gender distribution (optional)
        income_distribution: Income level distribution (optional)
        ethnicity_distribution: Ethnicity distribution (optional)
        seed: Random seed for reproducibility
    """

    strategy: SamplingStrategy
    cohort_size: int
    age_distribution: Optional[Dict[str, float]] = None
    gender_distribution: Optional[Dict[str, float]] = None
    income_distribution: Optional[Dict[str, float]] = None
    ethnicity_distribution: Optional[Dict[str, float]] = None
    seed: Optional[int] = None


class DemographicSampler:
    """
    Generates representative demographic cohorts

    Features:
    - US Census-based distributions
    - Multiple sampling strategies
    - Reproducible sampling with seeds
    - Validation of distributions
    """

    def __init__(self, seed: Optional[int] = None):
        """
        Initialize demographic sampler

        Args:
            seed: Random seed for reproducibility
        """
        if seed is not None:
            random.seed(seed)

        # US Census 2020-based distributions (approximate)
        self.us_census_distributions = {
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

        # US cities and states for location sampling
        self.us_locations = [
            ("New York", "NY", "Northeast"),
            ("Los Angeles", "CA", "West Coast"),
            ("Chicago", "IL", "Midwest"),
            ("Houston", "TX", "Southwest"),
            ("Phoenix", "AZ", "Southwest"),
            ("Philadelphia", "PA", "Northeast"),
            ("San Antonio", "TX", "Southwest"),
            ("San Diego", "CA", "West Coast"),
            ("Dallas", "TX", "Southwest"),
            ("San Francisco", "CA", "West Coast"),
            ("Austin", "TX", "Southwest"),
            ("Jacksonville", "FL", "Southeast"),
            ("Columbus", "OH", "Midwest"),
            ("Indianapolis", "IN", "Midwest"),
            ("Charlotte", "NC", "Southeast"),
            ("Seattle", "WA", "West Coast"),
            ("Denver", "CO", "Mountain"),
            ("Boston", "MA", "Northeast"),
            ("Portland", "OR", "West Coast"),
            ("Miami", "FL", "Southeast"),
        ]

    def generate_cohort(self, config: SamplingConfig) -> List[DemographicProfile]:
        """
        Generate demographic cohort based on sampling configuration

        Args:
            config: Sampling configuration

        Returns:
            List of DemographicProfile objects

        Raises:
            ValueError: If configuration is invalid
        """
        if config.cohort_size <= 0:
            raise ValueError(f"Cohort size must be positive, got {config.cohort_size}")

        if config.seed is not None:
            random.seed(config.seed)

        if config.strategy == SamplingStrategy.STRATIFIED:
            return self._stratified_sampling(config)
        elif config.strategy == SamplingStrategy.QUOTA:
            return self._quota_sampling(config)
        elif config.strategy == SamplingStrategy.CUSTOM:
            return self._custom_sampling(config)
        else:
            raise ValueError(f"Unknown sampling strategy: {config.strategy}")

    def _stratified_sampling(self, config: SamplingConfig) -> List[DemographicProfile]:
        """
        Stratified sampling: Proportional representation

        Uses US Census distributions by default, or custom distributions if provided
        """
        # Use provided distributions or defaults
        age_dist = config.age_distribution or self.us_census_distributions["age_groups"]
        gender_dist = (
            config.gender_distribution or self.us_census_distributions["gender"]
        )
        income_dist = (
            config.income_distribution or self.us_census_distributions["income_levels"]
        )
        ethnicity_dist = (
            config.ethnicity_distribution or self.us_census_distributions["ethnicity"]
        )

        # Validate distributions sum to ~1.0
        self._validate_distribution(age_dist, "age")
        self._validate_distribution(gender_dist, "gender")
        self._validate_distribution(income_dist, "income")
        self._validate_distribution(ethnicity_dist, "ethnicity")

        profiles = []
        for _ in range(config.cohort_size):
            # Sample from each distribution
            age_group = self._weighted_choice(age_dist)
            age = self._sample_age_from_group(age_group)

            gender = self._weighted_choice(gender_dist)
            income_level = self._weighted_choice(income_dist)
            ethnicity = self._weighted_choice(ethnicity_dist)

            # Sample location
            city, state, region = random.choice(self.us_locations)
            location = Location(city=city, state=state, country="USA", region=region)

            profile = DemographicProfile(
                age=age,
                gender=gender,
                income_level=income_level,
                location=location,
                ethnicity=ethnicity,
            )
            profiles.append(profile)

        return profiles

    def _quota_sampling(self, config: SamplingConfig) -> List[DemographicProfile]:
        """
        Quota sampling: Fixed quotas per category

        Ensures minimum representation for each demographic group
        """
        if not config.age_distribution:
            raise ValueError("Quota sampling requires age_distribution")

        profiles = []

        # Calculate quotas for each age group
        for age_group, proportion in config.age_distribution.items():
            quota = int(config.cohort_size * proportion)

            for _ in range(quota):
                age = self._sample_age_from_group(age_group)

                # Sample other attributes
                gender_dist = (
                    config.gender_distribution or self.us_census_distributions["gender"]
                )
                income_dist = (
                    config.income_distribution
                    or self.us_census_distributions["income_levels"]
                )
                ethnicity_dist = (
                    config.ethnicity_distribution
                    or self.us_census_distributions["ethnicity"]
                )

                gender = self._weighted_choice(gender_dist)
                income_level = self._weighted_choice(income_dist)
                ethnicity = self._weighted_choice(ethnicity_dist)

                city, state, region = random.choice(self.us_locations)
                location = Location(
                    city=city, state=state, country="USA", region=region
                )

                profile = DemographicProfile(
                    age=age,
                    gender=gender,
                    income_level=income_level,
                    location=location,
                    ethnicity=ethnicity,
                )
                profiles.append(profile)

        # Fill remaining slots if quotas don't sum to exactly cohort_size
        while len(profiles) < config.cohort_size:
            age_group = self._weighted_choice(config.age_distribution)
            age = self._sample_age_from_group(age_group)

            gender_dist = (
                config.gender_distribution or self.us_census_distributions["gender"]
            )
            income_dist = (
                config.income_distribution
                or self.us_census_distributions["income_levels"]
            )
            ethnicity_dist = (
                config.ethnicity_distribution
                or self.us_census_distributions["ethnicity"]
            )

            gender = self._weighted_choice(gender_dist)
            income_level = self._weighted_choice(income_dist)
            ethnicity = self._weighted_choice(ethnicity_dist)

            city, state, region = random.choice(self.us_locations)
            location = Location(city=city, state=state, country="USA", region=region)

            profile = DemographicProfile(
                age=age,
                gender=gender,
                income_level=income_level,
                location=location,
                ethnicity=ethnicity,
            )
            profiles.append(profile)

        return profiles

    def _custom_sampling(self, config: SamplingConfig) -> List[DemographicProfile]:
        """
        Custom sampling: User-defined distribution

        Requires all distributions to be specified
        """
        if not all(
            [
                config.age_distribution,
                config.gender_distribution,
                config.income_distribution,
                config.ethnicity_distribution,
            ]
        ):
            raise ValueError(
                "Custom sampling requires all distributions to be specified"
            )

        return self._stratified_sampling(config)

    def _weighted_choice(self, distribution: Dict[str, float]) -> str:
        """Select item based on weighted distribution"""
        items = list(distribution.keys())
        weights = list(distribution.values())
        return random.choices(items, weights=weights, k=1)[0]

    def _sample_age_from_group(self, age_group: str) -> int:
        """Sample specific age from age group"""
        age_ranges = {
            "18-24": (18, 24),
            "25-34": (25, 34),
            "35-44": (35, 44),
            "45-54": (45, 54),
            "55-64": (55, 64),
            "65+": (65, 80),
        }

        if age_group not in age_ranges:
            raise ValueError(f"Unknown age group: {age_group}")

        min_age, max_age = age_ranges[age_group]
        return random.randint(min_age, max_age)

    def _validate_distribution(self, distribution: Dict[str, float], name: str):
        """Validate that distribution sums to approximately 1.0"""
        total = sum(distribution.values())
        if not 0.95 <= total <= 1.05:
            raise ValueError(
                f"{name} distribution must sum to ~1.0, got {total:.2f}. "
                f"Distribution: {distribution}"
            )

    def get_cohort_statistics(
        self, profiles: List[DemographicProfile]
    ) -> Dict[str, Any]:
        """
        Calculate statistics for generated cohort

        Args:
            profiles: List of demographic profiles

        Returns:
            Dictionary with cohort statistics
        """
        if not profiles:
            return {}

        # Age statistics
        ages = [p.age for p in profiles]
        age_groups = [p.get_age_group() for p in profiles]

        # Gender distribution
        gender_counts = {}
        for p in profiles:
            gender_counts[p.gender] = gender_counts.get(p.gender, 0) + 1

        # Income distribution
        income_groups = [p.get_income_group() for p in profiles]
        income_group_counts = {}
        for group in income_groups:
            income_group_counts[group] = income_group_counts.get(group, 0) + 1

        # Ethnicity distribution
        ethnicity_counts = {}
        for p in profiles:
            ethnicity_counts[p.ethnicity] = ethnicity_counts.get(p.ethnicity, 0) + 1

        # Location distribution
        state_counts = {}
        for p in profiles:
            state_counts[p.location.state] = state_counts.get(p.location.state, 0) + 1

        cohort_size = len(profiles)

        return {
            "cohort_size": cohort_size,
            "age_statistics": {
                "mean": sum(ages) / cohort_size,
                "min": min(ages),
                "max": max(ages),
                "groups": self._count_to_distribution(age_groups),
            },
            "gender_distribution": {
                k: v / cohort_size for k, v in gender_counts.items()
            },
            "income_distribution": {
                k: v / cohort_size for k, v in income_group_counts.items()
            },
            "ethnicity_distribution": {
                k: v / cohort_size for k, v in ethnicity_counts.items()
            },
            "location_distribution": {
                k: v / cohort_size for k, v in state_counts.items()
            },
        }

    def _count_to_distribution(self, items: List[str]) -> Dict[str, float]:
        """Convert list of items to distribution"""
        counts = {}
        for item in items:
            counts[item] = counts.get(item, 0) + 1

        total = len(items)
        return {k: v / total for k, v in counts.items()}


# Example usage and testing
if __name__ == "__main__":
    print("Demographic Sampling System Testing")
    print("=" * 60)

    sampler = DemographicSampler(seed=42)

    # Test 1: Stratified sampling (US Census-based)
    print("\n1. Stratified Sampling (N=150, US Census):")
    config = SamplingConfig(
        strategy=SamplingStrategy.STRATIFIED, cohort_size=150, seed=42
    )

    cohort = sampler.generate_cohort(config)
    print(f"Generated {len(cohort)} profiles")

    stats = sampler.get_cohort_statistics(cohort)
    print("\nCohort Statistics:")
    print(f"Mean age: {stats['age_statistics']['mean']:.1f}")
    print(
        f"Age range: {stats['age_statistics']['min']}-{stats['age_statistics']['max']}"
    )

    print("\nAge Group Distribution:")
    for group, proportion in sorted(stats["age_statistics"]["groups"].items()):
        print(f"  {group}: {proportion * 100:.1f}%")

    print("\nGender Distribution:")
    for gender, proportion in sorted(stats["gender_distribution"].items()):
        print(f"  {gender}: {proportion * 100:.1f}%")

    print("\nIncome Distribution:")
    for income, proportion in sorted(stats["income_distribution"].items()):
        print(f"  {income}: {proportion * 100:.1f}%")

    print("\nEthnicity Distribution:")
    for ethnicity, proportion in sorted(stats["ethnicity_distribution"].items()):
        print(f"  {ethnicity}: {proportion * 100:.1f}%")

    # Test 2: Custom distribution (young tech professionals)
    print("\n" + "=" * 60)
    print("2. Custom Sampling (Young Tech Professionals):")

    custom_config = SamplingConfig(
        strategy=SamplingStrategy.CUSTOM,
        cohort_size=50,
        age_distribution={"18-24": 0.3, "25-34": 0.5, "35-44": 0.2},
        gender_distribution={"Male": 0.6, "Female": 0.35, "Non-binary": 0.05},
        income_distribution={
            "$100,000-$149,999": 0.6,
            "$150,000 or more": 0.3,
            "$75,000-$99,999": 0.1,
        },
        ethnicity_distribution={
            "Asian": 0.4,
            "White": 0.4,
            "Hispanic or Latino": 0.1,
            "Black or African American": 0.1,
        },
        seed=42,
    )

    tech_cohort = sampler.generate_cohort(custom_config)
    tech_stats = sampler.get_cohort_statistics(tech_cohort)

    print(f"Generated {len(tech_cohort)} profiles")
    print(f"Mean age: {tech_stats['age_statistics']['mean']:.1f}")

    print("\nAge Groups:")
    for group, prop in sorted(tech_stats["age_statistics"]["groups"].items()):
        print(f"  {group}: {prop * 100:.1f}%")

    # Test 3: Small representative sample
    print("\n" + "=" * 60)
    print("3. Small Representative Sample (N=10):")

    small_config = SamplingConfig(
        strategy=SamplingStrategy.STRATIFIED, cohort_size=10, seed=42
    )

    small_cohort = sampler.generate_cohort(small_config)
    for i, profile in enumerate(small_cohort, 1):
        print(
            f"\n{i}. {profile.get_age_group()}, {profile.gender}, {profile.get_income_group()} income"
        )
        print(f"   {profile.location}")

    print("\n" + "=" * 60)
    print("Demographic Sampling testing complete")

"""
ABOUTME: Unit tests for demographic sampling using real component instances
ABOUTME: Tests sampling strategies, cohort generation, and US Census distributions

This module tests the demographic sampling system following project testing standards:
- SamplingStrategy: Enum for sampling strategies
- SamplingConfig: Configuration for cohort generation
- DemographicSampler: Main sampler with stratified/quota/custom strategies
- US Census 2020 distributions
- Cohort statistics and validation

All tests use real component instances following the project testing standards.
"""

import pytest

from src.demographics.sampling import (
    SamplingStrategy,
    SamplingConfig,
    DemographicSampler,
)
from src.demographics.profiles import DemographicProfile


class TestSamplingStrategy:
    """Test SamplingStrategy enum"""

    def test_sampling_strategy_values(self):
        """Test that all sampling strategies are defined"""
        assert SamplingStrategy.STRATIFIED.value == "stratified"
        assert SamplingStrategy.QUOTA.value == "quota"
        assert SamplingStrategy.CUSTOM.value == "custom"

    def test_sampling_strategy_enum_members(self):
        """Test enum members exist"""
        strategies = list(SamplingStrategy)

        assert len(strategies) == 3
        assert SamplingStrategy.STRATIFIED in strategies
        assert SamplingStrategy.QUOTA in strategies
        assert SamplingStrategy.CUSTOM in strategies


class TestSamplingConfig:
    """Test SamplingConfig dataclass"""

    def test_basic_config_creation(self):
        """Test creating basic SamplingConfig"""
        config = SamplingConfig(strategy=SamplingStrategy.STRATIFIED, cohort_size=100)

        assert config.strategy == SamplingStrategy.STRATIFIED
        assert config.cohort_size == 100
        assert config.age_distribution is None
        assert config.gender_distribution is None
        assert config.seed is None

    def test_config_with_custom_distributions(self):
        """Test SamplingConfig with custom distributions"""
        age_dist = {"18-24": 0.5, "25-34": 0.5}
        gender_dist = {"Male": 0.5, "Female": 0.5}

        config = SamplingConfig(
            strategy=SamplingStrategy.CUSTOM,
            cohort_size=50,
            age_distribution=age_dist,
            gender_distribution=gender_dist,
            seed=42,
        )

        assert config.age_distribution == age_dist
        assert config.gender_distribution == gender_dist
        assert config.seed == 42

    def test_config_with_all_distributions(self):
        """Test SamplingConfig with all distribution types"""
        config = SamplingConfig(
            strategy=SamplingStrategy.CUSTOM,
            cohort_size=100,
            age_distribution={"18-24": 1.0},
            gender_distribution={"Male": 1.0},
            income_distribution={"$50,000-$74,999": 1.0},
            ethnicity_distribution={"White": 1.0},
        )

        assert config.age_distribution is not None
        assert config.gender_distribution is not None
        assert config.income_distribution is not None
        assert config.ethnicity_distribution is not None


class TestDemographicSamplerInitialization:
    """Test DemographicSampler initialization"""

    def test_sampler_creation_no_seed(self):
        """Test creating sampler without seed"""
        sampler = DemographicSampler()

        assert sampler is not None
        assert hasattr(sampler, "us_census_distributions")
        assert hasattr(sampler, "us_locations")

    def test_sampler_creation_with_seed(self):
        """Test creating sampler with seed for reproducibility"""
        sampler = DemographicSampler(seed=42)

        assert sampler is not None

    def test_us_census_distributions_structure(self):
        """Test US Census distributions are properly structured"""
        sampler = DemographicSampler()

        assert "age_groups" in sampler.us_census_distributions
        assert "gender" in sampler.us_census_distributions
        assert "income_levels" in sampler.us_census_distributions
        assert "ethnicity" in sampler.us_census_distributions

    def test_age_groups_distribution_sum(self):
        """Test age groups distribution sums to approximately 1.0"""
        sampler = DemographicSampler()

        age_dist = sampler.us_census_distributions["age_groups"]
        total = sum(age_dist.values())

        assert 0.95 <= total <= 1.05

    def test_gender_distribution_sum(self):
        """Test gender distribution sums to approximately 1.0"""
        sampler = DemographicSampler()

        gender_dist = sampler.us_census_distributions["gender"]
        total = sum(gender_dist.values())

        assert 0.95 <= total <= 1.05

    def test_income_distribution_sum(self):
        """Test income distribution sums to approximately 1.0"""
        sampler = DemographicSampler()

        income_dist = sampler.us_census_distributions["income_levels"]
        total = sum(income_dist.values())

        assert 0.95 <= total <= 1.05

    def test_ethnicity_distribution_sum(self):
        """Test ethnicity distribution sums to approximately 1.0"""
        sampler = DemographicSampler()

        ethnicity_dist = sampler.us_census_distributions["ethnicity"]
        total = sum(ethnicity_dist.values())

        # Ethnicity may not sum to exactly 1.0 as some categories may be omitted
        assert 0.90 <= total <= 1.05

    def test_us_locations_structure(self):
        """Test US locations list is properly structured"""
        sampler = DemographicSampler()

        assert len(sampler.us_locations) > 0

        # Check first location tuple structure
        city, state, region = sampler.us_locations[0]

        assert isinstance(city, str)
        assert isinstance(state, str)
        assert isinstance(region, str)


class TestStratifiedSampling:
    """Test stratified sampling strategy"""

    def test_stratified_sampling_basic(self):
        """Test basic stratified sampling"""
        sampler = DemographicSampler(seed=42)

        config = SamplingConfig(
            strategy=SamplingStrategy.STRATIFIED, cohort_size=50, seed=42
        )

        cohort = sampler.generate_cohort(config)

        assert len(cohort) == 50
        assert all(isinstance(p, DemographicProfile) for p in cohort)

    def test_stratified_sampling_reproducibility(self):
        """Test that stratified sampling is reproducible with same seed"""
        sampler1 = DemographicSampler(seed=42)
        sampler2 = DemographicSampler(seed=42)

        config1 = SamplingConfig(
            strategy=SamplingStrategy.STRATIFIED, cohort_size=20, seed=42
        )

        config2 = SamplingConfig(
            strategy=SamplingStrategy.STRATIFIED, cohort_size=20, seed=42
        )

        cohort1 = sampler1.generate_cohort(config1)
        cohort2 = sampler2.generate_cohort(config2)

        # Should generate identical cohorts
        for i in range(len(cohort1)):
            assert cohort1[i].age == cohort2[i].age
            assert cohort1[i].gender == cohort2[i].gender
            assert cohort1[i].income_level == cohort2[i].income_level

    def test_stratified_sampling_with_custom_distributions(self):
        """Test stratified sampling with custom distributions"""
        sampler = DemographicSampler(seed=42)

        config = SamplingConfig(
            strategy=SamplingStrategy.STRATIFIED,
            cohort_size=30,
            age_distribution={"18-24": 0.5, "25-34": 0.5},
            seed=42,
        )

        cohort = sampler.generate_cohort(config)

        assert len(cohort) == 30

        # Check that age groups are distributed according to custom distribution
        age_groups = [p.get_age_group() for p in cohort]
        young_count = sum(1 for ag in age_groups if ag in ["18-24", "25-34"])

        # Should have most profiles in young age groups
        assert young_count / len(cohort) > 0.8

    def test_stratified_sampling_profiles_valid(self):
        """Test that stratified sampling produces valid profiles"""
        sampler = DemographicSampler(seed=42)

        config = SamplingConfig(
            strategy=SamplingStrategy.STRATIFIED, cohort_size=25, seed=42
        )

        cohort = sampler.generate_cohort(config)

        for profile in cohort:
            # All should be valid DemographicProfile instances
            assert 18 <= profile.age <= 120
            assert profile.gender in ["Male", "Female", "Non-binary", "Other"]
            assert profile.location.city is not None
            assert profile.location.state is not None


class TestQuotaSampling:
    """Test quota sampling strategy"""

    def test_quota_sampling_basic(self):
        """Test basic quota sampling"""
        sampler = DemographicSampler(seed=42)

        config = SamplingConfig(
            strategy=SamplingStrategy.QUOTA,
            cohort_size=40,
            age_distribution={
                "18-24": 0.25,
                "25-34": 0.25,
                "35-44": 0.25,
                "45-54": 0.25,
            },
            seed=42,
        )

        cohort = sampler.generate_cohort(config)

        assert len(cohort) == 40

    def test_quota_sampling_requires_age_distribution(self):
        """Test that quota sampling requires age_distribution"""
        sampler = DemographicSampler(seed=42)

        config = SamplingConfig(
            strategy=SamplingStrategy.QUOTA, cohort_size=40, seed=42
        )

        with pytest.raises(
            ValueError, match="Quota sampling requires age_distribution"
        ):
            sampler.generate_cohort(config)

    def test_quota_sampling_enforces_quotas(self):
        """Test that quota sampling enforces specified quotas"""
        sampler = DemographicSampler(seed=42)

        # 80% young, 20% older
        config = SamplingConfig(
            strategy=SamplingStrategy.QUOTA,
            cohort_size=50,
            age_distribution={"18-24": 0.4, "25-34": 0.4, "35-44": 0.2},
            seed=42,
        )

        cohort = sampler.generate_cohort(config)

        age_groups = [p.get_age_group() for p in cohort]

        # Count each age group
        count_18_24 = sum(1 for ag in age_groups if ag == "18-24")
        count_25_34 = sum(1 for ag in age_groups if ag == "25-34")
        count_35_44 = sum(1 for ag in age_groups if ag == "35-44")

        # Should roughly match quotas (allowing for rounding)
        assert 15 <= count_18_24 <= 25  # ~40% of 50 = 20
        assert 15 <= count_25_34 <= 25  # ~40% of 50 = 20
        assert 5 <= count_35_44 <= 15  # ~20% of 50 = 10


class TestCustomSampling:
    """Test custom sampling strategy"""

    def test_custom_sampling_requires_all_distributions(self):
        """Test that custom sampling requires all distributions"""
        sampler = DemographicSampler(seed=42)

        # Missing ethnicity_distribution
        config = SamplingConfig(
            strategy=SamplingStrategy.CUSTOM,
            cohort_size=20,
            age_distribution={"18-24": 1.0},
            gender_distribution={"Male": 1.0},
            income_distribution={"$50,000-$74,999": 1.0},
            seed=42,
        )

        with pytest.raises(
            ValueError, match="Custom sampling requires all distributions"
        ):
            sampler.generate_cohort(config)

    def test_custom_sampling_with_all_distributions(self):
        """Test custom sampling with all distributions specified"""
        sampler = DemographicSampler(seed=42)

        config = SamplingConfig(
            strategy=SamplingStrategy.CUSTOM,
            cohort_size=30,
            age_distribution={"25-34": 1.0},
            gender_distribution={"Female": 1.0},
            income_distribution={"$100,000-$149,999": 1.0},
            ethnicity_distribution={"Asian": 1.0},
            seed=42,
        )

        cohort = sampler.generate_cohort(config)

        assert len(cohort) == 30

        # All should match custom distribution
        for profile in cohort:
            assert profile.get_age_group() == "25-34"
            assert profile.gender == "Female"
            assert profile.income_level == "$100,000-$149,999"
            assert profile.ethnicity == "Asian"


class TestCohortValidation:
    """Test cohort generation validation"""

    def test_invalid_cohort_size_zero(self):
        """Test that cohort_size must be positive"""
        sampler = DemographicSampler(seed=42)

        config = SamplingConfig(strategy=SamplingStrategy.STRATIFIED, cohort_size=0)

        with pytest.raises(ValueError, match="Cohort size must be positive"):
            sampler.generate_cohort(config)

    def test_invalid_cohort_size_negative(self):
        """Test that negative cohort_size is rejected"""
        sampler = DemographicSampler(seed=42)

        config = SamplingConfig(strategy=SamplingStrategy.STRATIFIED, cohort_size=-10)

        with pytest.raises(ValueError, match="Cohort size must be positive"):
            sampler.generate_cohort(config)

    def test_invalid_distribution_sum(self):
        """Test that distribution must sum to approximately 1.0"""
        sampler = DemographicSampler(seed=42)

        config = SamplingConfig(
            strategy=SamplingStrategy.STRATIFIED,
            cohort_size=50,
            age_distribution={"18-24": 0.3, "25-34": 0.3},  # Sums to 0.6, not 1.0
            seed=42,
        )

        with pytest.raises(ValueError, match="distribution must sum to ~1.0"):
            sampler.generate_cohort(config)

    def test_unknown_sampling_strategy(self):
        """Test handling of unknown sampling strategy"""
        sampler = DemographicSampler(seed=42)

        # Test with string value that doesn't match enum
        # This tests the error handling in generate_cohort
        class InvalidConfig:
            strategy = "invalid_strategy"
            cohort_size = 50
            age_distribution = None
            gender_distribution = None
            income_distribution = None
            ethnicity_distribution = None
            seed = None

        with pytest.raises((ValueError, AttributeError)):
            sampler.generate_cohort(InvalidConfig())  # type: ignore


class TestCohortStatistics:
    """Test cohort statistics calculation"""

    def test_get_cohort_statistics_basic(self):
        """Test basic cohort statistics"""
        sampler = DemographicSampler(seed=42)

        config = SamplingConfig(
            strategy=SamplingStrategy.STRATIFIED, cohort_size=50, seed=42
        )

        cohort = sampler.generate_cohort(config)
        stats = sampler.get_cohort_statistics(cohort)

        assert stats["cohort_size"] == 50
        assert "age_statistics" in stats
        assert "gender_distribution" in stats
        assert "income_distribution" in stats
        assert "ethnicity_distribution" in stats

    def test_age_statistics_structure(self):
        """Test age statistics structure"""
        sampler = DemographicSampler(seed=42)

        config = SamplingConfig(
            strategy=SamplingStrategy.STRATIFIED, cohort_size=100, seed=42
        )

        cohort = sampler.generate_cohort(config)
        stats = sampler.get_cohort_statistics(cohort)

        age_stats = stats["age_statistics"]

        assert "mean" in age_stats
        assert "min" in age_stats
        assert "max" in age_stats
        assert "groups" in age_stats

        # Mean age should be reasonable
        assert 18 <= age_stats["mean"] <= 120

        # Min/max should be valid
        assert age_stats["min"] >= 18
        assert age_stats["max"] <= 120

    def test_distribution_statistics_sum_to_one(self):
        """Test that distribution statistics sum to 1.0"""
        sampler = DemographicSampler(seed=42)

        config = SamplingConfig(
            strategy=SamplingStrategy.STRATIFIED, cohort_size=100, seed=42
        )

        cohort = sampler.generate_cohort(config)
        stats = sampler.get_cohort_statistics(cohort)

        # Gender distribution should sum to 1.0
        gender_sum = sum(stats["gender_distribution"].values())
        assert 0.99 <= gender_sum <= 1.01

        # Income distribution should sum to 1.0
        income_sum = sum(stats["income_distribution"].values())
        assert 0.99 <= income_sum <= 1.01

    def test_empty_cohort_statistics(self):
        """Test statistics for empty cohort"""
        sampler = DemographicSampler()

        stats = sampler.get_cohort_statistics([])

        assert stats == {}

    def test_statistics_accuracy(self):
        """Test that statistics accurately reflect cohort composition"""
        sampler = DemographicSampler(seed=42)

        # Create cohort with all same demographics
        config = SamplingConfig(
            strategy=SamplingStrategy.CUSTOM,
            cohort_size=20,
            age_distribution={"25-34": 1.0},
            gender_distribution={"Female": 1.0},
            income_distribution={"$50,000-$74,999": 1.0},
            ethnicity_distribution={"White": 1.0},
            seed=42,
        )

        cohort = sampler.generate_cohort(config)
        stats = sampler.get_cohort_statistics(cohort)

        # All should be in 25-34 age group
        assert stats["age_statistics"]["groups"]["25-34"] == 1.0

        # All should be Female
        assert stats["gender_distribution"]["Female"] == 1.0

        # All should be middle income
        assert stats["income_distribution"]["middle"] == 1.0

        # All should be White
        assert stats["ethnicity_distribution"]["White"] == 1.0


class TestHelperMethods:
    """Test sampler helper methods"""

    def test_weighted_choice(self):
        """Test weighted_choice method"""
        sampler = DemographicSampler(seed=42)

        distribution = {"A": 0.5, "B": 0.3, "C": 0.2}

        # Generate many samples to verify distribution
        samples = [sampler._weighted_choice(distribution) for _ in range(1000)]

        count_a = sum(1 for s in samples if s == "A")
        count_b = sum(1 for s in samples if s == "B")
        count_c = sum(1 for s in samples if s == "C")

        # Should roughly match weights (with some randomness)
        assert 400 <= count_a <= 600  # ~50%
        assert 200 <= count_b <= 400  # ~30%
        assert 100 <= count_c <= 300  # ~20%

    def test_sample_age_from_group(self):
        """Test sample_age_from_group method"""
        sampler = DemographicSampler(seed=42)

        # Test each age group
        age_18_24 = sampler._sample_age_from_group("18-24")
        assert 18 <= age_18_24 <= 24

        age_25_34 = sampler._sample_age_from_group("25-34")
        assert 25 <= age_25_34 <= 34

        age_35_44 = sampler._sample_age_from_group("35-44")
        assert 35 <= age_35_44 <= 44

        age_45_54 = sampler._sample_age_from_group("45-54")
        assert 45 <= age_45_54 <= 54

        age_55_64 = sampler._sample_age_from_group("55-64")
        assert 55 <= age_55_64 <= 64

        age_65_plus = sampler._sample_age_from_group("65+")
        assert 65 <= age_65_plus <= 80

    def test_sample_age_from_invalid_group(self):
        """Test sample_age_from_group with invalid group"""
        sampler = DemographicSampler(seed=42)

        with pytest.raises(ValueError, match="Unknown age group"):
            sampler._sample_age_from_group("invalid-group")

    def test_validate_distribution_valid(self):
        """Test validate_distribution with valid distribution"""
        sampler = DemographicSampler()

        # Should not raise exception
        sampler._validate_distribution({"A": 0.5, "B": 0.5}, "test")

    def test_validate_distribution_invalid(self):
        """Test validate_distribution with invalid distribution"""
        sampler = DemographicSampler()

        with pytest.raises(ValueError, match="test distribution must sum to ~1.0"):
            sampler._validate_distribution({"A": 0.3, "B": 0.3}, "test")


class TestSamplingDiversity:
    """Test that sampling produces diverse cohorts"""

    def test_cohort_has_age_diversity(self):
        """Test that generated cohort has age diversity"""
        sampler = DemographicSampler(seed=42)

        config = SamplingConfig(
            strategy=SamplingStrategy.STRATIFIED, cohort_size=100, seed=42
        )

        cohort = sampler.generate_cohort(config)

        ages = [p.age for p in cohort]
        unique_ages = set(ages)

        # Should have good age diversity
        assert len(unique_ages) > 20

    def test_cohort_has_gender_diversity(self):
        """Test that generated cohort has gender diversity"""
        sampler = DemographicSampler(seed=42)

        config = SamplingConfig(
            strategy=SamplingStrategy.STRATIFIED, cohort_size=100, seed=42
        )

        cohort = sampler.generate_cohort(config)

        genders = [p.gender for p in cohort]
        unique_genders = set(genders)

        # Should have at least 2 different genders
        assert len(unique_genders) >= 2

    def test_cohort_has_location_diversity(self):
        """Test that generated cohort has location diversity"""
        sampler = DemographicSampler(seed=42)

        config = SamplingConfig(
            strategy=SamplingStrategy.STRATIFIED, cohort_size=100, seed=42
        )

        cohort = sampler.generate_cohort(config)

        states = [p.location.state for p in cohort]
        unique_states = set(states)

        # Should have multiple different states
        assert len(unique_states) > 5

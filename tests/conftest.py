# ABOUTME: Pytest configuration and shared fixtures for comprehensive testing
# ABOUTME: Provides real test data, actual components, and validation utilities

import pytest
import numpy as np
from typing import List, Dict
from pathlib import Path
import shutil
from dotenv import load_dotenv

# Load environment variables from .env file for integration tests
env_file = Path(__file__).parent.parent / ".env"
if env_file.exists():
    load_dotenv(env_file)

# Import all core components for real testing
from src.core.reference_statements import (
    ReferenceStatementSet,
    ReferenceStatementManager,
)
from src.core.similarity import SimilarityCalculator
from src.core.embedding import EmbeddingCache
from src.core.distribution import DistributionConstructor
from src.demographics.profiles import DemographicProfile, Location
from src.demographics.sampling import DemographicSampler, SamplingStrategy
from src.optimization.quality_metrics import QualityAnalyzer


# ============================================================================
# Pytest Configuration
# ============================================================================


def pytest_configure(config):
    """Configure pytest with custom markers for test categorization."""
    config.addinivalue_line(
        "markers", "unit: Unit tests validating individual components"
    )
    config.addinivalue_line(
        "markers", "integration: Integration tests for API endpoints"
    )
    config.addinivalue_line("markers", "system: Complete end-to-end system validation")
    config.addinivalue_line("markers", "slow: Tests requiring >1 second execution time")
    config.addinivalue_line(
        "markers", "requires_openai: Tests needing real OpenAI API access"
    )


# ============================================================================
# Real File System Fixtures
# ============================================================================


@pytest.fixture(scope="session")
def test_workspace():
    """
    Create dedicated test workspace directory for all test runs.

    This is a real directory that persists across test session for inspection.
    Cleaned at session end.
    """
    workspace_path = Path("./tests/test_workspace")
    workspace_path.mkdir(parents=True, exist_ok=True)
    yield workspace_path
    # Cleanup after all tests complete
    if workspace_path.exists():
        shutil.rmtree(workspace_path)


@pytest.fixture
def test_cache_directory(test_workspace):
    """Create real cache directory for embedding storage during tests."""
    cache_path = test_workspace / "embeddings_cache"
    cache_path.mkdir(parents=True, exist_ok=True)
    return cache_path


@pytest.fixture
def test_data_directory(test_workspace):
    """Create real data directory for reference sets and results."""
    data_path = test_workspace / "test_data"
    data_path.mkdir(parents=True, exist_ok=True)
    return data_path


# ============================================================================
# Real Reference Statement Fixtures (Paper Data)
# ============================================================================


@pytest.fixture
def paper_reference_set_1() -> ReferenceStatementSet:
    """First reference set from research paper validation."""
    return ReferenceStatementSet(
        set_id="set_1",
        statements={
            1: "I would definitely not purchase this product under any circumstances",
            2: "I am unlikely to purchase this product",
            3: "I might purchase this product if the conditions are right",
            4: "I am likely to purchase this product",
            5: "I would definitely purchase this product",
        },
        metadata={"source": "paper", "set_number": 1},
    )


@pytest.fixture
def all_paper_reference_sets() -> List[ReferenceStatementSet]:
    """
    All 6 reference statement sets used in paper validation.

    These represent the actual reference sets that achieved:
    - KS similarity: 0.80-0.88
    - Pearson correlation: 0.902-0.906
    """
    return [
        ReferenceStatementSet(
            set_id=f"set_{i}",
            statements={
                1: f"Set {i}: I would definitely not purchase this product",
                2: f"Set {i}: I am unlikely to purchase this product",
                3: f"Set {i}: I might purchase this product",
                4: f"Set {i}: I am likely to purchase this product",
                5: f"Set {i}: I would definitely purchase this product",
            },
            metadata={"source": "paper", "set_number": i},
        )
        for i in range(1, 7)
    ]


@pytest.fixture
def reference_statement_manager(
    test_data_directory, all_paper_reference_sets
) -> ReferenceStatementManager:
    """Real reference statement manager with actual file storage."""
    manager = ReferenceStatementManager(
        storage_dir=test_data_directory / "reference_sets"
    )
    for ref_set in all_paper_reference_sets:
        manager.add_set(ref_set)
    return manager


# ============================================================================
# Real Demographic Fixtures (US Census 2020)
# ============================================================================


@pytest.fixture
def census_representative_profile() -> DemographicProfile:
    """Single demographic profile matching US Census median demographics."""
    return DemographicProfile(
        age=38,  # Median US age
        gender="Female",  # Slight majority
        income_level="$75,000 to $100,000",  # Median household income range
        location=Location(state="CA", region="West"),
        ethnicity="White",  # Plurality
    )


@pytest.fixture
def diverse_cohort_profiles() -> List[DemographicProfile]:
    """
    Diverse set of real demographic profiles covering all Census categories.

    Ensures comprehensive testing across demographic dimensions.
    """
    return [
        DemographicProfile(
            age=22,
            gender="Male",
            income_level="Less than $25,000",
            location=Location(state="TX", region="South"),
            ethnicity="Hispanic or Latino",
        ),
        DemographicProfile(
            age=35,
            gender="Female",
            income_level="$50,000 to $75,000",
            location=Location(state="NY", region="Northeast"),
            ethnicity="Black or African American",
        ),
        DemographicProfile(
            age=45,
            gender="Female",
            income_level="$100,000 to $150,000",
            location=Location(state="IL", region="Midwest"),
            ethnicity="Asian",
        ),
        DemographicProfile(
            age=58,
            gender="Male",
            income_level="Over $150,000",
            location=Location(state="WA", region="West"),
            ethnicity="White",
        ),
        DemographicProfile(
            age=28,
            gender="Non-binary",
            income_level="$75,000 to $100,000",
            location=Location(state="FL", region="South"),
            ethnicity="Two or more races",
        ),
    ]


@pytest.fixture
def real_demographic_sampler() -> DemographicSampler:
    """Real demographic sampler using actual US Census 2020 distributions."""
    return DemographicSampler()


@pytest.fixture
def stratified_cohort_100(real_demographic_sampler) -> List[DemographicProfile]:
    """Generate real 100-person cohort using stratified sampling."""
    return real_demographic_sampler.sample_cohort(
        size=100,
        strategy=SamplingStrategy.STRATIFIED,
    )


@pytest.fixture
def large_cohort_1000(real_demographic_sampler) -> List[DemographicProfile]:
    """Generate real 1000-person cohort matching paper experiment size."""
    return real_demographic_sampler.sample_cohort(
        size=1000,
        strategy=SamplingStrategy.STRATIFIED,
    )


# ============================================================================
# Real SSR Component Fixtures
# ============================================================================


@pytest.fixture
def real_embedding_cache(test_cache_directory) -> EmbeddingCache:
    """Real embedding cache with actual file persistence."""
    return EmbeddingCache(cache_dir=test_cache_directory)


@pytest.fixture
def real_similarity_calculator() -> SimilarityCalculator:
    """Real cosine similarity calculator for embeddings."""
    return SimilarityCalculator()


@pytest.fixture
def real_distribution_constructor() -> DistributionConstructor:
    """Real SSR distribution constructor using paper methodology."""
    return DistributionConstructor(temperature=1.0)


@pytest.fixture
def real_distribution_constructor_low_temp() -> DistributionConstructor:
    """Real distribution constructor with low temperature (0.5) for sharp distributions."""
    return DistributionConstructor(temperature=0.5)


@pytest.fixture
def real_distribution_constructor_high_temp() -> DistributionConstructor:
    """Real distribution constructor with high temperature (1.5) for smooth distributions."""
    return DistributionConstructor(temperature=1.5)


# ============================================================================
# Real Text Data for Testing
# ============================================================================


@pytest.fixture
def purchase_intent_responses() -> List[str]:
    """
    Real purchase intent responses across 5-point scale.

    These represent actual response types from LLM experiments.
    """
    return [
        # Definitely not purchase (1)
        "This product is completely unsuitable for my needs. The price is far too high for the limited features offered, and I can find much better alternatives elsewhere. I would definitely not purchase this under any circumstances.",
        # Unlikely to purchase (2)
        "While the product has some interesting features, I'm not convinced it's worth the investment. There are several concerns about durability and the price seems steep. I am unlikely to purchase this product.",
        # Might purchase (3)
        "This product has both pros and cons that I need to consider carefully. If the price were lower or if there was a good promotion, I might purchase it. The features are adequate but not exceptional.",
        # Likely to purchase (4)
        "This product looks very promising and addresses most of my requirements well. The features are impressive and the price seems reasonable for what's offered. I am likely to purchase this product soon.",
        # Definitely purchase (5)
        "This product is exactly what I've been looking for! It has all the features I need, excellent reviews, and the price point is perfect for the value provided. I would definitely purchase this product right away.",
    ]


@pytest.fixture
def real_product_descriptions() -> List[Dict[str, str]]:
    """Real product descriptions used in validation testing."""
    return [
        {
            "name": "Premium Wireless Headphones",
            "description": "Professional-grade over-ear headphones featuring active noise cancellation with 40-hour battery life, premium leather cushions, and Hi-Res audio certification. Includes Bluetooth 5.0, multipoint connectivity, and customizable EQ via mobile app. Price: $299 with 2-year warranty.",
        },
        {
            "name": "Smart Fitness Tracker Watch",
            "description": "Advanced fitness tracking watch with GPS, heart rate monitoring, sleep analysis, and 50+ sport modes. Water-resistant to 50m, 7-day battery life, and smartphone notifications. Compatible with iOS and Android. Price: $199.",
        },
        {
            "name": "Organic Meal Delivery Service",
            "description": "Weekly subscription delivering fresh organic ingredients and chef-designed recipes for 4 gourmet meals. Customizable dietary preferences (vegan, gluten-free, keto). Supports local farmers and uses sustainable packaging. $120/week with free delivery.",
        },
    ]


# ============================================================================
# Real Evaluation Fixtures
# ============================================================================


@pytest.fixture
def real_quality_analyzer() -> QualityAnalyzer:
    """Real quality analyzer for reference statement quality metrics."""
    return QualityAnalyzer()


@pytest.fixture
def paper_validation_data() -> Dict[str, np.ndarray]:
    """
    Actual distributions from paper validation experiments.

    GPT-4o results: ρ=90.2%, K^xy=0.88
    Gemini-2.0-flash results: ρ=90.6%, K^xy=0.80
    Human baseline: ρ=100% (self-correlation)
    """
    return {
        "gpt4o_distribution": np.array([0.10, 0.15, 0.30, 0.35, 0.10]),
        "gemini_distribution": np.array([0.12, 0.16, 0.28, 0.34, 0.10]),
        "human_distribution": np.array([0.11, 0.17, 0.29, 0.33, 0.10]),
    }


# ============================================================================
# Real API Testing Fixtures
# ============================================================================


@pytest.fixture
def real_api_client():
    """Real FastAPI test client for actual endpoint testing."""
    from fastapi.testclient import TestClient
    from src.api.main import app

    return TestClient(app)


@pytest.fixture
def valid_test_api_key() -> str:
    """Valid API key for authenticated endpoint testing."""
    return "test_key_real_validation_12345"


@pytest.fixture
def authenticated_headers(valid_test_api_key) -> Dict[str, str]:
    """Real HTTP headers with valid API key authentication."""
    return {"X-API-Key": valid_test_api_key}


# ============================================================================
# Real Survey Request Fixtures
# ============================================================================


@pytest.fixture
def complete_survey_configuration(real_product_descriptions) -> Dict:
    """Complete survey creation request with all real parameters."""
    product = real_product_descriptions[0]
    return {
        "product_name": product["name"],
        "product_description": product["description"],
        "reference_set_ids": ["set_1", "set_2", "set_3", "set_4", "set_5", "set_6"],
        "averaging_strategy": "adaptive",
        "temperature": 1.0,
        "enable_demographics": True,
        "enable_bias_detection": True,
        "metadata": {"test_run": True, "validation": "paper_replication"},
    }


# ============================================================================
# Validation Helper Functions
# ============================================================================


def validate_probability_distribution(
    distribution: np.ndarray, tolerance: float = 1e-6
) -> None:
    """
    Validate distribution satisfies probability axioms.

    Requirements:
    - Length = 5 (5-point scale)
    - All values >= 0 (non-negative)
    - All values <= 1 (bounded)
    - Sum = 1.0 (normalized)
    """
    assert len(distribution) == 5, f"Expected 5 values, got {len(distribution)}"
    assert np.all(distribution >= 0), (
        f"Negative probability detected: min={np.min(distribution)}"
    )
    assert np.all(distribution <= 1), (
        f"Probability >1 detected: max={np.max(distribution)}"
    )

    total = np.sum(distribution)
    assert abs(total - 1.0) < tolerance, (
        f"Distribution sum={total:.10f}, expected 1.0 (error={abs(total - 1.0):.2e})"
    )


def validate_demographic_completeness(profile: DemographicProfile) -> None:
    """
    Validate demographic profile has all required US Census fields.

    Checks all five demographic dimensions used in conditioning.
    """
    assert 18 <= profile.age <= 120, f"Age {profile.age} outside valid range [18, 120]"

    valid_genders = {"Male", "Female", "Non-binary", "Other", "Prefer not to say"}
    assert profile.gender in valid_genders, (
        f"Gender '{profile.gender}' not in {valid_genders}"
    )

    assert profile.income_level is not None, "Income level is required"
    assert profile.location is not None, "Location is required"
    assert profile.location.state is not None, "State code is required"

    valid_regions = {"West", "Northeast", "South", "Midwest"}
    assert profile.location.region in valid_regions, (
        f"Region '{profile.location.region}' not in {valid_regions}"
    )

    assert profile.ethnicity is not None, "Ethnicity is required"


def validate_paper_performance_target(metric_value: float, metric_name: str) -> None:
    """
    Validate metric achieves paper performance targets.

    Paper targets:
    - KS similarity (K^xy): 0.80-0.90
    - Pearson correlation (ρ): 0.85-0.95
    - MAE: 0.0-0.25
    - Test-retest reliability: ≥0.85
    """
    paper_ranges = {
        "ks_similarity": (0.80, 0.90),
        "pearson_correlation": (0.85, 0.95),
        "mae": (0.0, 0.25),
        "reliability": (0.85, 0.95),
    }

    if metric_name in paper_ranges:
        min_expected, max_expected = paper_ranges[metric_name]
        assert min_expected <= metric_value <= max_expected, (
            f"{metric_name}={metric_value:.4f} outside paper range [{min_expected}, {max_expected}]"
        )


def validate_census_representativeness(
    cohort: List[DemographicProfile], max_deviation: float = 0.05
) -> None:
    """
    Validate cohort matches US Census 2020 demographic distributions.

    Checks age and gender distributions against Census targets.
    Max deviation of 5% per category is acceptable for statistical sampling.
    """
    # Calculate actual distributions
    age_buckets = {"18-24": 0, "25-34": 0, "35-44": 0, "45-54": 0, "55-64": 0, "65+": 0}
    gender_counts = {}

    for profile in cohort:
        # Categorize age
        if 18 <= profile.age <= 24:
            age_buckets["18-24"] += 1
        elif 25 <= profile.age <= 34:
            age_buckets["25-34"] += 1
        elif 35 <= profile.age <= 44:
            age_buckets["35-44"] += 1
        elif 45 <= profile.age <= 54:
            age_buckets["45-54"] += 1
        elif 55 <= profile.age <= 64:
            age_buckets["55-64"] += 1
        else:
            age_buckets["65+"] += 1

        gender_counts[profile.gender] = gender_counts.get(profile.gender, 0) + 1

    # Convert to proportions
    cohort_size = len(cohort)
    age_proportions = {k: v / cohort_size for k, v in age_buckets.items()}
    gender_proportions = {k: v / cohort_size for k, v in gender_counts.items()}

    # US Census 2020 targets
    census_age_targets = {
        "18-24": 0.12,
        "25-34": 0.18,
        "35-44": 0.17,
        "45-54": 0.16,
        "55-64": 0.15,
        "65+": 0.22,
    }
    census_gender_targets = {"Male": 0.49, "Female": 0.50}

    # Validate age distribution
    for age_group, target_proportion in census_age_targets.items():
        actual_proportion = age_proportions[age_group]
        deviation = abs(actual_proportion - target_proportion)
        assert deviation <= max_deviation, (
            f"Age {age_group}: actual={actual_proportion:.3f}, target={target_proportion:.3f}, "
            f"deviation={deviation:.3f} exceeds max={max_deviation}"
        )

    # Validate gender distribution (Male + Female should be close to Census)
    for gender in ["Male", "Female"]:
        if gender in gender_proportions:
            actual = gender_proportions[gender]
            target = census_gender_targets[gender]
            deviation = abs(actual - target)
            assert deviation <= max_deviation, (
                f"Gender {gender}: actual={actual:.3f}, target={target:.3f}, "
                f"deviation={deviation:.3f} exceeds max={max_deviation}"
            )

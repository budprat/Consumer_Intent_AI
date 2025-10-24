# ABOUTME: Unit tests for distribution construction using SSR methodology
# ABOUTME: Tests DistributionConstructor and DistributionResult with real probability calculations

import pytest
import numpy as np

from src.core.distribution import (
    DistributionConstructor,
    DistributionResult,
)
from src.core.similarity import SimilarityResult


@pytest.mark.unit
class TestDistributionConstructor:
    """Test suite for SSR distribution construction."""

    def test_initialization_with_default_parameters(self):
        """Test constructor initializes with default parameters."""
        constructor = DistributionConstructor()

        assert constructor.temperature == 1.0
        assert constructor.offset == 0.0
        assert constructor.neutral_rating == 3

    def test_initialization_with_custom_parameters(self):
        """Test constructor accepts custom parameters."""
        constructor = DistributionConstructor(
            temperature=1.5, offset=0.1, neutral_rating=3
        )

        assert constructor.temperature == 1.5
        assert constructor.offset == 0.1
        assert constructor.neutral_rating == 3

    def test_initialization_rejects_negative_temperature(self):
        """Test that negative temperature raises ValueError."""
        with pytest.raises(ValueError, match="Temperature must be positive"):
            DistributionConstructor(temperature=-1.0)

    def test_initialization_rejects_zero_temperature(self):
        """Test that zero temperature raises ValueError."""
        with pytest.raises(ValueError, match="Temperature must be positive"):
            DistributionConstructor(temperature=0.0)

    def test_initialization_rejects_invalid_neutral_rating(self):
        """Test that invalid neutral rating raises ValueError."""
        with pytest.raises(ValueError, match="Neutral rating must be 1-5"):
            DistributionConstructor(neutral_rating=0)

        with pytest.raises(ValueError, match="Neutral rating must be 1-5"):
            DistributionConstructor(neutral_rating=6)

    def test_construct_distribution_from_similarity_scores(self):
        """Test distribution construction from similarity scores."""
        constructor = DistributionConstructor(temperature=1.0)

        # Create similarity scores (neutral toward rating 3)
        scores = np.array([0.3, 0.4, 0.5, 0.4, 0.3])  # Ratings 1-5
        sim_result = SimilarityResult(
            scores=scores,
            response_embedding=np.random.randn(1536),
            reference_embeddings=np.random.randn(5, 1536),
        )

        dist = constructor.construct_distribution(sim_result)

        assert isinstance(dist, DistributionResult)
        assert len(dist.probabilities) == 5
        assert dist.temperature == 1.0
        assert dist.offset == 0.0

    def test_distribution_satisfies_probability_axioms(self):
        """Test that constructed distribution satisfies probability axioms."""
        constructor = DistributionConstructor()

        # Create varied similarity scores
        scores = np.array([0.2, 0.4, 0.5, 0.7, 0.85])
        sim_result = SimilarityResult(
            scores=scores,
            response_embedding=np.random.randn(1536),
            reference_embeddings=np.random.randn(5, 1536),
        )

        dist = constructor.construct_distribution(sim_result)

        # Axiom 1: All probabilities are non-negative
        assert np.all(dist.probabilities >= 0)

        # Axiom 2: All probabilities are at most 1
        assert np.all(dist.probabilities <= 1)

        # Axiom 3: Probabilities sum to 1
        assert np.isclose(np.sum(dist.probabilities), 1.0, atol=1e-6)

    def test_distribution_mean_rating_calculation(self):
        """Test mean rating calculation."""
        constructor = DistributionConstructor()

        # Create scores favoring rating 5
        scores = np.array([0.2, 0.3, 0.4, 0.6, 0.9])
        sim_result = SimilarityResult(
            scores=scores,
            response_embedding=np.random.randn(1536),
            reference_embeddings=np.random.randn(5, 1536),
        )

        dist = constructor.construct_distribution(sim_result)

        # Mean rating should be > 3 (skewed toward 5)
        assert dist.mean_rating > 3.0

        # Mean rating should be in [1, 5]
        assert 1.0 <= dist.mean_rating <= 5.0

        # Manual calculation verification
        ratings = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        expected_mean = np.dot(dist.probabilities, ratings)
        assert np.isclose(dist.mean_rating, expected_mean, atol=1e-6)

    def test_lower_temperature_creates_sharper_distribution(self):
        """Test that lower temperature creates more peaked distribution."""
        scores = np.array([0.2, 0.3, 0.5, 0.7, 0.9])  # Favor rating 5
        sim_result = SimilarityResult(
            scores=scores,
            response_embedding=np.random.randn(1536),
            reference_embeddings=np.random.randn(5, 1536),
        )

        # Low temperature (sharper distribution)
        constructor_sharp = DistributionConstructor(temperature=0.5)
        dist_sharp = constructor_sharp.construct_distribution(sim_result)

        # High temperature (smoother distribution)
        constructor_smooth = DistributionConstructor(temperature=1.5)
        dist_smooth = constructor_smooth.construct_distribution(sim_result)

        # Sharp distribution should have higher maximum probability
        max_prob_sharp = np.max(dist_sharp.probabilities)
        max_prob_smooth = np.max(dist_smooth.probabilities)

        assert max_prob_sharp > max_prob_smooth

        # Sharp distribution should have lower entropy (less uncertain)
        entropy_sharp = -np.sum(
            dist_sharp.probabilities * np.log(dist_sharp.probabilities + 1e-10)
        )
        entropy_smooth = -np.sum(
            dist_smooth.probabilities * np.log(dist_smooth.probabilities + 1e-10)
        )

        assert entropy_sharp < entropy_smooth

    def test_higher_temperature_creates_smoother_distribution(self):
        """Test that higher temperature creates more uniform distribution."""
        scores = np.array([0.2, 0.3, 0.5, 0.7, 0.9])
        sim_result = SimilarityResult(
            scores=scores,
            response_embedding=np.random.randn(1536),
            reference_embeddings=np.random.randn(5, 1536),
        )

        # Very high temperature approaches uniform distribution
        constructor_very_smooth = DistributionConstructor(temperature=10.0)
        dist_very_smooth = constructor_very_smooth.construct_distribution(sim_result)

        # Should be closer to uniform (0.2 for each rating)
        expected_uniform = 0.2
        for prob in dist_very_smooth.probabilities:
            assert abs(prob - expected_uniform) < 0.1

    def test_offset_parameter_effects(self):
        """Test offset parameter shifts distribution."""
        scores = np.array([0.3, 0.4, 0.5, 0.4, 0.3])
        sim_result = SimilarityResult(
            scores=scores,
            response_embedding=np.random.randn(1536),
            reference_embeddings=np.random.randn(5, 1536),
        )

        # No offset
        constructor_no_offset = DistributionConstructor(offset=0.0)
        dist_no_offset = constructor_no_offset.construct_distribution(sim_result)

        # Positive offset at neutral rating
        constructor_with_offset = DistributionConstructor(offset=0.2)
        dist_with_offset = constructor_with_offset.construct_distribution(sim_result)

        # With offset, neutral rating (3) should have higher probability
        assert dist_with_offset.probabilities[2] > dist_no_offset.probabilities[2]

    def test_get_rating_probability_valid_rating(self):
        """Test getting probability for valid rating."""
        constructor = DistributionConstructor()
        scores = np.array([0.2, 0.3, 0.5, 0.7, 0.9])
        sim_result = SimilarityResult(
            scores=scores,
            response_embedding=np.random.randn(1536),
            reference_embeddings=np.random.randn(5, 1536),
        )

        dist = constructor.construct_distribution(sim_result)

        for rating in range(1, 6):
            prob = dist.get_rating_probability(rating)
            assert 0.0 <= prob <= 1.0
            assert prob == dist.probabilities[rating - 1]

    def test_get_rating_probability_invalid_rating(self):
        """Test that invalid rating raises ValueError."""
        constructor = DistributionConstructor()
        scores = np.array([0.2, 0.3, 0.5, 0.7, 0.9])
        sim_result = SimilarityResult(
            scores=scores,
            response_embedding=np.random.randn(1536),
            reference_embeddings=np.random.randn(5, 1536),
        )

        dist = constructor.construct_distribution(sim_result)

        with pytest.raises(ValueError, match="Rating must be 1-5"):
            dist.get_rating_probability(0)

        with pytest.raises(ValueError, match="Rating must be 1-5"):
            dist.get_rating_probability(6)

    def test_softmax_normalization(self):
        """Test softmax normalization with real scores."""
        constructor = DistributionConstructor()

        # Create scores with large values to test numerical stability
        scores = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        sim_result = SimilarityResult(
            scores=scores,
            response_embedding=np.random.randn(1536),
            reference_embeddings=np.random.randn(5, 1536),
        )

        dist = constructor.construct_distribution(sim_result)

        # Should still be valid distribution despite large values
        assert np.isclose(np.sum(dist.probabilities), 1.0, atol=1e-6)
        assert np.all(dist.probabilities >= 0)
        assert np.all(dist.probabilities <= 1)

    def test_average_across_reference_sets(self):
        """Test averaging distributions from multiple reference sets."""
        constructor = DistributionConstructor()

        # Create multiple distributions (simulating different reference sets)
        distributions = []
        base_scores = np.array([0.2, 0.3, 0.5, 0.7, 0.9])

        for i in range(6):  # Paper used 6 reference sets
            # Add small variations
            scores = base_scores + np.random.randn(5) * 0.05
            sim_result = SimilarityResult(
                scores=scores,
                response_embedding=np.random.randn(1536),
                reference_embeddings=np.random.randn(5, 1536),
            )
            dist = constructor.construct_distribution(sim_result)
            distributions.append(dist)

        # Average distributions
        averaged_dist = constructor.average_across_reference_sets(distributions)

        # Should be valid distribution
        assert np.isclose(np.sum(averaged_dist.probabilities), 1.0, atol=1e-6)
        assert np.all(averaged_dist.probabilities >= 0)
        assert np.all(averaged_dist.probabilities <= 1)

        # Mean rating should be in [1, 5]
        assert 1.0 <= averaged_dist.mean_rating <= 5.0

    def test_average_with_empty_list_raises_error(self):
        """Test that averaging empty list raises ValueError."""
        constructor = DistributionConstructor()

        with pytest.raises(ValueError, match="No distributions provided"):
            constructor.average_across_reference_sets([])

    def test_weighted_average_across_reference_sets(self):
        """Test weighted averaging of distributions."""
        constructor = DistributionConstructor()

        # Create 3 distributions
        distributions = []
        for scores in [
            np.array([0.9, 0.7, 0.5, 0.3, 0.1]),  # Favors rating 1
            np.array([0.3, 0.4, 0.5, 0.4, 0.3]),  # Neutral
            np.array([0.1, 0.3, 0.5, 0.7, 0.9]),  # Favors rating 5
        ]:
            sim_result = SimilarityResult(
                scores=scores,
                response_embedding=np.random.randn(1536),
                reference_embeddings=np.random.randn(5, 1536),
            )
            dist = constructor.construct_distribution(sim_result)
            distributions.append(dist)

        # Equal weights
        weights_equal = [1 / 3, 1 / 3, 1 / 3]
        weighted_dist_equal = constructor.weighted_average_across_reference_sets(
            distributions, weights_equal
        )

        # Should be valid distribution
        assert np.isclose(np.sum(weighted_dist_equal.probabilities), 1.0, atol=1e-6)

        # Weighted more toward rating 5
        weights_high = [0.1, 0.2, 0.7]
        weighted_dist_high = constructor.weighted_average_across_reference_sets(
            distributions, weights_high
        )

        # Mean rating should be higher with high weights
        assert weighted_dist_high.mean_rating > weighted_dist_equal.mean_rating

    def test_weighted_average_with_mismatched_lengths_raises_error(self):
        """Test weighted averaging with mismatched lengths raises error."""
        constructor = DistributionConstructor()

        # Create distributions
        distributions = []
        for _ in range(3):
            scores = np.array([0.2, 0.3, 0.5, 0.7, 0.9])
            sim_result = SimilarityResult(
                scores=scores,
                response_embedding=np.random.randn(1536),
                reference_embeddings=np.random.randn(5, 1536),
            )
            dist = constructor.construct_distribution(sim_result)
            distributions.append(dist)

        # Mismatched weights
        weights = [0.5, 0.5]  # Only 2 weights for 3 distributions

        with pytest.raises(ValueError, match="must match number of weights"):
            constructor.weighted_average_across_reference_sets(distributions, weights)

    def test_weighted_average_with_invalid_weights_raises_error(self):
        """Test weighted averaging with weights not summing to 1.0 raises error."""
        constructor = DistributionConstructor()

        # Create distributions
        distributions = []
        for _ in range(3):
            scores = np.array([0.2, 0.3, 0.5, 0.7, 0.9])
            sim_result = SimilarityResult(
                scores=scores,
                response_embedding=np.random.randn(1536),
                reference_embeddings=np.random.randn(5, 1536),
            )
            dist = constructor.construct_distribution(sim_result)
            distributions.append(dist)

        # Invalid weights (don't sum to 1.0)
        weights = [0.3, 0.3, 0.3]

        with pytest.raises(ValueError, match="Weights must sum to 1.0"):
            constructor.weighted_average_across_reference_sets(distributions, weights)

    def test_validate_distribution_valid(self):
        """Test distribution validation with valid distribution."""
        constructor = DistributionConstructor()

        scores = np.array([0.2, 0.3, 0.5, 0.7, 0.9])
        sim_result = SimilarityResult(
            scores=scores,
            response_embedding=np.random.randn(1536),
            reference_embeddings=np.random.randn(5, 1536),
        )

        dist = constructor.construct_distribution(sim_result)

        assert constructor.validate_distribution(dist) is True

    def test_validate_distribution_invalid_probabilities(self):
        """Test distribution validation with invalid probabilities."""
        constructor = DistributionConstructor()

        # Create invalid distribution manually
        invalid_dist = DistributionResult(
            probabilities=np.array([0.3, 0.2, 0.1, 0.2, 0.1]),  # Sums to 0.9, not 1.0
            mean_rating=3.0,
            similarity_scores=np.array([0.2, 0.3, 0.5, 0.7, 0.9]),
            temperature=1.0,
            offset=0.0,
        )

        assert constructor.validate_distribution(invalid_dist) is False

    def test_validate_distribution_invalid_mean_rating(self):
        """Test distribution validation with invalid mean rating."""
        constructor = DistributionConstructor()

        # Create distribution with invalid mean rating
        invalid_dist = DistributionResult(
            probabilities=np.array([0.2, 0.2, 0.2, 0.2, 0.2]),
            mean_rating=6.0,  # Invalid: outside [1, 5]
            similarity_scores=np.array([0.2, 0.3, 0.5, 0.7, 0.9]),
            temperature=1.0,
            offset=0.0,
        )

        assert constructor.validate_distribution(invalid_dist) is False

    def test_get_distribution_statistics(self):
        """Test distribution statistics calculation."""
        constructor = DistributionConstructor()

        scores = np.array([0.2, 0.3, 0.5, 0.7, 0.9])
        sim_result = SimilarityResult(
            scores=scores,
            response_embedding=np.random.randn(1536),
            reference_embeddings=np.random.randn(5, 1536),
        )

        dist = constructor.construct_distribution(sim_result)

        stats = constructor.get_distribution_statistics(dist)

        # Check that all expected keys are present
        assert "mean" in stats
        assert "variance" in stats
        assert "std" in stats
        assert "mode" in stats
        assert "entropy" in stats

        # Validate values
        assert 1.0 <= stats["mean"] <= 5.0
        assert stats["variance"] >= 0
        assert stats["std"] >= 0
        assert 1 <= stats["mode"] <= 5
        assert stats["entropy"] >= 0

        # Mean should match distribution's mean_rating
        assert np.isclose(stats["mean"], dist.mean_rating, atol=1e-6)

    def test_distribution_statistics_mode_is_highest_probability(self):
        """Test that mode is the rating with highest probability."""
        constructor = DistributionConstructor()

        # Create scores strongly favoring rating 5
        scores = np.array([0.1, 0.2, 0.3, 0.6, 0.95])
        sim_result = SimilarityResult(
            scores=scores,
            response_embedding=np.random.randn(1536),
            reference_embeddings=np.random.randn(5, 1536),
        )

        dist = constructor.construct_distribution(sim_result)
        stats = constructor.get_distribution_statistics(dist)

        # Mode should be 5 (highest probability)
        assert stats["mode"] == 5

        # Verify manually
        manual_mode = int(np.argmax(dist.probabilities) + 1)
        assert stats["mode"] == manual_mode

    def test_entropy_higher_for_uniform_distribution(self):
        """Test that entropy is higher for more uniform distributions."""
        scores_peaked = np.array([0.1, 0.2, 0.3, 0.6, 0.95])  # Peaked at 5
        scores_uniform = np.array([0.5, 0.5, 0.5, 0.5, 0.5])  # Uniform

        sim_peaked = SimilarityResult(
            scores=scores_peaked,
            response_embedding=np.random.randn(1536),
            reference_embeddings=np.random.randn(5, 1536),
        )

        sim_uniform = SimilarityResult(
            scores=scores_uniform,
            response_embedding=np.random.randn(1536),
            reference_embeddings=np.random.randn(5, 1536),
        )

        constructor = DistributionConstructor()
        dist_peaked = constructor.construct_distribution(sim_peaked)
        dist_uniform = constructor.construct_distribution(sim_uniform)

        stats_peaked = constructor.get_distribution_statistics(dist_peaked)
        stats_uniform = constructor.get_distribution_statistics(dist_uniform)

        # Uniform distribution should have higher entropy
        assert stats_uniform["entropy"] > stats_peaked["entropy"]

    def test_variance_calculation(self):
        """Test variance calculation for distribution."""
        constructor = DistributionConstructor()

        scores = np.array([0.2, 0.3, 0.5, 0.7, 0.9])
        sim_result = SimilarityResult(
            scores=scores,
            response_embedding=np.random.randn(1536),
            reference_embeddings=np.random.randn(5, 1536),
        )

        dist = constructor.construct_distribution(sim_result)
        stats = constructor.get_distribution_statistics(dist)

        # Manual variance calculation
        ratings = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        mean = dist.mean_rating
        manual_variance = np.dot(dist.probabilities, (ratings - mean) ** 2)

        assert np.isclose(stats["variance"], manual_variance, atol=1e-6)

        # Std should be sqrt of variance
        assert np.isclose(stats["std"], np.sqrt(stats["variance"]), atol=1e-6)

    def test_construct_distribution_with_wrong_number_of_scores(self):
        """Test that wrong number of scores raises ValueError."""
        constructor = DistributionConstructor()

        # Only 4 scores instead of 5
        scores = np.array([0.2, 0.3, 0.5, 0.7])
        sim_result = SimilarityResult(
            scores=scores,
            response_embedding=np.random.randn(1536),
            reference_embeddings=np.random.randn(4, 1536),
        )

        with pytest.raises(ValueError, match="Expected 5 scores"):
            constructor.construct_distribution(sim_result)

    def test_paper_temperature_values(self):
        """Test with temperature values used in paper (0.5, 1.0, 1.5)."""
        scores = np.array([0.2, 0.3, 0.5, 0.7, 0.9])
        sim_result = SimilarityResult(
            scores=scores,
            response_embedding=np.random.randn(1536),
            reference_embeddings=np.random.randn(5, 1536),
        )

        # Test all three paper temperatures
        for temperature in [0.5, 1.0, 1.5]:
            constructor = DistributionConstructor(temperature=temperature)
            dist = constructor.construct_distribution(sim_result)

            # Should create valid distribution
            assert constructor.validate_distribution(dist)
            assert np.isclose(np.sum(dist.probabilities), 1.0, atol=1e-6)

    def test_numerical_stability_with_extreme_values(self):
        """Test numerical stability with extreme similarity scores."""
        constructor = DistributionConstructor()

        # Very large scores
        scores_large = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
        sim_large = SimilarityResult(
            scores=scores_large,
            response_embedding=np.random.randn(1536),
            reference_embeddings=np.random.randn(5, 1536),
        )

        dist_large = constructor.construct_distribution(sim_large)

        # Should still create valid distribution
        assert constructor.validate_distribution(dist_large)
        assert np.isclose(np.sum(dist_large.probabilities), 1.0, atol=1e-6)
        assert not np.any(np.isnan(dist_large.probabilities))
        assert not np.any(np.isinf(dist_large.probabilities))

        # Very small scores
        scores_small = np.array([1e-10, 2e-10, 3e-10, 4e-10, 5e-10])
        sim_small = SimilarityResult(
            scores=scores_small,
            response_embedding=np.random.randn(1536),
            reference_embeddings=np.random.randn(5, 1536),
        )

        dist_small = constructor.construct_distribution(sim_small)

        assert constructor.validate_distribution(dist_small)
        assert np.isclose(np.sum(dist_small.probabilities), 1.0, atol=1e-6)

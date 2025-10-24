"""
ABOUTME: Unit tests for SSR Engine orchestration and configuration
ABOUTME: Tests SSRConfig, SSRResult, and SSREngine using real components

This module tests the main SSR Engine which orchestrates all components:
- SSRConfig: Configuration validation and defaults
- SSRResult: Result data structure and methods
- SSREngine: Main processing logic and component coordination

All tests use real component instances following the project testing standards.
"""

import pytest
import numpy as np

from src.core.ssr_engine import SSREngine, SSRConfig, SSRResult
from src.core.distribution import DistributionResult
from src.core.similarity import SimilarityResult


class TestSSRConfig:
    """Test SSRConfig validation and defaults"""

    def test_default_configuration(self):
        """Test that default config has sensible values"""
        config = SSRConfig()

        assert config.temperature == 1.0
        assert config.offset == 0.0
        assert config.use_multi_set_averaging is True
        assert config.embedding_model == "text-embedding-3-small"
        assert config.embedding_dim == 1536
        assert config.enable_cache is True

    def test_default_reference_sets(self):
        """Test that default reference sets are paper's 6 sets"""
        config = SSRConfig()

        assert config.reference_set_ids is not None
        assert len(config.reference_set_ids) == 6
        assert all(
            set_id.startswith("paper_set_") for set_id in config.reference_set_ids
        )

    def test_custom_temperature(self):
        """Test custom temperature configuration"""
        config = SSRConfig(temperature=1.5)
        assert config.temperature == 1.5

    def test_temperature_validation(self):
        """Test that temperature must be positive"""
        with pytest.raises(ValueError, match="Temperature must be positive"):
            SSRConfig(temperature=0.0)

        with pytest.raises(ValueError, match="Temperature must be positive"):
            SSRConfig(temperature=-0.5)

    def test_embedding_dim_validation(self):
        """Test that embedding dimension must be positive"""
        with pytest.raises(ValueError, match="Embedding dimension must be positive"):
            SSRConfig(embedding_dim=0)

        with pytest.raises(ValueError, match="Embedding dimension must be positive"):
            SSRConfig(embedding_dim=-100)

    def test_custom_reference_sets(self):
        """Test custom reference set configuration"""
        custom_sets = ["set_a", "set_b", "set_c"]
        config = SSRConfig(reference_set_ids=custom_sets)

        assert config.reference_set_ids == custom_sets

    def test_disable_multi_set_averaging(self):
        """Test disabling multi-set averaging"""
        config = SSRConfig(use_multi_set_averaging=False)
        assert config.use_multi_set_averaging is False

    def test_disable_caching(self):
        """Test disabling embedding cache"""
        config = SSRConfig(enable_cache=False)
        assert config.enable_cache is False


class TestSSRResult:
    """Test SSRResult data structure and methods"""

    @pytest.fixture
    def sample_distribution(self):
        """Create a sample distribution for testing"""
        probabilities = np.array([0.1, 0.2, 0.3, 0.25, 0.15])
        mean_rating = 3.05
        return DistributionResult(
            probabilities=probabilities,
            mean_rating=mean_rating,
            similarity_scores=np.array([0.2, 0.4, 0.5, 0.7, 0.85]),
            temperature=1.0,
            offset=0.0,
        )

    @pytest.fixture
    def sample_ssr_result(self, sample_distribution):
        """Create a sample SSR result for testing"""
        return SSRResult(
            response_text="I'd probably buy it.",
            distribution=sample_distribution,
            mean_rating=3.05,
            reference_sets_used=6,
        )

    def test_ssr_result_initialization(self, sample_ssr_result):
        """Test SSRResult can be initialized properly"""
        assert sample_ssr_result.response_text == "I'd probably buy it."
        assert sample_ssr_result.mean_rating == 3.05
        assert sample_ssr_result.reference_sets_used == 6

    def test_get_rating_probability(self, sample_ssr_result):
        """Test getting probability for specific rating"""
        prob_1 = sample_ssr_result.get_rating_probability(1)
        prob_3 = sample_ssr_result.get_rating_probability(3)
        prob_5 = sample_ssr_result.get_rating_probability(5)

        assert prob_1 == 0.1
        assert prob_3 == 0.3
        assert prob_5 == 0.15

    def test_get_most_likely_rating(self, sample_ssr_result):
        """Test finding the most probable rating (mode)"""
        most_likely = sample_ssr_result.get_most_likely_rating()
        assert most_likely == 3  # Rating 3 has highest probability (0.3)

    def test_ssr_result_with_similarity_results(self, sample_distribution):
        """Test SSRResult with similarity results"""
        sim_result = SimilarityResult(
            scores=np.array([0.2, 0.4, 0.5, 0.7, 0.85]),
            response_embedding=np.random.randn(1536),
            reference_embeddings=np.random.randn(5, 1536),
        )

        ssr_result = SSRResult(
            response_text="Test response",
            distribution=sample_distribution,
            mean_rating=3.05,
            reference_sets_used=6,
            similarity_results=[sim_result],
        )

        assert len(ssr_result.similarity_results) == 1
        assert isinstance(ssr_result.similarity_results[0], SimilarityResult)


class TestSSREngineConfiguration:
    """
    Test SSR Engine configuration using real component instances

    These tests verify config validation and component initialization
    without requiring external API calls.
    """

    def test_ssr_config_attributes(self):
        """Test that SSRConfig properly stores configuration"""
        config = SSRConfig(
            temperature=1.5,
            offset=0.1,
            use_multi_set_averaging=True,
            embedding_model="text-embedding-3-small",
            embedding_dim=1536,
        )

        assert config.temperature == 1.5
        assert config.offset == 0.1
        assert config.use_multi_set_averaging is True
        assert config.embedding_model == "text-embedding-3-small"
        assert config.embedding_dim == 1536

    def test_config_validation_positive_temperature(self):
        """Test config validation rejects non-positive temperature"""
        with pytest.raises(ValueError, match="Temperature must be positive"):
            SSRConfig(temperature=0.0)

        with pytest.raises(ValueError, match="Temperature must be positive"):
            SSRConfig(temperature=-1.0)

    def test_config_validation_positive_embedding_dim(self):
        """Test config validation rejects non-positive embedding dimension"""
        with pytest.raises(ValueError, match="Embedding dimension must be positive"):
            SSRConfig(embedding_dim=0)

        with pytest.raises(ValueError, match="Embedding dimension must be positive"):
            SSRConfig(embedding_dim=-512)

    def test_ssr_result_structure(self):
        """Test SSRResult data structure"""
        # Create a simple distribution
        probabilities = np.array([0.05, 0.15, 0.40, 0.30, 0.10])
        dist = DistributionResult(
            probabilities=probabilities,
            mean_rating=3.25,
            similarity_scores=np.array([0.2, 0.4, 0.6, 0.75, 0.85]),
            temperature=1.0,
            offset=0.0,
        )

        # Create SSR result
        result = SSRResult(
            response_text="I might buy it.",
            distribution=dist,
            mean_rating=3.25,
            reference_sets_used=1,
        )

        # Verify structure
        assert result.response_text == "I might buy it."
        assert result.mean_rating == 3.25
        assert result.reference_sets_used == 1
        assert np.array_equal(result.distribution.probabilities, probabilities)

    def test_ssr_result_get_rating_probability(self):
        """Test getting probability for specific rating from SSRResult"""
        probabilities = np.array([0.10, 0.20, 0.35, 0.25, 0.10])
        dist = DistributionResult(
            probabilities=probabilities,
            mean_rating=3.05,
            similarity_scores=np.array([0.2, 0.4, 0.6, 0.75, 0.85]),
            temperature=1.0,
            offset=0.0,
        )

        result = SSRResult(
            response_text="Test",
            distribution=dist,
            mean_rating=3.05,
            reference_sets_used=1,
        )

        # Test each rating
        assert result.get_rating_probability(1) == 0.10
        assert result.get_rating_probability(2) == 0.20
        assert result.get_rating_probability(3) == 0.35
        assert result.get_rating_probability(4) == 0.25
        assert result.get_rating_probability(5) == 0.10

    def test_ssr_result_get_most_likely_rating(self):
        """Test finding most likely rating (mode) from SSRResult"""
        # Distribution strongly peaked at rating 4
        probabilities = np.array([0.05, 0.10, 0.15, 0.60, 0.10])
        dist = DistributionResult(
            probabilities=probabilities,
            mean_rating=3.6,
            similarity_scores=np.array([0.2, 0.4, 0.6, 0.85, 0.75]),
            temperature=1.0,
            offset=0.0,
        )

        result = SSRResult(
            response_text="Very likely to buy",
            distribution=dist,
            mean_rating=3.6,
            reference_sets_used=1,
        )

        most_likely = result.get_most_likely_rating()
        assert most_likely == 4  # Rating 4 has highest probability

    def test_config_update_temperature(self):
        """Test updating temperature configuration"""
        config = SSRConfig(temperature=1.0)
        assert config.temperature == 1.0

        # Create new config with different temperature
        config2 = SSRConfig(temperature=1.5)
        assert config2.temperature == 1.5

    def test_config_update_offset(self):
        """Test updating offset configuration"""
        config = SSRConfig(offset=0.0)
        assert config.offset == 0.0

        config2 = SSRConfig(offset=0.1)
        assert config2.offset == 0.1

    def test_config_update_multi_set_averaging(self):
        """Test toggling multi-set averaging"""
        config = SSRConfig(use_multi_set_averaging=True)
        assert config.use_multi_set_averaging is True

        config2 = SSRConfig(use_multi_set_averaging=False)
        assert config2.use_multi_set_averaging is False


@pytest.mark.requires_openai
class TestSSREngineWithAPI:
    """
    Tests requiring actual OpenAI API calls

    These tests are marked with @pytest.mark.requires_openai and will be skipped
    unless explicitly enabled with: pytest -m requires_openai

    They test the full end-to-end functionality with real API integration.
    """

    @pytest.fixture
    def real_engine(self, tmp_path):
        """
        Create SSR engine with real API connection

        Requires OPENAI_API_KEY environment variable to be set.
        """
        import os

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            pytest.skip("OPENAI_API_KEY not set")

        config = SSRConfig(temperature=1.0, use_multi_set_averaging=False)
        engine = SSREngine(api_key=api_key, config=config, data_dir=tmp_path)
        return engine

    def test_process_single_response_real_api(self, real_engine):
        """Test processing a single response with real API"""
        response_text = "I'd probably buy it."
        result = real_engine.process_response(response_text)

        # Verify result structure
        assert result.response_text == response_text
        assert 1.0 <= result.mean_rating <= 5.0
        assert result.reference_sets_used >= 1

        # Verify distribution is valid
        assert len(result.distribution.probabilities) == 5
        assert np.isclose(np.sum(result.distribution.probabilities), 1.0, atol=1e-6)
        assert np.all(result.distribution.probabilities >= 0)

    def test_process_batch_responses_real_api(self, real_engine):
        """Test batch processing with real API"""
        responses = [
            "Very likely I'd buy it.",
            "I probably wouldn't buy it.",
            "Not sure if I'd buy it.",
        ]

        results = real_engine.process_responses_batch(responses)

        assert len(results) == 3
        for i, result in enumerate(results):
            assert result.response_text == responses[i]
            assert 1.0 <= result.mean_rating <= 5.0
            assert np.isclose(np.sum(result.distribution.probabilities), 1.0, atol=1e-6)

    def test_engine_statistics_real_api(self, real_engine):
        """Test engine statistics tracking with real API"""
        # Process some responses
        real_engine.process_response("I'd buy it.")
        real_engine.process_response("I wouldn't buy it.")

        # Get statistics
        stats = real_engine.get_statistics()

        assert stats["responses_processed"] == 2
        assert stats["reference_sets_available"] >= 0
        assert stats["reference_sets_used"] >= 1
        assert stats["temperature"] == 1.0

    def test_update_config_real_api(self, real_engine):
        """Test updating configuration with real API"""
        # Initial config
        assert real_engine.config.temperature == 1.0

        # Update temperature
        real_engine.update_config(temperature=1.5)
        assert real_engine.config.temperature == 1.5

        # Process response with new config
        result = real_engine.process_response("Test response")
        assert result.config.temperature == 1.5

    def test_empty_response_validation(self, real_engine):
        """Test that empty response text is rejected"""
        with pytest.raises(ValueError, match="Response text cannot be empty"):
            real_engine.process_response("")

        with pytest.raises(ValueError, match="Response text cannot be empty"):
            real_engine.process_response("   ")  # Whitespace only

    def test_empty_batch_validation(self, real_engine):
        """Test that empty batch is rejected"""
        with pytest.raises(ValueError, match="Response texts list cannot be empty"):
            real_engine.process_responses_batch([])

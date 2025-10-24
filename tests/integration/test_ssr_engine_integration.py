"""Integration tests for SSR Engine with real OpenAI API calls."""

import pytest
import os
import numpy as np
from src.core.ssr_engine import SSREngine


@pytest.fixture
def api_key():
    """Get OpenAI API key from environment."""
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        pytest.skip("OPENAI_API_KEY not set")
    return key


@pytest.fixture
def ssr_engine(api_key):
    """Create SSREngine instance with real API key."""
    return SSREngine(api_key=api_key)


@pytest.fixture
def sample_reference_statements():
    """Sample reference statements for SSR testing."""
    return {
        "1": [
            "I would definitely buy this product immediately",
            "This is exactly what I've been looking for",
            "I'm very excited about this product",
        ],
        "2": [
            "I would probably buy this product",
            "This seems like a good option",
            "I'm interested in purchasing this",
        ],
        "3": [
            "I'm neutral about this product",
            "I might consider it",
            "Not sure if I need this",
        ],
        "4": [
            "I probably wouldn't buy this product",
            "This doesn't really appeal to me",
            "I'm not very interested",
        ],
        "5": [
            "I would never buy this product",
            "This is definitely not for me",
            "I have no interest in this at all",
        ],
    }


class TestSSREngineIntegration:
    """Integration test suite for SSR Engine with real API calls."""

    def test_get_single_embedding(self, ssr_engine):
        """Test getting embedding for a single text."""
        text = "I love this product and would buy it immediately"
        embedding = ssr_engine.get_embedding(text)

        # Validate embedding structure
        assert isinstance(embedding, np.ndarray)
        assert len(embedding.shape) == 1
        assert embedding.shape[0] == 1536  # text-embedding-3-small dimension

        # Embedding values should be normalized
        magnitude = np.linalg.norm(embedding)
        assert 0.99 < magnitude < 1.01  # Should be approximately unit length

    def test_get_embeddings_batch(self, ssr_engine):
        """Test getting embeddings for multiple texts."""
        texts = [
            "I really want this product",
            "This is okay I guess",
            "I don't like this at all",
        ]
        embeddings = ssr_engine.get_embeddings_batch(texts)

        # Validate batch structure
        assert len(embeddings) == 3
        assert all(isinstance(e, np.ndarray) for e in embeddings)
        assert all(e.shape[0] == 1536 for e in embeddings)

        # Embeddings should be different
        assert not np.allclose(embeddings[0], embeddings[1])
        assert not np.allclose(embeddings[1], embeddings[2])

    def test_cosine_similarity_calculation(self, ssr_engine):
        """Test cosine similarity between embeddings."""
        # Similar texts
        text1 = "I love this product"
        text2 = "I really like this product"

        # Dissimilar texts
        text3 = "I hate this product"

        emb1 = ssr_engine.get_embedding(text1)
        emb2 = ssr_engine.get_embedding(text2)
        emb3 = ssr_engine.get_embedding(text3)

        # Similar texts should have high similarity
        sim_similar = ssr_engine.cosine_similarity(emb1, emb2)
        assert sim_similar > 0.7

        # Opposite sentiment should have lower similarity
        sim_opposite = ssr_engine.cosine_similarity(emb1, emb3)
        assert sim_opposite < sim_similar

    def test_ssr_calculation_high_intent(self, ssr_engine, sample_reference_statements):
        """Test SSR calculation for high purchase intent response."""
        response = "I absolutely love this product and would buy it right now!"

        # Get embeddings
        response_embedding = ssr_engine.get_embedding(response)

        # Get reference embeddings
        reference_embeddings = {}
        for rating, statements in sample_reference_statements.items():
            reference_embeddings[rating] = ssr_engine.get_embeddings_batch(statements)

        # Calculate similarities to each rating
        max_similarities = {}
        for rating, ref_embs in reference_embeddings.items():
            similarities = [
                ssr_engine.cosine_similarity(response_embedding, ref_emb)
                for ref_emb in ref_embs
            ]
            max_similarities[rating] = max(similarities)

        # Highest similarity should be with rating "1" (high purchase intent)
        highest_rating = max(max_similarities, key=max_similarities.get)
        assert highest_rating in ["1", "2"], (
            f"High intent response should map to rating 1 or 2, got {highest_rating}"
        )

    def test_ssr_calculation_low_intent(self, ssr_engine, sample_reference_statements):
        """Test SSR calculation for low purchase intent response."""
        response = "I would never buy this product, it's completely useless to me"

        # Get embeddings
        response_embedding = ssr_engine.get_embedding(response)

        # Get reference embeddings
        reference_embeddings = {}
        for rating, statements in sample_reference_statements.items():
            reference_embeddings[rating] = ssr_engine.get_embeddings_batch(statements)

        # Calculate similarities to each rating
        max_similarities = {}
        for rating, ref_embs in reference_embeddings.items():
            similarities = [
                ssr_engine.cosine_similarity(response_embedding, ref_emb)
                for ref_emb in ref_embs
            ]
            max_similarities[rating] = max(similarities)

        # Highest similarity should be with rating "5" (low purchase intent)
        highest_rating = max(max_similarities, key=max_similarities.get)
        assert highest_rating in ["4", "5"], (
            f"Low intent response should map to rating 4 or 5, got {highest_rating}"
        )

    def test_distribution_calculation_uniform(self, ssr_engine):
        """Test distribution calculation with uniform averaging."""
        ssr_scores = [1, 2, 3, 4, 5, 1, 2, 3, 4, 5]

        distribution = ssr_engine.calculate_distribution(
            ssr_scores, averaging_strategy="uniform"
        )

        # Should have 5 bins
        assert len(distribution) == 5

        # Should sum to 1.0
        assert abs(sum(distribution) - 1.0) < 1e-6

        # Each rating appears twice, so should be 0.2 each
        expected = [0.2, 0.2, 0.2, 0.2, 0.2]
        np.testing.assert_array_almost_equal(distribution, expected)

    def test_distribution_calculation_weighted(self, ssr_engine):
        """Test distribution calculation with weighted averaging."""
        # Mostly rating "3" with some "2" and "4"
        ssr_scores = [3, 3, 3, 3, 3, 2, 4]

        distribution = ssr_engine.calculate_distribution(
            ssr_scores, averaging_strategy="weighted"
        )

        # Should have 5 bins
        assert len(distribution) == 5

        # Should sum to 1.0
        assert abs(sum(distribution) - 1.0) < 1e-6

        # Rating "3" should have highest probability
        assert distribution[2] > distribution[0]
        assert distribution[2] > distribution[1]
        assert distribution[2] > distribution[3]
        assert distribution[2] > distribution[4]

    def test_mean_rating_calculation(self, ssr_engine):
        """Test mean rating calculation from distribution."""
        # Centered distribution
        distribution = [0.1, 0.2, 0.4, 0.2, 0.1]
        mean = ssr_engine.calculate_mean_rating(distribution)

        # Mean should be close to 3.0
        assert 2.9 < mean < 3.1

    def test_std_rating_calculation(self, ssr_engine):
        """Test standard deviation calculation."""
        # Narrow distribution (low std)
        narrow_dist = [0.0, 0.0, 1.0, 0.0, 0.0]
        narrow_mean = ssr_engine.calculate_mean_rating(narrow_dist)
        narrow_std = ssr_engine.calculate_std_rating(narrow_dist, narrow_mean)

        # Wide distribution (high std)
        wide_dist = [0.5, 0.0, 0.0, 0.0, 0.5]
        wide_mean = ssr_engine.calculate_mean_rating(wide_dist)
        wide_std = ssr_engine.calculate_std_rating(wide_dist, wide_mean)

        # Wide distribution should have higher std
        assert wide_std > narrow_std

    def test_confidence_calculation(self, ssr_engine):
        """Test confidence calculation from distribution."""
        # High confidence (concentrated distribution)
        high_conf_dist = [0.9, 0.1, 0.0, 0.0, 0.0]
        high_conf = ssr_engine.calculate_confidence(high_conf_dist)

        # Low confidence (uniform distribution)
        low_conf_dist = [0.2, 0.2, 0.2, 0.2, 0.2]
        low_conf = ssr_engine.calculate_confidence(low_conf_dist)

        # High confidence should be greater
        assert high_conf > low_conf
        assert high_conf > 0.8
        assert low_conf < 0.5

    def test_embedding_consistency(self, ssr_engine):
        """Test that same text produces consistent embeddings."""
        text = "This is a test product"

        # Get embedding twice
        emb1 = ssr_engine.get_embedding(text)
        emb2 = ssr_engine.get_embedding(text)

        # Should be identical (deterministic)
        np.testing.assert_array_almost_equal(emb1, emb2, decimal=5)

    def test_embedding_semantic_similarity(self, ssr_engine):
        """Test that semantically similar texts have high similarity."""
        # Synonymous phrases
        text1 = "I want to purchase this item"
        text2 = "I would like to buy this product"

        emb1 = ssr_engine.get_embedding(text1)
        emb2 = ssr_engine.get_embedding(text2)

        similarity = ssr_engine.cosine_similarity(emb1, emb2)

        # Should have high similarity
        assert similarity > 0.7

    def test_full_ssr_workflow(self, ssr_engine, sample_reference_statements):
        """Test complete SSR workflow with multiple consumer responses."""
        # Simulate consumer responses
        consumer_responses = [
            "I would definitely buy this",
            "Not really interested",
            "It's okay, might consider it",
            "I love this product!",
            "Definitely not for me",
        ]

        # Get embeddings for all responses
        response_embeddings = ssr_engine.get_embeddings_batch(consumer_responses)

        # Get reference embeddings
        reference_embeddings = {}
        for rating, statements in sample_reference_statements.items():
            reference_embeddings[rating] = ssr_engine.get_embeddings_batch(statements)

        # Calculate SSR for each response
        ssr_scores = []
        for response_emb in response_embeddings:
            max_similarity = -1
            best_rating = None

            for rating, ref_embs in reference_embeddings.items():
                for ref_emb in ref_embs:
                    sim = ssr_engine.cosine_similarity(response_emb, ref_emb)
                    if sim > max_similarity:
                        max_similarity = sim
                        best_rating = int(rating)

            ssr_scores.append(best_rating)

        # Calculate distribution
        distribution = ssr_engine.calculate_distribution(ssr_scores)

        # Calculate metrics
        mean_rating = ssr_engine.calculate_mean_rating(distribution)
        std_rating = ssr_engine.calculate_std_rating(distribution, mean_rating)
        confidence = ssr_engine.calculate_confidence(distribution)

        # Validate results
        assert len(ssr_scores) == 5
        assert all(1 <= score <= 5 for score in ssr_scores)
        assert len(distribution) == 5
        assert abs(sum(distribution) - 1.0) < 1e-6
        assert 1.0 <= mean_rating <= 5.0
        assert 0.0 <= std_rating <= 2.0
        assert 0.0 <= confidence <= 1.0

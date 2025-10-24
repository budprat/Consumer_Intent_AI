# ABOUTME: Unit tests for similarity calculation using cosine similarity
# ABOUTME: Tests SimilarityCalculator and EmbeddingCache with real computations

import pytest
import numpy as np

from src.ssr.similarity import EmbeddingCache


@pytest.mark.unit
class TestSimilarityCalculator:
    """Test suite for cosine similarity calculations."""

    def test_identical_vectors_have_similarity_one(self, real_similarity_calculator):
        """Test that identical vectors have cosine similarity of 1.0."""
        calc = real_similarity_calculator

        # Create identical vectors
        vector = np.random.randn(1536)  # text-embedding-3-small dimension

        similarity = calc.calculate_cosine_similarity(vector, vector)

        assert np.isclose(similarity, 1.0, atol=1e-6)

    def test_orthogonal_vectors_have_similarity_zero(self, real_similarity_calculator):
        """Test that orthogonal vectors have cosine similarity of 0.0."""
        calc = real_similarity_calculator

        # Create orthogonal vectors in 3D for clarity
        vec1 = np.array([1.0, 0.0, 0.0])
        vec2 = np.array([0.0, 1.0, 0.0])

        similarity = calc.calculate_cosine_similarity(vec1, vec2)

        assert np.isclose(similarity, 0.0, atol=1e-6)

    def test_opposite_vectors_have_negative_similarity(
        self, real_similarity_calculator
    ):
        """Test that opposite vectors have negative cosine similarity."""
        calc = real_similarity_calculator

        vec1 = np.array([1.0, 1.0, 1.0])
        vec2 = np.array([-1.0, -1.0, -1.0])

        similarity = calc.calculate_cosine_similarity(vec1, vec2)

        assert similarity < 0
        assert np.isclose(similarity, -1.0, atol=1e-6)

    def test_similarity_range_is_valid(self, real_similarity_calculator):
        """Test that similarity is always in [-1, 1] range."""
        calc = real_similarity_calculator

        # Generate random vectors
        np.random.seed(42)
        for _ in range(100):
            vec1 = np.random.randn(1536)
            vec2 = np.random.randn(1536)

            similarity = calc.calculate_cosine_similarity(vec1, vec2)

            assert -1.0 <= similarity <= 1.0, (
                f"Similarity {similarity} outside valid range [-1, 1]"
            )

    def test_similarity_is_symmetric(self, real_similarity_calculator):
        """Test that similarity is symmetric: sim(A,B) = sim(B,A)."""
        calc = real_similarity_calculator

        vec1 = np.random.randn(1536)
        vec2 = np.random.randn(1536)

        sim_forward = calc.calculate_cosine_similarity(vec1, vec2)
        sim_backward = calc.calculate_cosine_similarity(vec2, vec1)

        assert np.isclose(sim_forward, sim_backward, atol=1e-10)

    def test_batch_similarity_calculation(self, real_similarity_calculator):
        """Test calculating similarities between one vector and multiple vectors."""
        calc = real_similarity_calculator

        # Create query vector and reference vectors
        query = np.random.randn(1536)
        references = [np.random.randn(1536) for _ in range(5)]

        similarities = calc.calculate_similarities_batch(query, references)

        assert len(similarities) == 5
        assert all(-1.0 <= sim <= 1.0 for sim in similarities)

    def test_zero_vector_handling(self, real_similarity_calculator):
        """Test that zero vectors are handled correctly."""
        calc = real_similarity_calculator

        zero_vec = np.zeros(1536)
        nonzero_vec = np.random.randn(1536)

        # Similarity with zero vector should raise error or return NaN
        with pytest.raises((ValueError, ZeroDivisionError)):
            calc.calculate_cosine_similarity(zero_vec, nonzero_vec)

    def test_normalized_vectors_optimization(self, real_similarity_calculator):
        """Test that pre-normalized vectors give same results."""
        calc = real_similarity_calculator

        # Create random vector
        vec1 = np.random.randn(1536)
        vec2 = np.random.randn(1536)

        # Calculate with unnormalized vectors
        sim_unnormalized = calc.calculate_cosine_similarity(vec1, vec2)

        # Normalize vectors manually
        vec1_norm = vec1 / np.linalg.norm(vec1)
        vec2_norm = vec2 / np.linalg.norm(vec2)

        # Calculate with normalized vectors
        sim_normalized = calc.calculate_cosine_similarity(vec1_norm, vec2_norm)

        assert np.isclose(sim_unnormalized, sim_normalized, atol=1e-6)


@pytest.mark.unit
class TestEmbeddingCache:
    """Test suite for embedding cache with file persistence."""

    def test_cache_initialization(self, test_cache_directory):
        """Test cache initializes with storage directory."""
        cache = EmbeddingCache(cache_dir=test_cache_directory)

        assert cache.cache_dir.exists()
        assert cache.cache_dir.is_dir()

    def test_store_and_retrieve_embedding(self, real_embedding_cache):
        """Test storing and retrieving an embedding."""
        cache = real_embedding_cache

        # Create test embedding
        text = "This is a test statement"
        embedding = np.random.randn(1536)

        # Store
        cache.store(text, embedding)

        # Retrieve
        retrieved = cache.get(text)

        assert retrieved is not None
        assert np.array_equal(retrieved, embedding)

    def test_cache_miss_returns_none(self, real_embedding_cache):
        """Test that cache miss returns None."""
        cache = real_embedding_cache

        result = cache.get("nonexistent text")

        assert result is None

    def test_cache_hit_after_store(self, real_embedding_cache):
        """Test cache hit detection."""
        cache = real_embedding_cache

        text = "Test text for cache hit"
        embedding = np.random.randn(1536)

        assert not cache.has(text)  # Initially not in cache

        cache.store(text, embedding)

        assert cache.has(text)  # Now in cache

    def test_multiple_embeddings_storage(self, real_embedding_cache):
        """Test storing multiple embeddings."""
        cache = real_embedding_cache

        embeddings_data = {
            "text_1": np.random.randn(1536),
            "text_2": np.random.randn(1536),
            "text_3": np.random.randn(1536),
        }

        # Store all
        for text, embedding in embeddings_data.items():
            cache.store(text, embedding)

        # Retrieve and verify all
        for text, original_embedding in embeddings_data.items():
            retrieved = cache.get(text)
            assert np.array_equal(retrieved, original_embedding)

    def test_cache_persistence_across_instances(self, test_cache_directory):
        """Test cache persists across different cache instances."""
        cache_dir = test_cache_directory / "persistence"
        cache_dir.mkdir(exist_ok=True)

        # Create first cache instance and store
        cache1 = EmbeddingCache(cache_dir=cache_dir)
        text = "Persistent text"
        embedding = np.random.randn(1536)
        cache1.store(text, embedding)

        # Create second cache instance
        cache2 = EmbeddingCache(cache_dir=cache_dir)

        # Retrieve from second instance
        retrieved = cache2.get(text)

        assert retrieved is not None
        assert np.array_equal(retrieved, embedding)

    def test_cache_key_normalization(self, real_embedding_cache):
        """Test that cache keys are normalized (whitespace, case)."""
        cache = real_embedding_cache

        embedding = np.random.randn(1536)

        # Store with one format
        cache.store("  Test  Text  ", embedding)

        # Retrieve with different whitespace (should normalize)
        retrieved = cache.get("Test Text")

        assert retrieved is not None
        assert np.array_equal(retrieved, embedding)

    def test_cache_size_tracking(self, real_embedding_cache):
        """Test tracking number of cached embeddings."""
        cache = real_embedding_cache

        initial_size = cache.size()

        # Add embeddings
        for i in range(10):
            cache.store(f"text_{i}", np.random.randn(1536))

        assert cache.size() == initial_size + 10

    def test_cache_clearing(self, real_embedding_cache):
        """Test clearing all cached embeddings."""
        cache = real_embedding_cache

        # Add embeddings
        for i in range(5):
            cache.store(f"text_{i}", np.random.randn(1536))

        assert cache.size() > 0

        # Clear cache
        cache.clear()

        assert cache.size() == 0
        assert cache.get("text_0") is None

    def test_cache_update_existing_entry(self, real_embedding_cache):
        """Test updating an existing cache entry."""
        cache = real_embedding_cache

        text = "Update test"
        embedding1 = np.random.randn(1536)
        embedding2 = np.random.randn(1536)

        # Store first embedding
        cache.store(text, embedding1)

        # Update with second embedding
        cache.store(text, embedding2, overwrite=True)

        # Retrieve and verify it's the second embedding
        retrieved = cache.get(text)

        assert np.array_equal(retrieved, embedding2)
        assert not np.array_equal(retrieved, embedding1)

    def test_cache_export_import(self, real_embedding_cache):
        """Test exporting and importing cache data."""
        cache = real_embedding_cache

        # Store data
        text = "Export test"
        embedding = np.random.randn(1536)
        cache.store(text, embedding)

        # Export
        exported_data = cache.export_all()

        assert isinstance(exported_data, dict)
        assert text in exported_data
        assert np.array_equal(exported_data[text], embedding)

        # Clear and import
        cache.clear()
        cache.import_all(exported_data)

        # Verify imported
        retrieved = cache.get(text)
        assert np.array_equal(retrieved, embedding)


@pytest.mark.unit
class TestSimilarityWithRealStatements:
    """Test similarity calculations with real reference statements."""

    def test_similar_statements_high_similarity(self, real_similarity_calculator):
        """Test that semantically similar statements have high similarity."""
        calc = real_similarity_calculator

        # These would need actual embeddings from OpenAI in real test
        # For unit test, we simulate the expected behavior
        # In integration tests, we'd use actual API calls

        # Simulate embeddings for similar statements
        stmt1_embedding = np.random.randn(1536)
        stmt2_embedding = stmt1_embedding + np.random.randn(1536) * 0.1  # Very similar

        similarity = calc.calculate_cosine_similarity(stmt1_embedding, stmt2_embedding)

        # Similar embeddings should have high similarity (>0.8)
        assert similarity > 0.8

    def test_dissimilar_statements_low_similarity(self, real_similarity_calculator):
        """Test that dissimilar statements have low similarity."""
        calc = real_similarity_calculator

        # Simulate embeddings for dissimilar statements
        stmt1_embedding = np.random.randn(1536)
        stmt2_embedding = -stmt1_embedding + np.random.randn(1536) * 0.5

        similarity = calc.calculate_cosine_similarity(stmt1_embedding, stmt2_embedding)

        # Dissimilar embeddings should have low similarity (<0.3)
        assert similarity < 0.3

    def test_response_to_reference_similarities(
        self, real_similarity_calculator, paper_reference_set_1
    ):
        """Test calculating similarities between response and all reference statements."""
        calc = real_similarity_calculator

        # Simulate response embedding
        response_embedding = np.random.randn(1536)

        # Simulate reference embeddings (5 statements)
        reference_embeddings = [np.random.randn(1536) for _ in range(5)]

        # Calculate similarities
        similarities = calc.calculate_similarities_batch(
            response_embedding, reference_embeddings
        )

        assert len(similarities) == 5
        assert all(-1.0 <= sim <= 1.0 for sim in similarities)
        # In real scenario, one should be notably higher than others

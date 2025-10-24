"""Simplified integration tests for Consumer Generator and SSR Engine services."""

import pytest
import os
from src.services.consumer_generator import ConsumerGenerator
from src.core.ssr_engine import SSREngine


@pytest.fixture
def api_key():
    """Get OpenAI API key from environment."""
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        pytest.skip("OPENAI_API_KEY not set")
    return key


class TestConsumerGeneratorIntegration:
    """Integration tests for ConsumerGenerator with real OpenAI API."""

    def test_generate_diverse_consumers(self, api_key):
        """Test generating diverse consumers with demographics."""
        generator = ConsumerGenerator(api_key=api_key)
        consumers = generator.generate_consumers(count=5, demographics_enabled=True)

        assert len(consumers) == 5
        for consumer in consumers:
            assert 20 <= consumer.age <= 75
            assert consumer.gender in ["Male", "Female", "Non-binary"]
            assert consumer.income in ["Low", "Middle", "High"]

        # Check diversity
        ages = set(c.age for c in consumers)
        genders = set(c.gender for c in consumers)
        assert len(ages) >= 3
        assert len(genders) >= 1

    def test_generate_consumer_response(self, api_key):
        """Test generating purchase intent response."""
        generator = ConsumerGenerator(api_key=api_key)
        consumers = generator.generate_consumers(count=1, demographics_enabled=True)
        consumer = consumers[0]

        response = generator.generate_response(
            consumer=consumer,
            product_name="Tesla Model 3",
            product_description="Electric sedan with autopilot",
            llm_model="gpt-4o",
            temperature=1.0,
        )

        assert isinstance(response, str)
        assert len(response) > 10
        print(f"\nGenerated response: {response}")

    def test_generate_consumers_without_demographics(self, api_key):
        """Test generating generic consumers."""
        generator = ConsumerGenerator(api_key=api_key)
        consumers = generator.generate_consumers(count=3, demographics_enabled=False)

        assert len(consumers) == 3
        for consumer in consumers:
            assert consumer.age == 35
            assert consumer.gender == "Generic"
            assert consumer.income == "Middle"


class TestSSREngineIntegration:
    """Integration tests for SSR Engine with real OpenAI API."""

    def test_get_embedding(self, api_key):
        """Test getting text embedding."""
        engine = SSREngine(api_key=api_key)
        embedding = engine.get_embedding("I love this product")

        assert len(embedding.shape) == 1
        assert embedding.shape[0] == 1536  # text-embedding-3-small dimension

    def test_get_embeddings_batch(self, api_key):
        """Test getting multiple embeddings."""
        engine = SSREngine(api_key=api_key)
        texts = [
            "I really want this product",
            "This is okay I guess",
            "I don't like this at all",
        ]
        embeddings = engine.get_embeddings_batch(texts)

        assert len(embeddings) == 3
        assert all(e.shape[0] == 1536 for e in embeddings)

    def test_cosine_similarity(self, api_key):
        """Test cosine similarity calculation."""
        engine = SSREngine(api_key=api_key)

        # Similar texts should have high similarity
        emb1 = engine.get_embedding("I love this product")
        emb2 = engine.get_embedding("I really like this product")
        similarity = engine.cosine_similarity(emb1, emb2)

        assert similarity > 0.7
        print(f"\nSimilarity between similar texts: {similarity:.4f}")

    def test_calculate_distribution(self, api_key):
        """Test SSR distribution calculation."""
        engine = SSREngine(api_key=api_key)
        ssr_scores = [1, 2, 3, 4, 5, 1, 2, 3, 4, 5]

        distribution = engine.calculate_distribution(ssr_scores)

        assert len(distribution) == 5
        assert abs(sum(distribution) - 1.0) < 1e-6
        print(f"\nDistribution: {distribution}")

    def test_calculate_mean_and_std(self, api_key):
        """Test mean and standard deviation calculation."""
        engine = SSREngine(api_key=api_key)
        distribution = [0.1, 0.2, 0.4, 0.2, 0.1]

        mean = engine.calculate_mean_rating(distribution)
        std = engine.calculate_std_rating(distribution, mean)

        assert 1.0 <= mean <= 5.0
        assert 0.0 <= std <= 2.0
        print(f"\nMean: {mean:.2f}, Std: {std:.2f}")

    def test_calculate_confidence(self, api_key):
        """Test confidence calculation."""
        engine = SSREngine(api_key=api_key)

        # High confidence (concentrated)
        high_conf_dist = [0.9, 0.1, 0.0, 0.0, 0.0]
        high_conf = engine.calculate_confidence(high_conf_dist)

        # Low confidence (uniform)
        low_conf_dist = [0.2, 0.2, 0.2, 0.2, 0.2]
        low_conf = engine.calculate_confidence(low_conf_dist)

        assert high_conf > low_conf
        print(f"\nHigh confidence: {high_conf:.4f}, Low confidence: {low_conf:.4f}")


class TestEndToEndIntegration:
    """End-to-end integration test combining Consumer Generator and SSR Engine."""

    def test_complete_ssr_workflow(self, api_key):
        """Test complete SSR workflow from consumer generation to rating calculation."""
        # Generate consumers
        generator = ConsumerGenerator(api_key=api_key)
        consumers = generator.generate_consumers(count=3, demographics_enabled=True)

        # Generate responses
        responses = []
        for consumer in consumers:
            response = generator.generate_response(
                consumer=consumer,
                product_name="iPhone 15 Pro",
                product_description="Latest flagship smartphone",
                llm_model="gpt-4o",
                temperature=1.0,
            )
            responses.append(response)

        print(f"\nGenerated {len(responses)} consumer responses:")
        for i, response in enumerate(responses, 1):
            print(f"{i}. {response}")

        # Calculate SSR
        engine = SSREngine(api_key=api_key)

        # Get response embeddings
        response_embeddings = engine.get_embeddings_batch(responses)

        # Simple reference statements
        reference_statements = {
            "1": ["I would definitely buy this product"],
            "2": ["I would probably buy this product"],
            "3": ["I'm neutral about this product"],
            "4": ["I probably wouldn't buy this product"],
            "5": ["I would never buy this product"],
        }

        # Get reference embeddings
        reference_embeddings = {}
        for rating, statements in reference_statements.items():
            reference_embeddings[rating] = engine.get_embeddings_batch(statements)

        # Calculate SSR for each response
        ssr_scores = []
        for response_emb in response_embeddings:
            max_similarity = -1
            best_rating = None

            for rating, ref_embs in reference_embeddings.items():
                for ref_emb in ref_embs:
                    sim = engine.cosine_similarity(response_emb, ref_emb)
                    if sim > max_similarity:
                        max_similarity = sim
                        best_rating = int(rating)

            ssr_scores.append(best_rating)

        print(f"\nSSR Scores: {ssr_scores}")

        # Calculate distribution and metrics
        distribution = engine.calculate_distribution(ssr_scores)
        mean_rating = engine.calculate_mean_rating(distribution)
        std_rating = engine.calculate_std_rating(distribution, mean_rating)
        confidence = engine.calculate_confidence(distribution)

        print(f"Mean Rating: {mean_rating:.2f}")
        print(f"Std Deviation: {std_rating:.2f}")
        print(f"Confidence: {confidence:.4f}")
        print(f"Distribution: {distribution}")

        # Validate results
        assert len(ssr_scores) == 3
        assert all(1 <= score <= 5 for score in ssr_scores)
        assert 1.0 <= mean_rating <= 5.0
        assert 0.0 <= confidence <= 1.0

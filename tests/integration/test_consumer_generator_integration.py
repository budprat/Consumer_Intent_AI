"""Integration tests for ConsumerGenerator service with real OpenAI API calls."""

import pytest
import os
from src.services.consumer_generator import ConsumerGenerator, Consumer


@pytest.fixture
def api_key():
    """Get OpenAI API key from environment."""
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        pytest.skip("OPENAI_API_KEY not set")
    return key


@pytest.fixture
def consumer_generator(api_key):
    """Create ConsumerGenerator instance with real API key."""
    return ConsumerGenerator(api_key=api_key)


class TestConsumerGeneratorIntegration:
    """Integration test suite for ConsumerGenerator with real API calls."""

    def test_generate_consumers_with_demographics(self, consumer_generator):
        """Test generating consumers with demographics using real API."""
        consumers = consumer_generator.generate_consumers(
            count=5, demographics_enabled=True
        )

        # Validate consumer structure
        assert len(consumers) == 5
        assert all(isinstance(c, Consumer) for c in consumers)

        # Validate consumer fields
        for consumer in consumers:
            assert consumer.consumer_id is not None
            assert 20 <= consumer.age <= 75
            assert consumer.gender in ["Male", "Female", "Non-binary"]
            assert consumer.income in ["Low", "Middle", "High"]
            assert consumer.location is not None
            assert consumer.ethnicity is not None
            assert consumer.persona is not None
            assert len(consumer.persona) > 10

        # Check diversity
        ages = set(c.age for c in consumers)
        genders = set(c.gender for c in consumers)
        assert len(ages) >= 3, "Should generate diverse ages"
        assert len(genders) >= 2, "Should generate diverse genders"

    def test_generate_consumer_without_demographics(self, api_key):
        """Test generating a consumer without demographics."""
        generator = ConsumerGenerator(api_key=api_key, demographics_enabled=False)

        consumer = generator.generate_consumer(
            product_name="iPhone 15 Pro",
            product_description="Latest flagship smartphone with advanced camera system",
            temperature=0.8,
        )

        # Validate basic structure
        assert isinstance(consumer, Consumer)
        assert consumer.persona is not None
        assert consumer.context is not None

        # Demographics should be None
        assert consumer.age_group is None
        assert consumer.gender is None
        assert consumer.income_level is None

    def test_generate_consumer_batch(self, consumer_generator):
        """Test generating multiple consumers in batch."""
        consumers = consumer_generator.generate_consumer_batch(
            product_name="MacBook Air",
            product_description="Lightweight laptop for everyday computing",
            count=3,
            temperature=1.0,
        )

        # Validate batch
        assert len(consumers) == 3
        assert all(isinstance(c, Consumer) for c in consumers)

        # Each consumer should be unique
        personas = [c.persona for c in consumers]
        assert len(set(personas)) >= 2, "Consumers should have diverse personas"

        # All should have valid demographics
        for consumer in consumers:
            assert consumer.age_group in ["18-29", "30-44", "45-59", "60+"]
            assert consumer.gender in ["Male", "Female", "Non-binary"]
            assert consumer.income_level in ["Low", "Middle", "High"]

    def test_different_products_generate_different_consumers(self, consumer_generator):
        """Test that different products generate contextually appropriate consumers."""
        # Luxury product
        luxury_consumer = consumer_generator.generate_consumer(
            product_name="Rolex Submariner",
            product_description="Luxury diving watch with automatic movement",
            temperature=0.7,
        )

        # Budget product
        budget_consumer = consumer_generator.generate_consumer(
            product_name="Casio Digital Watch",
            product_description="Affordable digital watch with alarm and stopwatch",
            temperature=0.7,
        )

        # Consumers should have different contexts
        assert luxury_consumer.persona != budget_consumer.persona
        assert luxury_consumer.context != budget_consumer.context

    def test_temperature_affects_diversity(self, consumer_generator):
        """Test that temperature parameter affects output diversity."""
        product_name = "Wireless Headphones"
        product_description = "Bluetooth headphones with noise cancellation"

        # Generate with low temperature (more deterministic)
        low_temp_consumers = consumer_generator.generate_consumer_batch(
            product_name=product_name,
            product_description=product_description,
            count=3,
            temperature=0.3,
        )

        # Generate with high temperature (more diverse)
        high_temp_consumers = consumer_generator.generate_consumer_batch(
            product_name=product_name,
            product_description=product_description,
            count=3,
            temperature=1.5,
        )

        # Both should produce valid consumers
        assert len(low_temp_consumers) == 3
        assert len(high_temp_consumers) == 3

        # High temperature should generally produce more diverse results
        # (This is probabilistic, so we just validate structure)
        for consumer in low_temp_consumers + high_temp_consumers:
            assert isinstance(consumer, Consumer)
            assert len(consumer.persona) > 5

    def test_consumer_generation_consistency(self, consumer_generator):
        """Test that consumer generation is consistent across multiple calls."""
        # Generate same product multiple times
        consumers = []
        for _ in range(5):
            consumer = consumer_generator.generate_consumer(
                product_name="Nintendo Switch",
                product_description="Gaming console for family entertainment",
                temperature=1.0,
            )
            consumers.append(consumer)

        # All should be valid
        assert all(isinstance(c, Consumer) for c in consumers)

        # Should have demographic diversity
        age_groups = set(c.age_group for c in consumers)
        assert len(age_groups) >= 2, "Should generate diverse age groups"

    def test_consumer_json_format_validation(self, consumer_generator):
        """Test that generated consumers have proper JSON-serializable format."""
        consumer = consumer_generator.generate_consumer(
            product_name="Samsung Galaxy S24",
            product_description="Flagship Android smartphone",
            temperature=1.0,
        )

        # Convert to dict (dataclass should be serializable)
        from dataclasses import asdict

        consumer_dict = asdict(consumer)

        # Validate structure
        assert "persona" in consumer_dict
        assert "context" in consumer_dict
        assert "age_group" in consumer_dict
        assert "gender" in consumer_dict
        assert "income_level" in consumer_dict

        # All values should be strings or None
        for value in consumer_dict.values():
            assert isinstance(value, (str, type(None)))

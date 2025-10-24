"""
ABOUTME: Unit tests for LLM interfaces module
ABOUTME: Tests ProductConcept, LLMResponse, and all LLM interface implementations

This test module covers:
1. ProductConcept dataclass creation and validation
2. LLMResponse dataclass creation and attributes
3. LLMInterface abstract base class
4. MockLLMInterface (real production class for offline testing)
5. GPT4oInterface and GeminiInterface initialization

Integration tests with real API calls are in tests/integration/test_llm_api_integration.py
"""

import pytest

from src.llm.interfaces import (
    ProductConcept,
    LLMResponse,
    LLMInterface,
    GPT4oInterface,
    GeminiInterface,
    MockLLMInterface,
)


# ============================================================================
# Test ProductConcept Dataclass
# ============================================================================


class TestProductConcept:
    """Test ProductConcept dataclass"""

    def test_basic_product_creation(self):
        """Test creating basic product concept with required fields"""
        product = ProductConcept(
            name="Smart Watch", description="A fitness tracking watch"
        )

        assert product.name == "Smart Watch"
        assert product.description == "A fitness tracking watch"
        assert product.image_url is None
        assert product.price is None
        assert product.category is None

    def test_product_with_all_fields(self):
        """Test creating product concept with all fields"""
        product = ProductConcept(
            name="Smart Watch Pro",
            description="Advanced fitness tracking watch with GPS",
            image_url="https://example.com/watch.jpg",
            price="$299",
            category="electronics",
        )

        assert product.name == "Smart Watch Pro"
        assert product.description == "Advanced fitness tracking watch with GPS"
        assert product.image_url == "https://example.com/watch.jpg"
        assert product.price == "$299"
        assert product.category == "electronics"

    def test_product_with_optional_fields(self):
        """Test creating product with some optional fields"""
        product = ProductConcept(
            name="Organic Tea", description="Premium green tea", price="$15.99"
        )

        assert product.name == "Organic Tea"
        assert product.description == "Premium green tea"
        assert product.price == "$15.99"
        assert product.image_url is None
        assert product.category is None

    def test_product_long_description(self):
        """Test product with multi-paragraph description"""
        long_desc = (
            "This advanced fitness watch offers comprehensive health tracking. "
            "Monitor your heart rate, sleep patterns, and daily activity levels. "
            "Includes GPS tracking for outdoor workouts and smart notifications."
        )
        product = ProductConcept(name="Fitness Tracker", description=long_desc)

        assert product.description == long_desc
        assert len(product.description) > 100


# ============================================================================
# Test LLMResponse Dataclass
# ============================================================================


class TestLLMResponse:
    """Test LLMResponse dataclass"""

    def test_basic_response_creation(self):
        """Test creating basic LLM response"""
        response = LLMResponse(
            text="I would probably buy this product.",
            model="gpt-4o",
            temperature=1.0,
            latency_ms=250.5,
        )

        assert response.text == "I would probably buy this product."
        assert response.model == "gpt-4o"
        assert response.temperature == 1.0
        assert response.latency_ms == 250.5
        assert response.token_count is None
        assert response.cached is False

    def test_response_with_all_fields(self):
        """Test LLM response with all fields"""
        response = LLMResponse(
            text="This product looks interesting to me.",
            model="gemini-2.0-flash-exp",
            temperature=0.8,
            latency_ms=180.2,
            token_count=45,
            cached=True,
        )

        assert response.text == "This product looks interesting to me."
        assert response.model == "gemini-2.0-flash-exp"
        assert response.temperature == 0.8
        assert response.latency_ms == 180.2
        assert response.token_count == 45
        assert response.cached is True

    def test_response_different_temperatures(self):
        """Test responses with different temperature values"""
        temps = [0.5, 1.0, 1.5]

        for temp in temps:
            response = LLMResponse(
                text="Test response",
                model="test-model",
                temperature=temp,
                latency_ms=100.0,
            )
            assert response.temperature == temp

    def test_response_with_long_text(self):
        """Test response with multi-sentence text"""
        long_text = (
            "I really like this fitness watch. "
            "It has all the features I need for my daily workouts. "
            "The price seems reasonable for the quality offered."
        )
        response = LLMResponse(
            text=long_text, model="gpt-4o", temperature=1.0, latency_ms=300.0
        )

        assert response.text == long_text
        assert len(response.text) > 100


# ============================================================================
# Test MockLLMInterface (Real Production Class)
# ============================================================================


class TestMockLLMInterface:
    """Test MockLLMInterface - real production class for offline testing"""

    def test_mock_llm_initialization(self):
        """Test creating MockLLMInterface"""
        mock_llm = MockLLMInterface()

        assert mock_llm.model_name == "mock-llm"
        assert mock_llm.default_temperature == 1.0
        assert mock_llm.requests_made == 0
        assert mock_llm.total_tokens == 0

    def test_mock_llm_custom_initialization(self):
        """Test creating MockLLMInterface with custom parameters"""
        mock_llm = MockLLMInterface(model_name="test-mock", default_temperature=0.7)

        assert mock_llm.model_name == "test-mock"
        assert mock_llm.default_temperature == 0.7

    def test_mock_llm_young_consumer_positive(self):
        """Test MockLLMInterface generates positive response for young consumer (age < 30)"""
        mock_llm = MockLLMInterface()

        demographics = {
            "age": 25,
            "gender": "Female",
            "income_level": "$50,000-$75,000",
            "location": "San Francisco, CA, USA",
            "ethnicity": "Asian",
        }

        product = ProductConcept(
            name="Smart Watch", description="Fitness tracking watch"
        )

        question = "Would you buy this product?"

        response = mock_llm.generate_response(demographics, product, question)

        assert isinstance(response, LLMResponse)
        assert "Smart Watch" in response.text
        assert response.model == "mock-llm"
        assert response.temperature == 1.0
        assert response.latency_ms > 0
        assert response.token_count > 0
        assert not response.cached

        # Young consumers should get positive response
        assert any(
            word in response.text.lower()
            for word in ["like", "innovative", "probably buy"]
        )

    def test_mock_llm_middle_aged_neutral(self):
        """Test MockLLMInterface generates neutral response for middle-aged consumer (30-50)"""
        mock_llm = MockLLMInterface()

        demographics = {
            "age": 40,
            "gender": "Male",
            "income_level": "$75,000-$99,999",
            "location": "New York, NY, USA",
            "ethnicity": "White",
        }

        product = ProductConcept(
            name="Tablet Device", description="10-inch tablet for productivity"
        )

        question = "What are your thoughts on this product?"

        response = mock_llm.generate_response(demographics, product, question)

        # Middle-aged consumers should get neutral response
        assert "interesting" in response.text.lower() or "consider" in response.text.lower()

    def test_mock_llm_senior_cautious(self):
        """Test MockLLMInterface generates cautious response for senior consumer (age >= 50)"""
        mock_llm = MockLLMInterface()

        demographics = {
            "age": 65,
            "gender": "Female",
            "income_level": "$100,000-$149,999",
            "location": "Chicago, IL, USA",
            "ethnicity": "Black",
        }

        product = ProductConcept(
            name="Virtual Reality Headset",
            description="Immersive VR gaming experience",
        )

        question = "How do you feel about purchasing this?"

        response = mock_llm.generate_response(demographics, product, question)

        # Senior consumers should get cautious response
        assert any(
            word in response.text.lower()
            for word in ["not convinced", "need", "probably won't"]
        )

    def test_mock_llm_updates_statistics(self):
        """Test that MockLLMInterface updates request and token statistics"""
        mock_llm = MockLLMInterface()

        assert mock_llm.requests_made == 0
        assert mock_llm.total_tokens == 0

        demographics = {"age": 30, "gender": "Male", "income_level": "$50,000-$75,000"}
        product = ProductConcept(name="Product", description="Description")
        question = "Question?"

        response1 = mock_llm.generate_response(demographics, product, question)

        assert mock_llm.requests_made == 1
        assert mock_llm.total_tokens == response1.token_count

        response2 = mock_llm.generate_response(demographics, product, question)

        assert mock_llm.requests_made == 2
        assert mock_llm.total_tokens == response1.token_count + response2.token_count

    def test_mock_llm_custom_temperature(self):
        """Test MockLLMInterface with custom temperature parameter"""
        mock_llm = MockLLMInterface(default_temperature=0.5)

        demographics = {"age": 28}
        product = ProductConcept(name="Test", description="Test product")

        # Use default temperature
        response1 = mock_llm.generate_response(demographics, product, "Question?")
        assert response1.temperature == 0.5

        # Use custom temperature
        response2 = mock_llm.generate_response(
            demographics, product, "Question?", temperature=1.2
        )
        assert response2.temperature == 1.2

    def test_mock_llm_get_statistics(self):
        """Test getting statistics from MockLLMInterface"""
        mock_llm = MockLLMInterface()

        demographics = {"age": 25}
        product = ProductConcept(name="Test", description="Test")

        mock_llm.generate_response(demographics, product, "Q?")
        mock_llm.generate_response(demographics, product, "Q?")

        stats = mock_llm.get_statistics()

        assert stats["model"] == "mock-llm"
        assert stats["requests_made"] == 2
        assert stats["total_tokens"] > 0

    def test_mock_llm_response_includes_product_name(self):
        """Test that MockLLMInterface response always includes product name"""
        mock_llm = MockLLMInterface()

        demographics = {"age": 35}
        product = ProductConcept(name="EcoBottle", description="Reusable water bottle")

        response = mock_llm.generate_response(demographics, product, "Question?")

        assert "EcoBottle" in response.text


# ============================================================================
# Test LLMInterface Abstract Base Class
# ============================================================================


class TestLLMInterface:
    """Test LLMInterface abstract base class"""

    def test_llm_interface_is_abstract(self):
        """Test that LLMInterface cannot be instantiated directly"""
        with pytest.raises(TypeError):
            # Cannot instantiate abstract class
            LLMInterface(model_name="test", default_temperature=1.0)  # type: ignore

    def test_llm_interface_requires_generate_response(self):
        """Test that subclasses must implement generate_response"""

        class IncompleteInterface(LLMInterface):
            """Incomplete implementation missing generate_response"""

            pass

        with pytest.raises(TypeError):
            # Missing abstract method generate_response
            IncompleteInterface(model_name="test", default_temperature=1.0)  # type: ignore


# ============================================================================
# Test GPT4oInterface Initialization (No API Calls)
# ============================================================================


class TestGPT4oInterfaceInit:
    """Test GPT4oInterface initialization without API calls"""

    def test_gpt4o_default_initialization(self):
        """Test GPT4o interface initialization with defaults"""
        try:
            # This requires openai library installed
            gpt4o = GPT4oInterface(api_key="test-key")

            assert gpt4o.model_name == "gpt-4o"
            assert gpt4o.default_temperature == 1.0
            assert gpt4o.requests_made == 0
            assert gpt4o.total_tokens == 0
            assert gpt4o.client is not None

        except ImportError:
            pytest.skip("OpenAI library not installed")

    def test_gpt4o_custom_initialization(self):
        """Test GPT4o interface with custom parameters"""
        try:
            gpt4o = GPT4oInterface(
                api_key="test-key", model_name="gpt-4o-mini", default_temperature=0.7
            )

            assert gpt4o.model_name == "gpt-4o-mini"
            assert gpt4o.default_temperature == 0.7

        except ImportError:
            pytest.skip("OpenAI library not installed")

    def test_gpt4o_requires_openai_library(self):
        """Test that GPT4o interface requires OpenAI library"""
        # Note: This test will pass if openai is installed, skip if not
        try:
            import openai

            # If openai is installed, initialization should work
            gpt4o = GPT4oInterface(api_key="test-key")
            assert gpt4o is not None

        except ImportError:
            # If openai is not installed, should raise ImportError
            with pytest.raises(ImportError, match="OpenAI library not installed"):
                GPT4oInterface(api_key="test-key")


# ============================================================================
# Test GeminiInterface Initialization (No API Calls)
# ============================================================================


class TestGeminiInterfaceInit:
    """Test GeminiInterface initialization without API calls"""

    def test_gemini_default_initialization(self):
        """Test Gemini interface initialization with defaults"""
        try:
            gemini = GeminiInterface(api_key="test-key")

            assert gemini.model_name == "gemini-2.0-flash-exp"
            assert gemini.default_temperature == 1.0
            assert gemini.requests_made == 0
            assert gemini.total_tokens == 0
            assert gemini.model is not None

        except ImportError:
            pytest.skip("Google Generative AI library not installed")

    def test_gemini_custom_initialization(self):
        """Test Gemini interface with custom parameters"""
        try:
            gemini = GeminiInterface(
                api_key="test-key",
                model_name="gemini-2.0-flash-exp-1.5",
                default_temperature=0.8,
            )

            assert gemini.model_name == "gemini-2.0-flash-exp-1.5"
            assert gemini.default_temperature == 0.8

        except ImportError:
            pytest.skip("Google Generative AI library not installed")

    def test_gemini_requires_genai_library(self):
        """Test that Gemini interface requires google-generativeai library"""
        try:
            import google.generativeai as genai

            # If genai is installed, initialization should work
            gemini = GeminiInterface(api_key="test-key")
            assert gemini is not None

        except ImportError:
            # If genai is not installed, should raise ImportError
            with pytest.raises(
                ImportError, match="Google Generative AI library not installed"
            ):
                GeminiInterface(api_key="test-key")


# ============================================================================
# Test Edge Cases and Error Handling
# ============================================================================


class TestEdgeCases:
    """Test edge cases and error handling"""

    def test_product_empty_strings(self):
        """Test product with empty strings"""
        product = ProductConcept(name="", description="")

        assert product.name == ""
        assert product.description == ""

    def test_response_zero_latency(self):
        """Test response with zero latency (edge case)"""
        response = LLMResponse(
            text="Response", model="test", temperature=1.0, latency_ms=0.0
        )

        assert response.latency_ms == 0.0

    def test_mock_llm_missing_demographic_keys(self):
        """Test MockLLMInterface with incomplete demographic data"""
        mock_llm = MockLLMInterface()

        # Missing some demographic fields
        demographics = {"age": 30}  # Only age provided

        product = ProductConcept(name="Test", description="Test product")

        response = mock_llm.generate_response(demographics, product, "Question?")

        # Should still generate a response
        assert isinstance(response, LLMResponse)
        assert len(response.text) > 0

    def test_mock_llm_zero_age(self):
        """Test MockLLMInterface with age 0 (edge case)"""
        mock_llm = MockLLMInterface()

        demographics = {"age": 0}
        product = ProductConcept(name="Test", description="Test")

        response = mock_llm.generate_response(demographics, product, "Q?")

        # Should generate positive response (age < 30)
        assert isinstance(response, LLMResponse)

    def test_mock_llm_very_high_age(self):
        """Test MockLLMInterface with very high age"""
        mock_llm = MockLLMInterface()

        demographics = {"age": 120}
        product = ProductConcept(name="Test", description="Test")

        response = mock_llm.generate_response(demographics, product, "Q?")

        # Should generate cautious response (age >= 50)
        assert "not convinced" in response.text.lower() or "probably won't" in response.text.lower()


# ============================================================================
# Test Paper Methodology Compliance
# ============================================================================


class TestPaperMethodology:
    """Test compliance with research paper methodology"""

    def test_response_length_appropriate(self):
        """Test that MockLLMInterface generates responses of appropriate length (1-3 sentences)"""
        mock_llm = MockLLMInterface()

        demographics = {"age": 30}
        product = ProductConcept(name="Test", description="Test product")

        response = mock_llm.generate_response(demographics, product, "Question?")

        # Count sentences (approximate - split by periods)
        sentences = [s.strip() for s in response.text.split(".") if s.strip()]

        # Should be 2-3 sentences based on implementation
        assert 2 <= len(sentences) <= 3

    def test_response_expresses_purchase_intent(self):
        """Test that responses express purchase intent/opinion"""
        mock_llm = MockLLMInterface()

        demographics = {"age": 25}
        product = ProductConcept(name="Product", description="Description")

        response = mock_llm.generate_response(demographics, product, "Would you buy?")

        # Should contain purchase-related language
        purchase_words = ["buy", "purchase", "like", "love", "consider", "convinced"]
        assert any(word in response.text.lower() for word in purchase_words)

    def test_demographic_conditioning_affects_response(self):
        """Test that different demographics produce different responses"""
        mock_llm = MockLLMInterface()

        product = ProductConcept(name="Tech Gadget", description="Latest technology")

        # Young consumer (positive)
        young_demo = {"age": 22}
        young_response = mock_llm.generate_response(young_demo, product, "Buy?")

        # Senior consumer (cautious)
        senior_demo = {"age": 70}
        senior_response = mock_llm.generate_response(senior_demo, product, "Buy?")

        # Responses should be different based on age
        assert young_response.text != senior_response.text
        assert "like" in young_response.text.lower() or "innovative" in young_response.text.lower()
        assert "not convinced" in senior_response.text.lower() or "probably won't" in senior_response.text.lower()

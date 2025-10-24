"""
ABOUTME: Abstract LLM interface and concrete implementations for GPT-4o and Gemini-2.0-flash
ABOUTME: Provides unified API for generating synthetic consumer responses with demographic conditioning

This module implements the Text Elicitation component of the SSR pipeline.
It generates brief textual responses (1-3 sentences) that express purchase intent
based on demographic attributes and product concepts.

Supported Models (from paper):
- GPT-4o (OpenAI): ρ = 90.2%, K^xy = 0.88
- Gemini-2.0-flash (Google): ρ = 90.6%, K^xy = 0.80
"""

import time
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from dataclasses import dataclass

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

try:
    from openai import OpenAI, OpenAIError
except ImportError:
    OpenAI = None
    OpenAIError = Exception

try:
    import google.generativeai as genai
except ImportError:
    genai = None


@dataclass
class ProductConcept:
    """
    Product concept for consumer evaluation

    Attributes:
        name: Product name
        description: Product description (1-3 paragraphs)
        image_url: Optional URL to product image
        price: Optional price information
        category: Product category (e.g., "electronics", "food")
    """

    name: str
    description: str
    image_url: Optional[str] = None
    price: Optional[str] = None
    category: Optional[str] = None


@dataclass
class LLMResponse:
    """
    Response from LLM generation

    Attributes:
        text: Generated response text
        model: Model used for generation
        temperature: Temperature parameter used
        latency_ms: Time taken to generate (milliseconds)
        token_count: Number of tokens in response (if available)
        cached: Whether response was retrieved from cache
    """

    text: str
    model: str
    temperature: float
    latency_ms: float
    token_count: Optional[int] = None
    cached: bool = False


class LLMInterface(ABC):
    """
    Abstract interface for LLM providers

    All LLM implementations must inherit from this class and implement
    the generate_response method. This ensures consistency across different
    LLM providers (OpenAI, Google, etc.).
    """

    def __init__(self, model_name: str, default_temperature: float = 1.0):
        """
        Initialize LLM interface

        Args:
            model_name: Name of the model to use
            default_temperature: Default sampling temperature
        """
        self.model_name = model_name
        self.default_temperature = default_temperature
        self.requests_made = 0
        self.total_tokens = 0

    @abstractmethod
    def generate_response(
        self,
        demographic_attributes: Dict[str, Any],
        product_concept: ProductConcept,
        question_prompt: str,
        temperature: Optional[float] = None,
    ) -> LLMResponse:
        """
        Generate synthetic consumer response

        Args:
            demographic_attributes: {age, gender, income, location, ethnicity}
            product_concept: Product to evaluate
            question_prompt: Question about purchase intent
            temperature: Sampling temperature (uses default if None)

        Returns:
            LLMResponse with brief textual response (1-3 sentences)

        Raises:
            Exception: If generation fails after retries
        """
        pass

    def get_statistics(self) -> Dict[str, Any]:
        """Get usage statistics"""
        return {
            "model": self.model_name,
            "requests_made": self.requests_made,
            "total_tokens": self.total_tokens,
        }


class GPT4oInterface(LLMInterface):
    """
    OpenAI GPT-4o implementation

    Performance (from paper):
    - Correlation attainment: ρ = 90.2%
    - KS similarity: K^xy = 0.88
    - Temperature range tested: 0.5 - 1.5
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gpt-4o",
        default_temperature: float = 1.0,
    ):
        """
        Initialize GPT-4o interface

        Args:
            api_key: OpenAI API key (or use OPENAI_API_KEY env var)
            model_name: Model name (default: gpt-4o)
            default_temperature: Default sampling temperature
        """
        super().__init__(model_name, default_temperature)

        if OpenAI is None:
            raise ImportError(
                "OpenAI library not installed. Install with: pip install openai"
            )

        self.client = OpenAI(api_key=api_key)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(OpenAIError),
    )
    def generate_response(
        self,
        demographic_attributes: Dict[str, Any],
        product_concept: ProductConcept,
        question_prompt: str,
        temperature: Optional[float] = None,
    ) -> LLMResponse:
        """
        Generate response using GPT-4o

        Args:
            demographic_attributes: Demographic profile
            product_concept: Product to evaluate
            question_prompt: Question prompt
            temperature: Sampling temperature

        Returns:
            LLMResponse with generated text
        """
        if temperature is None:
            temperature = self.default_temperature

        # Construct system and user prompts (will be done by PromptManager in 2.2)
        system_prompt = (
            "You are participating in a consumer research study. "
            "Please respond as yourself based on the demographic information provided."
        )

        # Format demographics
        demo_text = f"""Demographic Profile:
- Age: {demographic_attributes.get('age', 'Unknown')}
- Gender: {demographic_attributes.get('gender', 'Unknown')}
- Annual Household Income: {demographic_attributes.get('income_level', 'Unknown')}
- Location: {demographic_attributes.get('location', 'Unknown')}
- Ethnicity: {demographic_attributes.get('ethnicity', 'Unknown')}"""

        # Format product
        product_text = f"""Please examine the following product concept:

Product: {product_concept.name}
{product_concept.description}"""

        if product_concept.price:
            product_text += f"\nPrice: {product_concept.price}"

        # Combine into user message
        user_message = f"""{demo_text}

{product_text}

{question_prompt}

Please respond in 1-3 sentences expressing your feelings about purchasing this product."""

        # Call OpenAI API
        start_time = time.time()

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                temperature=temperature,
                max_tokens=150,  # Keep responses brief
            )

            latency_ms = (time.time() - start_time) * 1000

            response_text = response.choices[0].message.content.strip()
            token_count = response.usage.total_tokens

            # Update statistics
            self.requests_made += 1
            self.total_tokens += token_count

            return LLMResponse(
                text=response_text,
                model=self.model_name,
                temperature=temperature,
                latency_ms=latency_ms,
                token_count=token_count,
                cached=False,
            )

        except Exception as e:
            raise OpenAIError(f"Failed to generate response with GPT-4o: {str(e)}")


class GeminiInterface(LLMInterface):
    """
    Google Gemini-2.0-flash implementation

    Performance (from paper):
    - Correlation attainment: ρ = 90.6%
    - KS similarity: K^xy = 0.80
    - Temperature range tested: 0.5 - 1.5
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-2.0-flash-exp",
        default_temperature: float = 1.0,
    ):
        """
        Initialize Gemini interface

        Args:
            api_key: Google API key (or use GOOGLE_API_KEY env var)
            model_name: Model name (default: gemini-2.0-flash-exp)
            default_temperature: Default sampling temperature
        """
        super().__init__(model_name, default_temperature)

        if genai is None:
            raise ImportError(
                "Google Generative AI library not installed. "
                "Install with: pip install google-generativeai"
            )

        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    def generate_response(
        self,
            demographic_attributes: Dict[str, Any],
        product_concept: ProductConcept,
        question_prompt: str,
        temperature: Optional[float] = None,
    ) -> LLMResponse:
        """
        Generate response using Gemini

        Args:
            demographic_attributes: Demographic profile
            product_concept: Product to evaluate
            question_prompt: Question prompt
            temperature: Sampling temperature

        Returns:
            LLMResponse with generated text
        """
        if temperature is None:
            temperature = self.default_temperature

        # Construct prompt (similar to GPT-4o)
        system_instruction = (
            "You are participating in a consumer research study. "
            "Please respond as yourself based on the demographic information provided."
        )

        # Format demographics
        demo_text = f"""Demographic Profile:
- Age: {demographic_attributes.get('age', 'Unknown')}
- Gender: {demographic_attributes.get('gender', 'Unknown')}
- Annual Household Income: {demographic_attributes.get('income_level', 'Unknown')}
- Location: {demographic_attributes.get('location', 'Unknown')}
- Ethnicity: {demographic_attributes.get('ethnicity', 'Unknown')}"""

        # Format product
        product_text = f"""Please examine the following product concept:

Product: {product_concept.name}
{product_concept.description}"""

        if product_concept.price:
            product_text += f"\nPrice: {product_concept.price}"

        # Combine prompt
        full_prompt = f"""{system_instruction}

{demo_text}

{product_text}

{question_prompt}

Please respond in 1-3 sentences expressing your feelings about purchasing this product."""

        # Call Gemini API
        start_time = time.time()

        try:
            generation_config = genai.types.GenerationConfig(
                temperature=temperature, max_output_tokens=150
            )

            response = self.model.generate_content(
                full_prompt, generation_config=generation_config
            )

            latency_ms = (time.time() - start_time) * 1000

            response_text = response.text.strip()

            # Update statistics (Gemini doesn't provide token counts in same way)
            self.requests_made += 1
            self.total_tokens += len(response_text.split())  # Approximate

            return LLMResponse(
                text=response_text,
                model=self.model_name,
                temperature=temperature,
                latency_ms=latency_ms,
                token_count=None,  # Not directly available from Gemini
                cached=False,
            )

        except Exception as e:
            raise Exception(f"Failed to generate response with Gemini: {str(e)}")


class MockLLMInterface(LLMInterface):
    """
    Mock LLM for testing and development

    Generates deterministic responses based on demographic attributes
    without making actual API calls.
    """

    def __init__(self, model_name: str = "mock-llm", default_temperature: float = 1.0):
        """Initialize mock LLM"""
        super().__init__(model_name, default_temperature)

    def generate_response(
        self,
        demographic_attributes: Dict[str, Any],
        product_concept: ProductConcept,
        question_prompt: str,
        temperature: Optional[float] = None,
    ) -> LLMResponse:
        """
        Generate mock response

        Response sentiment varies based on age to simulate demographic effects:
        - Age 18-30: More positive (tech-forward)
        - Age 31-50: Neutral to positive
        - Age 51+: More cautious
        """
        if temperature is None:
            temperature = self.default_temperature

        age = demographic_attributes.get("age", 35)
        income = demographic_attributes.get("income_level", "$50,000-$75,000")

        start_time = time.time()

        # Simulate sentiment based on demographics
        if age < 30:
            sentiment = "positive"
            response = (
                f"I really like the concept of {product_concept.name}. "
                "It seems innovative and would fit my lifestyle well. "
                "I'd probably buy it."
            )
        elif age < 50:
            sentiment = "neutral"
            response = (
                f"The {product_concept.name} looks interesting. "
                "I'm not entirely sure if it's for me, but I'd consider it. "
                "Need to think about it more."
            )
        else:
            sentiment = "cautious"
            response = (
                f"I'm not convinced about {product_concept.name}. "
                "I'd need to see more reviews and understand the value better. "
                "Probably won't buy it right now."
            )

        latency_ms = (time.time() - start_time) * 1000

        self.requests_made += 1
        self.total_tokens += len(response.split())

        return LLMResponse(
            text=response,
            model=self.model_name,
            temperature=temperature,
            latency_ms=latency_ms,
            token_count=len(response.split()),
            cached=False,
        )


# Example usage and testing
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    load_dotenv()

    # Test product concept
    product = ProductConcept(
        name="Smart Fitness Watch Pro",
        description=(
            "Advanced fitness tracking watch with heart rate monitoring, "
            "GPS, sleep tracking, and 7-day battery life. "
            "Syncs with your smartphone for notifications and music control."
        ),
        price="$299",
        category="electronics",
    )

    # Test demographics
    demographics = {
        "age": 28,
        "gender": "Female",
        "income_level": "$50,000-$75,000",
        "location": "San Francisco, CA, USA",
        "ethnicity": "Asian",
    }

    question = (
        "In a few sentences, please describe your feelings about purchasing "
        "this product. Focus on your likelihood of buying it and the reasons "
        "behind your view."
    )

    print("Testing LLM Interfaces")
    print("=" * 60)

    # Test Mock LLM
    print("\n1. Testing Mock LLM:")
    mock_llm = MockLLMInterface()
    mock_response = mock_llm.generate_response(demographics, product, question)
    print(f"Response: {mock_response.text}")
    print(f"Latency: {mock_response.latency_ms:.2f}ms")
    print(f"Tokens: {mock_response.token_count}")

    # Test GPT-4o (if API key available)
    if os.getenv("OPENAI_API_KEY"):
        print("\n2. Testing GPT-4o:")
        try:
            gpt4o = GPT4oInterface(api_key=os.getenv("OPENAI_API_KEY"))
            gpt_response = gpt4o.generate_response(demographics, product, question)
            print(f"Response: {gpt_response.text}")
            print(f"Latency: {gpt_response.latency_ms:.2f}ms")
            print(f"Tokens: {gpt_response.token_count}")
        except Exception as e:
            print(f"GPT-4o test failed: {e}")
    else:
        print("\n2. Skipping GPT-4o test (no API key)")

    # Test Gemini (if API key available)
    if os.getenv("GOOGLE_API_KEY"):
        print("\n3. Testing Gemini:")
        try:
            gemini = GeminiInterface(api_key=os.getenv("GOOGLE_API_KEY"))
            gemini_response = gemini.generate_response(demographics, product, question)
            print(f"Response: {gemini_response.text}")
            print(f"Latency: {gemini_response.latency_ms:.2f}ms")
        except Exception as e:
            print(f"Gemini test failed: {e}")
    else:
        print("\n3. Skipping Gemini test (no API key)")

    print("\n" + "=" * 60)
    print("LLM Interface testing complete")

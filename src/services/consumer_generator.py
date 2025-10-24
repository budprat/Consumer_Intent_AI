"""
ABOUTME: Consumer Generator Service for creating synthetic consumers with demographics
ABOUTME: Generates LLM-based purchase intent responses with demographic conditioning
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from openai import OpenAI
import random


# ============================================================================
# DEMOGRAPHIC PROFILES (FROM PAPER)
# ============================================================================

DEMOGRAPHIC_PROFILES = {
    "age_ranges": [(20, 30), (31, 45), (46, 60), (61, 75)],
    "genders": ["Male", "Female", "Non-binary"],
    "income_levels": ["Low", "Middle", "High"],
    "locations": [
        "Urban Northeast",
        "Urban West",
        "Urban Midwest",
        "Urban South",
        "Suburban Northeast",
        "Suburban West",
        "Suburban Midwest",
        "Suburban South",
        "Rural",
    ],
    "ethnicities": [
        "Caucasian",
        "Hispanic",
        "African American",
        "Asian",
        "Mixed",
        "Other",
    ],
}


def get_income_statement(income_tier: str) -> str:
    """
    Convert income tier to descriptive statement from paper.
    From Table 3: Income level statements used in demographic conditioning.
    """
    INCOME_STATEMENTS = {
        "Low": "Living paycheck to paycheck",
        "Middle": "Managing but tight",
        "High": "Comfortable financially",
    }
    return INCOME_STATEMENTS.get(income_tier, "Managing but tight")


@dataclass
class Consumer:
    """Synthetic consumer with demographic profile"""

    consumer_id: str
    age: int
    gender: str
    income: str
    location: str
    ethnicity: str
    persona: str  # Short description of consumer persona


class ConsumerGenerator:
    """
    Service for generating synthetic consumers with purchase intent responses.

    This implements the demographic conditioning approach from the paper,
    creating diverse consumer profiles and generating realistic responses
    using LLM with demographic prompts.
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Consumer Generator.

        Args:
            api_key: OpenAI API key (or use OPENAI_API_KEY env var)
        """
        self.client = OpenAI(api_key=api_key)

    def generate_consumers(
        self,
        count: int,
        demographics_enabled: bool = True,
        demographic_filters: Optional[Dict[str, any]] = None,
    ) -> List[Consumer]:
        """
        Generate diverse synthetic consumers with optional demographic filtering.

        Args:
            count: Number of consumers to generate (3-5 recommended)
            demographics_enabled: If True, generate diverse demographics;
                                 If False, generate generic consumers
            demographic_filters: Optional filters to constrain demographics:
                - gender: "male" | "female" | "other" (maps to "Male" | "Female" | "Non-binary")
                - income_bracket: "low" | "middle" | "high" (maps to "Low" | "Middle" | "High")
                - location: str (matches against location list)

        Returns:
            List of Consumer objects with demographic profiles
        """
        import logging

        logger = logging.getLogger(__name__)

        if count < 1:
            raise ValueError("Count must be at least 1")

        consumers = []

        if demographics_enabled:
            # Parse and log demographic filters
            gender_filter = None
            income_filter = None
            location_filter = None

            if demographic_filters:
                logger.info(f"Applying demographic filters: {demographic_filters}")

                # Map frontend values to backend values
                if "gender" in demographic_filters and demographic_filters["gender"]:
                    gender_map = {
                        "male": "Male",
                        "female": "Female",
                        "other": "Non-binary",
                    }
                    gender_filter = gender_map.get(
                        demographic_filters["gender"].lower()
                    )
                    logger.info(f"Gender filter: {gender_filter}")

                if (
                    "income_bracket" in demographic_filters
                    and demographic_filters["income_bracket"]
                ):
                    income_map = {"low": "Low", "middle": "Middle", "high": "High"}
                    income_filter = income_map.get(
                        demographic_filters["income_bracket"].lower()
                    )
                    logger.info(f"Income filter: {income_filter}")

                if (
                    "location" in demographic_filters
                    and demographic_filters["location"]
                ):
                    location_filter = demographic_filters["location"]
                    logger.info(f"Location filter: {location_filter}")
            else:
                logger.info(
                    "No demographic filters applied - generating diverse random consumers"
                )

            # Generate diverse demographic profiles with optional filtering
            for i in range(count):
                age_range = random.choice(DEMOGRAPHIC_PROFILES["age_ranges"])
                age = random.randint(age_range[0], age_range[1])

                # Apply gender filter or random selection
                if gender_filter:
                    gender = gender_filter
                else:
                    gender = random.choice(DEMOGRAPHIC_PROFILES["genders"])

                # Apply income filter or random selection
                if income_filter:
                    income = income_filter
                else:
                    income = random.choice(DEMOGRAPHIC_PROFILES["income_levels"])

                # Apply location filter or random selection
                if location_filter:
                    # Try to find matching location, fallback to filter value
                    matching_locations = [
                        loc
                        for loc in DEMOGRAPHIC_PROFILES["locations"]
                        if location_filter.lower() in loc.lower()
                    ]
                    location = (
                        matching_locations[0] if matching_locations else location_filter
                    )
                else:
                    location = random.choice(DEMOGRAPHIC_PROFILES["locations"])

                consumer = Consumer(
                    consumer_id=f"consumer_{i + 1}",
                    age=age,
                    gender=gender,
                    income=income,
                    location=location,
                    ethnicity=random.choice(DEMOGRAPHIC_PROFILES["ethnicities"]),
                    persona=self._generate_persona(age, gender),
                )
                consumers.append(consumer)
                logger.debug(
                    f"Generated consumer {i + 1}: {consumer.persona} - Gender: {gender}, Income: {income}, Location: {location}"
                )
        else:
            # Generate generic consumers (simplified demographics)
            logger.info("Demographics disabled - generating generic consumers")
            for i in range(count):
                consumer = Consumer(
                    consumer_id=f"consumer_{i + 1}",
                    age=35,
                    gender="Generic",
                    income="Middle",
                    location="United States",
                    ethnicity="Generic",
                    persona="General consumer",
                )
                consumers.append(consumer)

        return consumers

    def _generate_persona(self, age: int, gender: str) -> str:
        """Generate a short persona description based on demographics"""
        if age < 30:
            life_stage = "young professional"
        elif age < 45:
            life_stage = "mid-career professional"
        elif age < 60:
            life_stage = "established professional"
        else:
            life_stage = "senior professional"

        return f"{age}-year-old {gender.lower()}, {life_stage}"

    def generate_response(
        self,
        consumer: Consumer,
        product_name: str,
        product_description: str,
        llm_model: str = "gpt-3.5-turbo",
        temperature: float = 1.0,
    ) -> str:
        """
        Generate consumer's purchase intent response using LLM.

        This follows the demographic conditioning approach from the paper,
        including all 5 demographic factors for maximum correlation.

        Args:
            consumer: Consumer with demographic profile
            product_name: Name of the product
            product_description: Product description
            llm_model: LLM model to use (default: gpt-3.5-turbo)
            temperature: LLM temperature (default: 1.0 optimal from paper)

        Returns:
            Consumer's textual response expressing purchase intent
        """
        # Build demographic prompt (following paper - spec lines 627-633)
        # All 5 demographic factors from paper for maximum correlation
        system_prompt = f"""You are participating in a consumer research survey.
Impersonate a consumer with the following characteristics:
- Age: {consumer.age}
- Gender: {consumer.gender}
- Income Level: {get_income_statement(consumer.income)}
- Location: {consumer.location}
- Ethnicity: {consumer.ethnicity}

Respond naturally as this person would, considering their financial situation and life circumstances."""

        user_prompt = f"""Product: {product_name}
Description: {product_description}

How likely would you be to purchase this product? Reply briefly (1-2 sentences) to express your purchase intent."""

        try:
            response = self.client.chat.completions.create(
                model=llm_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                top_p=0.9,  # Important for response diversity (spec line 730)
                max_tokens=60,
            )

            return response.choices[0].message.content.strip()

        except Exception:
            # Fallback to rule-based response if LLM fails
            return self._generate_fallback_response(
                consumer, product_name, product_description
            )

    def _generate_fallback_response(
        self, consumer: Consumer, product_name: str, product_description: str
    ) -> str:
        """
        Generate rule-based fallback response when LLM fails.

        Uses simple heuristics based on income level to generate
        a reasonable purchase intent response.
        """
        income_map = {"Low": 2, "Middle": 3, "High": 4}
        intent_level = income_map.get(consumer.income, 3)

        # Add some randomness
        intent_level += random.choice([-1, 0, 1])
        intent_level = max(1, min(5, intent_level))

        responses = {
            1: f"It's very unlikely that I'd buy {product_name}.",
            2: f"I probably wouldn't purchase {product_name}.",
            3: f"I'm not sure if I would buy {product_name} or not.",
            4: f"I'd probably give {product_name} a try.",
            5: f"I'd definitely buy {product_name}.",
        }

        return responses.get(intent_level, responses[3])

    def generate_responses_batch(
        self,
        consumers: List[Consumer],
        product_name: str,
        product_description: str,
        llm_model: str = "gpt-3.5-turbo",
        temperature: float = 1.0,
    ) -> List[str]:
        """
        Generate responses for multiple consumers efficiently.

        Args:
            consumers: List of Consumer objects
            product_name: Name of the product
            product_description: Product description
            llm_model: LLM model to use
            temperature: LLM temperature

        Returns:
            List of consumer responses (same order as input consumers)
        """
        responses = []
        for consumer in consumers:
            response = self.generate_response(
                consumer, product_name, product_description, llm_model, temperature
            )
            responses.append(response)

        return responses

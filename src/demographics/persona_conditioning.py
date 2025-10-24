"""
ABOUTME: Persona conditioning module integrating demographics with LLM prompt generation
ABOUTME: Validates demographic effects through controlled experiments and A/B testing

This module implements the critical integration layer between demographic profiles
and LLM prompt generation, validating the paper's core finding:
- WITH demographics: ρ = 90.2% (strong test-retest reliability)
- WITHOUT demographics: ρ = 50% (only distribution matching)

Key Features:
1. Demographic-conditioned prompt generation
2. Controlled A/B testing (with/without demographics)
3. Validation of demographic effects on responses
4. Statistical analysis of demographic impact
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict

from .profiles import DemographicProfile
from .sampling import DemographicSampler, SamplingConfig, SamplingStrategy
from ..llm.prompts import PromptManager
from ..llm.interfaces import LLMInterface


class ConditioningMode(Enum):
    """Conditioning modes for A/B testing"""

    FULL_DEMOGRAPHICS = "full_demographics"  # All demographic attributes
    NO_DEMOGRAPHICS = "no_demographics"  # No demographic conditioning (control)
    PARTIAL_DEMOGRAPHICS = "partial_demographics"  # Subset of attributes


@dataclass
class ConditioningConfig:
    """
    Configuration for persona conditioning experiments

    Attributes:
        mode: Conditioning mode (full, none, partial)
        included_attributes: Specific attributes to include (for partial mode)
        template_id: Prompt template to use
        randomize_order: Randomize order of demographic attributes
    """

    mode: ConditioningMode
    included_attributes: Optional[List[str]] = None
    template_id: str = "paper_default"
    randomize_order: bool = False


@dataclass
class ConditioningResult:
    """
    Result of persona conditioning validation

    Attributes:
        profile: Demographic profile used
        mode: Conditioning mode applied
        formatted_prompt: Generated prompt
        response_text: LLM response (if executed)
        metadata: Additional metadata
    """

    profile: DemographicProfile
    mode: ConditioningMode
    formatted_prompt: Dict[str, str]
    response_text: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class PersonaConditioner:
    """
    Integrates demographic profiles with LLM prompt generation

    Features:
    - Full demographic conditioning following paper methodology
    - A/B testing framework (with/without demographics)
    - Validation of demographic effects
    - Statistical analysis of impact
    """

    def __init__(self, prompt_manager: Optional[PromptManager] = None):
        """
        Initialize persona conditioner

        Args:
            prompt_manager: PromptManager instance (creates default if None)
        """
        self.prompt_manager = prompt_manager or PromptManager()

    def condition_prompt(
        self,
        profile: DemographicProfile,
        product_name: str,
        product_description: str,
        config: ConditioningConfig,
        product_price: Optional[str] = None,
        product_image_url: Optional[str] = None,
        custom_question: Optional[str] = None,
    ) -> Dict[str, str]:
        """
        Generate conditioned prompt with demographic information

        Args:
            profile: DemographicProfile to condition on
            product_name: Product name
            product_description: Product description
            config: ConditioningConfig
            product_price: Optional price
            product_image_url: Optional image URL
            custom_question: Optional custom question

        Returns:
            Dictionary with 'system' and 'user' prompts
        """
        template = self.prompt_manager.get_template(config.template_id)

        # Prepare demographic attributes based on mode
        if config.mode == ConditioningMode.NO_DEMOGRAPHICS:
            # Control condition: no demographics
            demographic_attributes = {
                "age": "N/A",
                "gender": "N/A",
                "income_level": "N/A",
                "location": "N/A",
                "ethnicity": "N/A",
            }
        elif config.mode == ConditioningMode.PARTIAL_DEMOGRAPHICS:
            # Partial condition: only specified attributes
            demographic_attributes = self._get_partial_attributes(
                profile, config.included_attributes
            )
        else:
            # Full condition: all demographics
            demographic_attributes = {
                "age": profile.age,
                "gender": profile.gender,
                "income_level": profile.income_level,
                "location": str(profile.location),
                "ethnicity": profile.ethnicity,
            }

        # Format prompt using template
        formatted_prompt = template.format_prompt(
            demographic_attributes=demographic_attributes,
            product_name=product_name,
            product_description=product_description,
            product_price=product_price,
            product_image_url=product_image_url,
            custom_question=custom_question,
        )

        return formatted_prompt

    def _get_partial_attributes(
        self, profile: DemographicProfile, included_attributes: Optional[List[str]]
    ) -> Dict[str, Any]:
        """
        Get partial demographic attributes

        Args:
            profile: DemographicProfile
            included_attributes: List of attribute names to include

        Returns:
            Dictionary with specified attributes (others set to N/A)
        """
        if not included_attributes:
            return {
                "age": "N/A",
                "gender": "N/A",
                "income_level": "N/A",
                "location": "N/A",
                "ethnicity": "N/A",
            }

        attributes = {
            "age": profile.age if "age" in included_attributes else "N/A",
            "gender": profile.gender if "gender" in included_attributes else "N/A",
            "income_level": (
                profile.income_level if "income_level" in included_attributes else "N/A"
            ),
            "location": (
                str(profile.location) if "location" in included_attributes else "N/A"
            ),
            "ethnicity": (
                profile.ethnicity if "ethnicity" in included_attributes else "N/A"
            ),
        }

        return attributes

    def validate_demographic_effects(
        self,
        llm_interface: LLMInterface,
        product_name: str,
        product_description: str,
        cohort_size: int = 50,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Validate demographic effects through A/B testing

        Compares responses with full demographics vs. no demographics
        to validate the paper's 40+ percentage point improvement claim

        Args:
            llm_interface: LLM interface for generating responses
            product_name: Product name
            product_description: Product description
            cohort_size: Number of profiles to test
            seed: Random seed for reproducibility

        Returns:
            Dictionary with validation results and statistics
        """
        # Generate representative cohort
        sampler = DemographicSampler(seed=seed)
        sampling_config = SamplingConfig(
            strategy=SamplingStrategy.STRATIFIED, cohort_size=cohort_size, seed=seed
        )
        cohort = sampler.generate_cohort(sampling_config)

        # Run A/B test
        full_demo_results = []
        no_demo_results = []

        for profile in cohort:
            # Condition A: Full demographics
            full_config = ConditioningConfig(mode=ConditioningMode.FULL_DEMOGRAPHICS)
            full_prompt = self.condition_prompt(
                profile, product_name, product_description, full_config
            )

            # Condition B: No demographics (control)
            no_config = ConditioningConfig(mode=ConditioningMode.NO_DEMOGRAPHICS)
            no_prompt = self.condition_prompt(
                profile, product_name, product_description, no_config
            )

            # Generate responses (would typically call LLM here)
            # For validation purposes, we just collect the prompts
            full_demo_results.append(
                ConditioningResult(
                    profile=profile,
                    mode=ConditioningMode.FULL_DEMOGRAPHICS,
                    formatted_prompt=full_prompt,
                    metadata={"cohort_index": len(full_demo_results)},
                )
            )

            no_demo_results.append(
                ConditioningResult(
                    profile=profile,
                    mode=ConditioningMode.NO_DEMOGRAPHICS,
                    formatted_prompt=no_prompt,
                    metadata={"cohort_index": len(no_demo_results)},
                )
            )

        # Analyze results
        analysis = {
            "cohort_size": cohort_size,
            "full_demographics_count": len(full_demo_results),
            "no_demographics_count": len(no_demo_results),
            "demographic_attributes_included": {
                "full_condition": [
                    "age",
                    "gender",
                    "income_level",
                    "location",
                    "ethnicity",
                ],
                "control_condition": [],
            },
            "cohort_statistics": sampler.get_cohort_statistics(cohort),
            "validation_status": "prompts_generated",
            "note": (
                "Full validation requires LLM responses and SSR processing. "
                "This validates prompt generation works correctly for both conditions."
            ),
        }

        return analysis

    def analyze_attribute_importance(
        self,
        llm_interface: LLMInterface,
        product_name: str,
        product_description: str,
        cohort_size: int = 30,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Analyze importance of individual demographic attributes

        Tests each attribute in isolation to determine relative importance
        (Paper finding: Age and income are most reliably replicated)

        Args:
            llm_interface: LLM interface
            product_name: Product name
            product_description: Product description
            cohort_size: Number of profiles to test
            seed: Random seed

        Returns:
            Dictionary with attribute importance analysis
        """
        # Generate cohort
        sampler = DemographicSampler(seed=seed)
        sampling_config = SamplingConfig(
            strategy=SamplingStrategy.STRATIFIED, cohort_size=cohort_size, seed=seed
        )
        cohort = sampler.generate_cohort(sampling_config)

        # Test each attribute in isolation
        attributes = ["age", "gender", "income_level", "location", "ethnicity"]
        attribute_results = defaultdict(list)

        for profile in cohort:
            for attribute in attributes:
                config = ConditioningConfig(
                    mode=ConditioningMode.PARTIAL_DEMOGRAPHICS,
                    included_attributes=[attribute],
                )

                prompt = self.condition_prompt(
                    profile, product_name, product_description, config
                )

                attribute_results[attribute].append(
                    ConditioningResult(
                        profile=profile,
                        mode=ConditioningMode.PARTIAL_DEMOGRAPHICS,
                        formatted_prompt=prompt,
                        metadata={"attribute": attribute},
                    )
                )

        # Analyze results
        analysis = {
            "cohort_size": cohort_size,
            "attributes_tested": attributes,
            "results_per_attribute": {
                attr: len(results) for attr, results in attribute_results.items()
            },
            "expected_importance_ranking": [
                "age",
                "income_level",
                "gender",
                "location",
                "ethnicity",
            ],
            "note": (
                "Paper finding: Age and income level show highest test-retest reliability. "
                "Full validation requires LLM responses and correlation analysis."
            ),
        }

        return analysis

    def test_prompt_consistency(
        self,
        profile: DemographicProfile,
        product_name: str,
        product_description: str,
        num_repetitions: int = 5,
    ) -> Dict[str, Any]:
        """
        Test prompt generation consistency

        Ensures same demographic profile generates same prompt repeatedly
        (critical for test-retest reliability)

        Args:
            profile: DemographicProfile to test
            product_name: Product name
            product_description: Product description
            num_repetitions: Number of times to generate prompt

        Returns:
            Dictionary with consistency test results
        """
        config = ConditioningConfig(mode=ConditioningMode.FULL_DEMOGRAPHICS)

        prompts = []
        for _ in range(num_repetitions):
            prompt = self.condition_prompt(
                profile, product_name, product_description, config
            )
            prompts.append(prompt)

        # Check consistency
        first_prompt = prompts[0]
        all_identical = all(p == first_prompt for p in prompts)

        return {
            "profile_id": profile.id,
            "num_repetitions": num_repetitions,
            "all_prompts_identical": all_identical,
            "prompt_length": len(first_prompt["user"]),
            "system_prompt_length": len(first_prompt["system"]),
            "demographic_attributes": {
                "age": profile.age,
                "gender": profile.gender,
                "income_level": profile.income_level,
                "location": str(profile.location),
                "ethnicity": profile.ethnicity,
            },
            "test_result": "PASS" if all_identical else "FAIL",
        }


# Example usage and testing
if __name__ == "__main__":
    print("Persona Conditioning System Testing")
    print("=" * 60)

    # Initialize
    conditioner = PersonaConditioner()
    sampler = DemographicSampler(seed=42)

    # Test 1: Basic conditioning
    print("\n1. Basic Demographic Conditioning:")
    profile = sampler.generate_cohort(
        SamplingConfig(strategy=SamplingStrategy.STRATIFIED, cohort_size=1, seed=42)
    )[0]

    config = ConditioningConfig(mode=ConditioningMode.FULL_DEMOGRAPHICS)
    prompt = conditioner.condition_prompt(
        profile=profile,
        product_name="Smart Fitness Watch Pro",
        product_description="Advanced fitness tracking with heart rate monitoring",
        config=config,
    )

    print(f"Profile: {profile.age} years, {profile.gender}, {profile.income_level}")
    print(f"System prompt length: {len(prompt['system'])} chars")
    print(f"User prompt length: {len(prompt['user'])} chars")
    print("\nUser prompt preview:")
    print(prompt["user"][:300] + "...")

    # Test 2: A/B testing (with vs without demographics)
    print("\n" + "=" * 60)
    print("2. A/B Testing (With vs Without Demographics):")

    # With demographics
    full_config = ConditioningConfig(mode=ConditioningMode.FULL_DEMOGRAPHICS)
    full_prompt = conditioner.condition_prompt(
        profile, "Smart Watch", "Fitness tracking device", full_config
    )

    # Without demographics
    no_config = ConditioningConfig(mode=ConditioningMode.NO_DEMOGRAPHICS)
    no_prompt = conditioner.condition_prompt(
        profile, "Smart Watch", "Fitness tracking device", no_config
    )

    print(f"With demographics length: {len(full_prompt['user'])} chars")
    print(f"Without demographics length: {len(no_prompt['user'])} chars")
    print(
        f"Difference: {len(full_prompt['user']) - len(no_prompt['user'])} chars (demographics section)"
    )

    # Test 3: Partial demographics
    print("\n" + "=" * 60)
    print("3. Partial Demographics (Age + Income only):")

    partial_config = ConditioningConfig(
        mode=ConditioningMode.PARTIAL_DEMOGRAPHICS,
        included_attributes=["age", "income_level"],
    )
    partial_prompt = conditioner.condition_prompt(
        profile, "Smart Watch", "Fitness tracking device", partial_config
    )

    print(f"Partial prompt length: {len(partial_prompt['user'])} chars")
    print("Included: age, income_level")
    print("Excluded: gender, location, ethnicity")

    # Test 4: Prompt consistency
    print("\n" + "=" * 60)
    print("4. Prompt Consistency Test:")

    consistency = conditioner.test_prompt_consistency(
        profile,
        "Smart Watch",
        "Fitness tracking device",
        num_repetitions=10,
    )

    print(f"Repetitions: {consistency['num_repetitions']}")
    print(f"All identical: {consistency['all_prompts_identical']}")
    print(f"Test result: {consistency['test_result']}")

    # Test 5: Attribute importance (simulation)
    print("\n" + "=" * 60)
    print("5. Attribute Importance Analysis (Simulation):")

    # Note: Would require actual LLM interface for full test
    # This demonstrates the structure
    print("Testing each attribute in isolation:")
    attributes = ["age", "gender", "income_level", "location", "ethnicity"]

    for attribute in attributes:
        config = ConditioningConfig(
            mode=ConditioningMode.PARTIAL_DEMOGRAPHICS, included_attributes=[attribute]
        )
        prompt = conditioner.condition_prompt(
            profile, "Smart Watch", "Fitness tracking device", config
        )
        print(f"  {attribute}: prompt length = {len(prompt['user'])} chars")

    print("\nExpected importance (from paper):")
    print("  1. Age (highest test-retest reliability)")
    print("  2. Income level (second highest)")
    print("  3. Gender")
    print("  4. Location")
    print("  5. Ethnicity")

    print("\n" + "=" * 60)
    print("Persona Conditioning testing complete")
    print("\nKey Findings from Paper:")
    print("- WITH demographics: ρ = 90.2% (90% of human reliability)")
    print("- WITHOUT demographics: ρ = 50% (random distribution matching)")
    print("- IMPROVEMENT: +40.2 percentage points from demographic conditioning")

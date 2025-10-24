"""
ABOUTME: Prompt engineering and template system with demographic conditioning
ABOUTME: Provides systematic prompt construction for realistic synthetic consumer responses

This module implements the critical demographic conditioning that enables
the SSR system to achieve 90% of human test-retest reliability.

Key Finding from Paper:
- WITH demographics: ρ = 90.2% (strong correlation to human responses)
- WITHOUT demographics: ρ = 50% (only distribution similarity)
- Demographic conditioning is ESSENTIAL for accurate results

Prompt Structure:
1. System Prompt: Role definition
2. Demographic Conditioning: Age, gender, income, location, ethnicity
3. Product Presentation: Name, description, image, price
4. Question Prompt: Purchase intent inquiry
5. Response Format: 1-3 sentences
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from pathlib import Path
import yaml


@dataclass
class PromptTemplate:
    """
    Template for constructing LLM prompts

    Attributes:
        id: Unique template identifier
        name: Human-readable template name
        domain: Domain/category (e.g., "consumer_products")
        language: Language code (e.g., "en")
        system_prompt: System-level instruction
        demographic_template: Template for demographic information
        product_template: Template for product presentation
        question_template: Template for question prompt
        response_format: Instructions for response format
        description: Optional template description
    """

    id: str
    name: str
    domain: str
    language: str
    system_prompt: str
    demographic_template: str
    product_template: str
    question_template: str
    response_format: str
    description: Optional[str] = None

    def format_prompt(
        self,
        demographic_attributes: Dict[str, Any],
        product_name: str,
        product_description: str,
        product_price: Optional[str] = None,
        product_image_url: Optional[str] = None,
        custom_question: Optional[str] = None,
    ) -> Dict[str, str]:
        """
        Format complete prompt from template

        Args:
            demographic_attributes: Demographic data
            product_name: Product name
            product_description: Product description
            product_price: Optional price
            product_image_url: Optional image URL
            custom_question: Optional custom question (uses template default if None)

        Returns:
            Dictionary with 'system' and 'user' prompts
        """
        # Format demographics
        demographics_text = self.demographic_template.format(
            age=demographic_attributes.get("age", "Unknown"),
            gender=demographic_attributes.get("gender", "Unknown"),
            income_level=demographic_attributes.get("income_level", "Unknown"),
            location=demographic_attributes.get("location", "Unknown"),
            ethnicity=demographic_attributes.get("ethnicity", "Unknown"),
        )

        # Format product
        product_text = self.product_template.format(
            product_name=product_name, product_description=product_description
        )

        if product_price:
            product_text += f"\nPrice: {product_price}"

        if product_image_url:
            product_text += f"\n[Product Image: {product_image_url}]"

        # Format question
        if custom_question:
            question_text = custom_question
        else:
            question_text = self.question_template

        # Combine user message
        user_message = f"""{demographics_text}

{product_text}

{question_text}

{self.response_format}"""

        return {"system": self.system_prompt, "user": user_message}


class PromptManager:
    """
    Manages prompt templates with persistence and validation

    Features:
    - Multiple template support for different domains
    - YAML-based storage
    - Template validation
    - Default templates from research paper
    """

    def __init__(self, templates_dir: Optional[Path] = None):
        """
        Initialize prompt manager

        Args:
            templates_dir: Directory for template storage
        """
        if templates_dir is None:
            templates_dir = (
                Path(__file__).parent.parent.parent / "config" / "prompt_templates"
            )

        self.templates_dir = Path(templates_dir)
        self.templates_dir.mkdir(parents=True, exist_ok=True)

        self._templates: Dict[str, PromptTemplate] = {}
        self._load_templates()

        # If no templates loaded, create default
        if not self._templates:
            self._create_default_template()

    def _load_templates(self):
        """Load templates from YAML files"""
        for yaml_file in self.templates_dir.glob("*.yaml"):
            try:
                template = self.load_from_yaml(yaml_file)
                self._templates[template.id] = template
            except Exception as e:
                print(f"Warning: Failed to load template {yaml_file}: {e}")

    def _create_default_template(self):
        """
        Create default template based on research paper methodology

        This is the prompt structure that achieved ρ = 90.2% with GPT-4o
        """
        template = PromptTemplate(
            id="paper_default",
            name="Research Paper Default Template",
            domain="consumer_products",
            language="en",
            description=(
                "Default template from research paper achieving 90% test-retest reliability. "
                "Uses strong demographic conditioning with natural product presentation."
            ),
            system_prompt=(
                "You are participating in a consumer research study. "
                "Please respond as yourself based on the demographic information provided."
            ),
            demographic_template="""Demographic Profile:
- Age: {age}
- Gender: {gender}
- Annual Household Income: {income_level}
- Location: {location}
- Ethnicity: {ethnicity}""",
            product_template="""Please examine the following product concept:

Product: {product_name}
{product_description}""",
            question_template=(
                "In a few sentences, please describe your feelings about purchasing "
                "this product. Focus on your likelihood of buying it and the reasons "
                "behind your view."
            ),
            response_format="Response (1-3 sentences):",
        )

        self._templates[template.id] = template
        self.save_to_yaml(template)

    def add_template(self, template: PromptTemplate):
        """
        Add new template

        Args:
            template: PromptTemplate to add

        Raises:
            ValueError: If template ID already exists
        """
        if template.id in self._templates:
            raise ValueError(f"Template with ID '{template.id}' already exists")

        self._templates[template.id] = template
        self.save_to_yaml(template)

    def get_template(self, template_id: str = "paper_default") -> PromptTemplate:
        """
        Get template by ID

        Args:
            template_id: Template identifier (default: paper_default)

        Returns:
            PromptTemplate

        Raises:
            KeyError: If template not found
        """
        if template_id not in self._templates:
            raise KeyError(f"Template '{template_id}' not found")

        return self._templates[template_id]

    def get_all_templates(self) -> List[PromptTemplate]:
        """Get all available templates"""
        return list(self._templates.values())

    def save_to_yaml(self, template: PromptTemplate):
        """
        Save template to YAML file

        Args:
            template: PromptTemplate to save
        """
        yaml_path = self.templates_dir / f"{template.id}.yaml"

        data = {
            "id": template.id,
            "name": template.name,
            "domain": template.domain,
            "language": template.language,
            "description": template.description,
            "system_prompt": template.system_prompt,
            "demographic_template": template.demographic_template,
            "product_template": template.product_template,
            "question_template": template.question_template,
            "response_format": template.response_format,
        }

        with open(yaml_path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    def load_from_yaml(self, yaml_path: Path) -> PromptTemplate:
        """
        Load template from YAML file

        Args:
            yaml_path: Path to YAML file

        Returns:
            PromptTemplate loaded from file
        """
        with open(yaml_path) as f:
            data = yaml.safe_load(f)

        return PromptTemplate(
            id=data["id"],
            name=data["name"],
            domain=data["domain"],
            language=data["language"],
            system_prompt=data["system_prompt"],
            demographic_template=data["demographic_template"],
            product_template=data["product_template"],
            question_template=data["question_template"],
            response_format=data["response_format"],
            description=data.get("description"),
        )

    def validate_template(self, template: PromptTemplate) -> Dict[str, bool]:
        """
        Validate template structure

        Checks:
        - Required fields present
        - Template variables correct
        - Language code valid

        Args:
            template: PromptTemplate to validate

        Returns:
            Dictionary of validation results
        """
        results = {
            "has_required_fields": all(
                [
                    template.id,
                    template.name,
                    template.domain,
                    template.language,
                    template.system_prompt,
                    template.demographic_template,
                    template.product_template,
                    template.question_template,
                    template.response_format,
                ]
            ),
            "valid_language_code": len(template.language) == 2,  # ISO 639-1
        }

        # Check demographic template has required variables
        required_demo_vars = [
            "{age}",
            "{gender}",
            "{income_level}",
            "{location}",
            "{ethnicity}",
        ]
        results["has_demographic_vars"] = all(
            var in template.demographic_template for var in required_demo_vars
        )

        # Check product template has required variables
        required_product_vars = ["{product_name}", "{product_description}"]
        results["has_product_vars"] = all(
            var in template.product_template for var in required_product_vars
        )

        results["all_valid"] = all(results.values())

        return results

    def get_statistics(self) -> Dict[str, Any]:
        """Get template statistics"""
        domains = set(t.domain for t in self._templates.values())
        languages = set(t.language for t in self._templates.values())

        return {
            "total_templates": len(self._templates),
            "domains": list(domains),
            "languages": list(languages),
        }


# Example usage and testing
if __name__ == "__main__":
    # Initialize prompt manager
    manager = PromptManager()

    print("Prompt Manager initialized")
    print(f"Loaded {len(manager.get_all_templates())} template(s)\n")

    # Get default template
    template = manager.get_template("paper_default")
    print(f"Template: {template.name}")
    print(f"Domain: {template.domain}")
    print(f"Language: {template.language}\n")

    # Test demographic attributes
    demographics = {
        "age": 28,
        "gender": "Female",
        "income_level": "$50,000-$75,000",
        "location": "San Francisco, CA, USA",
        "ethnicity": "Asian",
    }

    # Test product
    product_name = "Smart Fitness Watch Pro"
    product_description = (
        "Advanced fitness tracking watch with heart rate monitoring, "
        "GPS, sleep tracking, and 7-day battery life. "
        "Syncs with your smartphone for notifications and music control."
    )
    product_price = "$299"

    # Format prompt
    prompts = template.format_prompt(
        demographic_attributes=demographics,
        product_name=product_name,
        product_description=product_description,
        product_price=product_price,
    )

    print("=" * 60)
    print("FORMATTED PROMPT:")
    print("=" * 60)
    print("\nSYSTEM PROMPT:")
    print(prompts["system"])
    print("\nUSER PROMPT:")
    print(prompts["user"])

    # Validate template
    print("\n" + "=" * 60)
    print("TEMPLATE VALIDATION:")
    print("=" * 60)
    validation = manager.validate_template(template)
    for check, passed in validation.items():
        status = "✓" if passed else "✗"
        print(f"{status} {check}")

    # Show statistics
    print("\n" + "=" * 60)
    print("STATISTICS:")
    print("=" * 60)
    stats = manager.get_statistics()
    for key, value in stats.items():
        print(f"{key}: {value}")

"""
ABOUTME: Unit tests for LLM prompts module
ABOUTME: Tests PromptTemplate and PromptManager for demographic conditioning

This test module covers:
1. PromptTemplate dataclass and format_prompt method
2. PromptManager template storage and retrieval
3. Template validation and persistence
4. Paper-compliant prompt structure

The prompts module is critical for achieving 90% test-retest reliability through
proper demographic conditioning.
"""

import pytest
import tempfile
from pathlib import Path

from src.llm.prompts import PromptTemplate, PromptManager


# ============================================================================
# Test PromptTemplate Dataclass
# ============================================================================


class TestPromptTemplate:
    """Test PromptTemplate dataclass"""

    def test_basic_template_creation(self):
        """Test creating basic prompt template"""
        template = PromptTemplate(
            id="test_template",
            name="Test Template",
            domain="consumer_products",
            language="en",
            system_prompt="You are a consumer.",
            demographic_template="Age: {age}, Gender: {gender}, Income: {income_level}, Location: {location}, Ethnicity: {ethnicity}",
            product_template="Product: {product_name}\n{product_description}",
            question_template="Would you buy this product?",
            response_format="Response (1-3 sentences):",
        )

        assert template.id == "test_template"
        assert template.name == "Test Template"
        assert template.domain == "consumer_products"
        assert template.language == "en"
        assert template.description is None

    def test_template_with_description(self):
        """Test template with optional description"""
        template = PromptTemplate(
            id="desc_template",
            name="Template with Description",
            domain="test",
            language="en",
            system_prompt="Test",
            demographic_template="Test",
            product_template="Test",
            question_template="Test",
            response_format="Test",
            description="This is a test template",
        )

        assert template.description == "This is a test template"

    def test_format_prompt_basic(self):
        """Test formatting prompt with basic demographic and product data"""
        template = PromptTemplate(
            id="test",
            name="Test",
            domain="test",
            language="en",
            system_prompt="You are a consumer.",
            demographic_template="Age: {age}, Income: {income_level}",
            product_template="Product: {product_name}\n{product_description}",
            question_template="Would you buy?",
            response_format="Answer:",
        )

        demographics = {
            "age": 30,
            "gender": "Male",
            "income_level": "$50,000-$75,000",
            "location": "NYC",
            "ethnicity": "White",
        }

        prompts = template.format_prompt(
            demographic_attributes=demographics,
            product_name="Smart Watch",
            product_description="Fitness tracker",
        )

        assert "system" in prompts
        assert "user" in prompts
        assert prompts["system"] == "You are a consumer."
        assert "Age: 30" in prompts["user"]
        assert "$50,000-$75,000" in prompts["user"]
        assert "Smart Watch" in prompts["user"]
        assert "Fitness tracker" in prompts["user"]

    def test_format_prompt_with_price(self):
        """Test formatting prompt with product price"""
        template = PromptTemplate(
            id="test",
            name="Test",
            domain="test",
            language="en",
            system_prompt="System",
            demographic_template="{age}",
            product_template="{product_name}: {product_description}",
            question_template="Question",
            response_format="Answer",
        )

        prompts = template.format_prompt(
            demographic_attributes={"age": 25},
            product_name="Product",
            product_description="Description",
            product_price="$99",
        )

        assert "Price: $99" in prompts["user"]

    def test_format_prompt_with_image_url(self):
        """Test formatting prompt with product image URL"""
        template = PromptTemplate(
            id="test",
            name="Test",
            domain="test",
            language="en",
            system_prompt="System",
            demographic_template="{age}",
            product_template="{product_name}",
            question_template="Q",
            response_format="A",
        )

        prompts = template.format_prompt(
            demographic_attributes={"age": 30},
            product_name="Product",
            product_description="Desc",
            product_image_url="https://example.com/image.jpg",
        )

        assert "[Product Image: https://example.com/image.jpg]" in prompts["user"]

    def test_format_prompt_custom_question(self):
        """Test formatting prompt with custom question"""
        template = PromptTemplate(
            id="test",
            name="Test",
            domain="test",
            language="en",
            system_prompt="S",
            demographic_template="{age}",
            product_template="{product_name}",
            question_template="Default question?",
            response_format="Answer:",
        )

        prompts = template.format_prompt(
            demographic_attributes={"age": 40},
            product_name="Product",
            product_description="Desc",
            custom_question="Custom question?",
        )

        assert "Custom question?" in prompts["user"]
        assert "Default question?" not in prompts["user"]

    def test_format_prompt_missing_demographic_values(self):
        """Test formatting with missing demographic values (should use 'Unknown')"""
        template = PromptTemplate(
            id="test",
            name="Test",
            domain="test",
            language="en",
            system_prompt="S",
            demographic_template="Age: {age}, Gender: {gender}",
            product_template="{product_name}",
            question_template="Q",
            response_format="A",
        )

        # Missing gender attribute
        demographics = {"age": 35}

        prompts = template.format_prompt(
            demographic_attributes=demographics,
            product_name="Product",
            product_description="Desc",
        )

        assert "Age: 35" in prompts["user"]
        assert "Gender: Unknown" in prompts["user"]


# ============================================================================
# Test PromptManager
# ============================================================================


class TestPromptManager:
    """Test PromptManager template storage and retrieval"""

    def test_prompt_manager_initialization(self):
        """Test creating PromptManager with temporary directory"""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = PromptManager(templates_dir=Path(tmpdir))

            assert manager.templates_dir == Path(tmpdir)
            # Should have default template created
            assert len(manager.get_all_templates()) >= 1

    def test_default_template_created(self):
        """Test that default paper template is created automatically"""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = PromptManager(templates_dir=Path(tmpdir))

            # Should have paper_default template
            template = manager.get_template("paper_default")

            assert template.id == "paper_default"
            assert template.name == "Research Paper Default Template"
            assert template.domain == "consumer_products"
            assert template.language == "en"

    def test_paper_default_template_structure(self):
        """Test that paper_default template has correct structure"""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = PromptManager(templates_dir=Path(tmpdir))
            template = manager.get_template("paper_default")

            # Check system prompt
            assert "consumer research study" in template.system_prompt.lower()

            # Check demographic template has all 5 attributes
            demo_template = template.demographic_template
            assert "{age}" in demo_template
            assert "{gender}" in demo_template
            assert "{income_level}" in demo_template
            assert "{location}" in demo_template
            assert "{ethnicity}" in demo_template

            # Check product template
            assert "{product_name}" in template.product_template
            assert "{product_description}" in template.product_template

            # Check question asks about purchase feelings
            assert (
                "feelings" in template.question_template.lower()
                or "likelihood" in template.question_template.lower()
            )

    def test_add_template(self):
        """Test adding new template"""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = PromptManager(templates_dir=Path(tmpdir))

            new_template = PromptTemplate(
                id="custom_template",
                name="Custom Template",
                domain="electronics",
                language="en",
                system_prompt="Custom system",
                demographic_template="{age}",
                product_template="{product_name}",
                question_template="Q",
                response_format="A",
            )

            manager.add_template(new_template)

            # Should be able to retrieve it
            retrieved = manager.get_template("custom_template")
            assert retrieved.id == "custom_template"
            assert retrieved.name == "Custom Template"

    def test_add_duplicate_template_raises_error(self):
        """Test that adding template with duplicate ID raises ValueError"""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = PromptManager(templates_dir=Path(tmpdir))

            template = PromptTemplate(
                id="paper_default",  # Duplicate ID
                name="Duplicate",
                domain="test",
                language="en",
                system_prompt="S",
                demographic_template="D",
                product_template="P",
                question_template="Q",
                response_format="R",
            )

            with pytest.raises(ValueError, match="already exists"):
                manager.add_template(template)

    def test_get_template_not_found(self):
        """Test that getting non-existent template raises KeyError"""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = PromptManager(templates_dir=Path(tmpdir))

            with pytest.raises(KeyError, match="not found"):
                manager.get_template("nonexistent_template")

    def test_get_all_templates(self):
        """Test getting all templates"""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = PromptManager(templates_dir=Path(tmpdir))

            templates = manager.get_all_templates()

            assert len(templates) >= 1
            assert any(t.id == "paper_default" for t in templates)

    def test_save_and_load_template(self):
        """Test saving template to YAML and loading it back"""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = PromptManager(templates_dir=Path(tmpdir))

            template = PromptTemplate(
                id="yaml_test",
                name="YAML Test",
                domain="test",
                language="en",
                description="Test description",
                system_prompt="System",
                demographic_template="{age}, {gender}",
                product_template="{product_name}: {product_description}",
                question_template="Would you buy?",
                response_format="Answer (1-3 sentences):",
            )

            manager.save_to_yaml(template)

            # Load it back
            yaml_path = Path(tmpdir) / "yaml_test.yaml"
            loaded = manager.load_from_yaml(yaml_path)

            assert loaded.id == template.id
            assert loaded.name == template.name
            assert loaded.domain == template.domain
            assert loaded.language == template.language
            assert loaded.description == template.description
            assert loaded.system_prompt == template.system_prompt

    def test_validate_template_valid(self):
        """Test validating valid template"""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = PromptManager(templates_dir=Path(tmpdir))
            template = manager.get_template("paper_default")

            validation = manager.validate_template(template)

            assert validation["has_required_fields"] is True
            assert validation["valid_language_code"] is True
            assert validation["has_demographic_vars"] is True
            assert validation["has_product_vars"] is True
            assert validation["all_valid"] is True

    def test_validate_template_missing_demographic_vars(self):
        """Test validation fails when demographic vars are missing"""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = PromptManager(templates_dir=Path(tmpdir))

            template = PromptTemplate(
                id="invalid",
                name="Invalid",
                domain="test",
                language="en",
                system_prompt="S",
                demographic_template="Age: {age}",  # Missing other vars
                product_template="{product_name}: {product_description}",
                question_template="Q",
                response_format="R",
            )

            validation = manager.validate_template(template)

            assert validation["has_demographic_vars"] is False
            assert validation["all_valid"] is False

    def test_validate_template_invalid_language_code(self):
        """Test validation fails with invalid language code"""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = PromptManager(templates_dir=Path(tmpdir))

            template = PromptTemplate(
                id="invalid_lang",
                name="Invalid Language",
                domain="test",
                language="eng",  # Should be 2 characters
                system_prompt="S",
                demographic_template="{age}, {gender}, {income_level}, {location}, {ethnicity}",
                product_template="{product_name}: {product_description}",
                question_template="Q",
                response_format="R",
            )

            validation = manager.validate_template(template)

            assert validation["valid_language_code"] is False
            assert validation["all_valid"] is False

    def test_get_statistics(self):
        """Test getting template statistics"""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = PromptManager(templates_dir=Path(tmpdir))

            # Add a second template with different domain and language
            manager.add_template(
                PromptTemplate(
                    id="spanish_template",
                    name="Spanish Template",
                    domain="electronics",
                    language="es",
                    system_prompt="S",
                    demographic_template="D",
                    product_template="P",
                    question_template="Q",
                    response_format="R",
                )
            )

            stats = manager.get_statistics()

            assert stats["total_templates"] >= 2
            assert "consumer_products" in stats["domains"]
            assert "electronics" in stats["domains"]
            assert "en" in stats["languages"]
            assert "es" in stats["languages"]


# ============================================================================
# Test Paper Methodology Compliance
# ============================================================================


class TestPaperMethodology:
    """Test compliance with research paper methodology"""

    def test_paper_template_includes_all_five_demographics(self):
        """Test that paper template includes all 5 demographic attributes"""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = PromptManager(templates_dir=Path(tmpdir))
            template = manager.get_template("paper_default")

            demographics = {
                "age": 28,
                "gender": "Female",
                "income_level": "$50,000-$75,000",
                "location": "San Francisco, CA",
                "ethnicity": "Asian",
            }

            prompts = template.format_prompt(
                demographic_attributes=demographics,
                product_name="Test Product",
                product_description="Test Description",
            )

            user_prompt = prompts["user"]

            # All 5 demographics should appear in prompt
            assert "28" in user_prompt
            assert "Female" in user_prompt
            assert "$50,000-$75,000" in user_prompt
            assert "San Francisco, CA" in user_prompt
            assert "Asian" in user_prompt

    def test_paper_template_response_format_specifies_length(self):
        """Test that paper template specifies response length (1-3 sentences)"""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = PromptManager(templates_dir=Path(tmpdir))
            template = manager.get_template("paper_default")

            # Response format should mention sentence count
            assert (
                "1-3" in template.response_format
                or "sentence" in template.response_format.lower()
            )

    def test_paper_template_asks_about_purchase_feelings(self):
        """Test that paper template asks about feelings/likelihood of purchase"""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = PromptManager(templates_dir=Path(tmpdir))
            template = manager.get_template("paper_default")

            question = template.question_template.lower()

            # Should ask about feelings and likelihood
            assert "feelings" in question or "feel" in question
            assert (
                "purchasing" in question
                or "buy" in question
                or "likelihood" in question
            )

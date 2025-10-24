"""
ABOUTME: Unit tests for LLM validation module
ABOUTME: Tests ResponseValidator for quality control of LLM-generated responses

This test module covers:
1. ValidationIssue enum values
2. ValidationResult dataclass
3. ResponseValidator validation logic
4. Retry prompt adjustments
5. Paper methodology compliance

The validation module ensures responses meet quality standards before SSR processing.
"""

from src.llm.validation import (
    ValidationIssue,
    ValidationResult,
    ResponseValidator,
)


# ============================================================================
# Test ValidationIssue Enum
# ============================================================================


class TestValidationIssue:
    """Test ValidationIssue enum"""

    def test_validation_issue_values(self):
        """Test all ValidationIssue enum values"""
        issues = [
            ValidationIssue.TOO_SHORT,
            ValidationIssue.TOO_LONG,
            ValidationIssue.NO_OPINION,
            ValidationIssue.META_COMMENTARY,
            ValidationIssue.OFF_TOPIC,
            ValidationIssue.CONTRADICTORY,
            ValidationIssue.INCOMPLETE,
            ValidationIssue.LANGUAGE_ERROR,
        ]

        assert len(issues) == 8

    def test_validation_issue_string_values(self):
        """Test ValidationIssue string representations"""
        assert ValidationIssue.TOO_SHORT.value == "too_short"
        assert ValidationIssue.TOO_LONG.value == "too_long"
        assert ValidationIssue.NO_OPINION.value == "no_opinion"
        assert ValidationIssue.META_COMMENTARY.value == "meta_commentary"


# ============================================================================
# Test ValidationResult Dataclass
# ============================================================================


class TestValidationResult:
    """Test ValidationResult dataclass"""

    def test_basic_validation_result(self):
        """Test creating basic validation result"""
        result = ValidationResult(
            valid=True,
            issues=[],
            word_count=25,
            sentence_count=2,
            confidence_score=1.0,
            suggestions=[],
        )

        assert result.valid is True
        assert len(result.issues) == 0
        assert result.word_count == 25
        assert result.sentence_count == 2
        assert result.confidence_score == 1.0

    def test_validation_result_with_issues(self):
        """Test validation result with issues"""
        result = ValidationResult(
            valid=False,
            issues=[ValidationIssue.TOO_SHORT, ValidationIssue.NO_OPINION],
            word_count=5,
            sentence_count=1,
            confidence_score=0.3,
            suggestions=["Add more detail", "Express opinion"],
        )

        assert result.valid is False
        assert len(result.issues) == 2
        assert ValidationIssue.TOO_SHORT in result.issues
        assert len(result.suggestions) == 2

    def test_get_primary_issue(self):
        """Test getting primary (first) issue"""
        result = ValidationResult(
            valid=False,
            issues=[ValidationIssue.TOO_LONG, ValidationIssue.CONTRADICTORY],
            word_count=150,
            sentence_count=5,
            confidence_score=0.5,
            suggestions=[],
        )

        primary = result.get_primary_issue()
        assert primary == ValidationIssue.TOO_LONG

    def test_get_primary_issue_no_issues(self):
        """Test getting primary issue when there are no issues"""
        result = ValidationResult(
            valid=True,
            issues=[],
            word_count=30,
            sentence_count=2,
            confidence_score=1.0,
            suggestions=[],
        )

        primary = result.get_primary_issue()
        assert primary is None


# ============================================================================
# Test ResponseValidator
# ============================================================================


class TestResponseValidator:
    """Test ResponseValidator validation logic"""

    def test_validator_initialization(self):
        """Test creating ResponseValidator with defaults"""
        validator = ResponseValidator()

        assert validator.min_words == 10
        assert validator.max_words == 100
        assert validator.min_sentences == 1
        assert validator.max_sentences == 5

    def test_validator_custom_initialization(self):
        """Test creating ResponseValidator with custom thresholds"""
        validator = ResponseValidator(
            min_words=15, max_words=80, min_sentences=2, max_sentences=4
        )

        assert validator.min_words == 15
        assert validator.max_words == 80
        assert validator.min_sentences == 2
        assert validator.max_sentences == 4

    def test_validate_valid_response(self):
        """Test validating valid response"""
        validator = ResponseValidator()

        response = (
            "I really like this fitness watch and would probably buy it. "
            "It has all the features I need for my daily workouts. "
            "The price seems reasonable for the quality offered."
        )

        result = validator.validate(response, product_name="Fitness Watch")

        assert result.valid is True
        assert len(result.issues) == 0
        assert result.word_count > 10
        assert result.sentence_count >= 1
        assert result.confidence_score == 1.0

    def test_validate_too_short(self):
        """Test validating response that is too short"""
        validator = ResponseValidator()

        response = "Yes."

        result = validator.validate(response)

        assert result.valid is False
        assert ValidationIssue.TOO_SHORT in result.issues
        assert result.word_count < 10
        assert result.confidence_score < 1.0
        assert any("too short" in s.lower() for s in result.suggestions)

    def test_validate_too_long(self):
        """Test validating response that is too long"""
        validator = ResponseValidator()

        # Create a very long response (>100 words)
        response = " ".join(["word"] * 120)

        result = validator.validate(response)

        assert result.valid is False
        assert ValidationIssue.TOO_LONG in result.issues
        assert result.word_count > 100
        assert any("too long" in s.lower() for s in result.suggestions)

    def test_validate_meta_commentary(self):
        """Test validating response with meta-commentary"""
        validator = ResponseValidator()

        response = (
            "As an AI language model, I cannot provide personal opinions about "
            "products. However, based on the features described, some consumers "
            "might find this appealing."
        )

        result = validator.validate(response)

        assert result.valid is False
        assert ValidationIssue.META_COMMENTARY in result.issues
        assert result.confidence_score < 0.5
        assert any("real consumer" in s.lower() for s in result.suggestions)

    def test_validate_no_opinion(self):
        """Test validating response without clear opinion"""
        validator = ResponseValidator()

        response = (
            "This is a product with many features. "
            "It comes in different colors. "
            "The specifications are listed on the website."
        )

        result = validator.validate(response)

        assert result.valid is False
        assert ValidationIssue.NO_OPINION in result.issues
        assert any("opinion" in s.lower() for s in result.suggestions)

    def test_validate_contradictory(self):
        """Test validating contradictory response"""
        validator = ResponseValidator()

        # Contradiction within same sentence
        response = (
            "I really love this product but I don't want to buy it and it's terrible."
        )

        result = validator.validate(response)

        assert result.valid is False
        assert ValidationIssue.CONTRADICTORY in result.issues
        assert any("contradictory" in s.lower() for s in result.suggestions)

    def test_validate_poor_language_quality(self):
        """Test validating response with poor language quality"""
        validator = ResponseValidator()

        response = "this looks good maybe buy it yes"

        result = validator.validate(response)

        assert result.valid is False
        assert ValidationIssue.LANGUAGE_ERROR in result.issues
        assert any("grammatically correct" in s.lower() for s in result.suggestions)

    def test_validate_empty_response(self):
        """Test validating empty response"""
        validator = ResponseValidator()

        result = validator.validate("")

        assert result.valid is False
        assert ValidationIssue.TOO_SHORT in result.issues
        assert result.word_count == 0
        assert result.sentence_count == 0
        assert result.confidence_score == 1.0

    def test_validate_with_product_name_on_topic(self):
        """Test validation with product name mentioned (on topic)"""
        validator = ResponseValidator()

        response = (
            "I really like the EcoBottle design. "
            "It's practical and environmentally friendly. "
            "I would definitely purchase it."
        )

        result = validator.validate(response, product_name="EcoBottle")

        assert result.valid is True
        assert ValidationIssue.OFF_TOPIC not in result.issues

    def test_validate_with_product_name_off_topic(self):
        """Test validation detects off-topic response"""
        validator = ResponseValidator()

        response = (
            "My grandmother used to make delicious cookies. "
            "The weather is nice today. "
            "Baseball is an interesting sport."
        )

        result = validator.validate(response, product_name="Smart Watch")

        assert result.valid is False
        assert ValidationIssue.OFF_TOPIC in result.issues

    def test_get_retry_prompt_adjustment_too_short(self):
        """Test getting retry prompt adjustment for too short response"""
        validator = ResponseValidator()

        result = ValidationResult(
            valid=False,
            issues=[ValidationIssue.TOO_SHORT],
            word_count=5,
            sentence_count=1,
            confidence_score=0.3,
            suggestions=["Too short"],
        )

        adjustment = validator.get_retry_prompt_adjustment(result)

        assert "more detail" in adjustment.lower()
        assert "2-3 sentences" in adjustment.lower()

    def test_get_retry_prompt_adjustment_too_long(self):
        """Test getting retry prompt adjustment for too long response"""
        validator = ResponseValidator()

        result = ValidationResult(
            valid=False,
            issues=[ValidationIssue.TOO_LONG],
            word_count=150,
            sentence_count=8,
            confidence_score=0.7,
            suggestions=["Too long"],
        )

        adjustment = validator.get_retry_prompt_adjustment(result)

        assert "concise" in adjustment.lower()
        assert "1-3 sentences" in adjustment.lower()

    def test_get_retry_prompt_adjustment_meta_commentary(self):
        """Test getting retry prompt adjustment for meta-commentary"""
        validator = ResponseValidator()

        result = ValidationResult(
            valid=False,
            issues=[ValidationIssue.META_COMMENTARY],
            word_count=30,
            sentence_count=2,
            confidence_score=0.1,
            suggestions=["No meta"],
        )

        adjustment = validator.get_retry_prompt_adjustment(result)

        assert "real consumer" in adjustment.lower()
        assert "personal opinions" in adjustment.lower()

    def test_get_retry_prompt_adjustment_no_opinion(self):
        """Test getting retry prompt adjustment for no opinion"""
        validator = ResponseValidator()

        result = ValidationResult(
            valid=False,
            issues=[ValidationIssue.NO_OPINION],
            word_count=25,
            sentence_count=2,
            confidence_score=0.4,
            suggestions=["Add opinion"],
        )

        adjustment = validator.get_retry_prompt_adjustment(result)

        assert "opinion" in adjustment.lower()
        assert "buy" in adjustment.lower() or "purchase" in adjustment.lower()

    def test_get_retry_prompt_adjustment_valid(self):
        """Test getting retry prompt adjustment for valid response (should be empty)"""
        validator = ResponseValidator()

        result = ValidationResult(
            valid=True,
            issues=[],
            word_count=30,
            sentence_count=2,
            confidence_score=1.0,
            suggestions=[],
        )

        adjustment = validator.get_retry_prompt_adjustment(result)

        assert adjustment == ""


# ============================================================================
# Test Edge Cases
# ============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions"""

    def test_validate_exactly_min_words(self):
        """Test response with exactly minimum word count"""
        validator = ResponseValidator(min_words=10)

        # Exactly 10 words
        response = (
            "This product seems interesting and I would consider buying it for sure."
        )

        result = validator.validate(response)

        # Should pass (>=10 words)
        assert ValidationIssue.TOO_SHORT not in result.issues

    def test_validate_exactly_max_words(self):
        """Test response with exactly maximum word count"""
        validator = ResponseValidator(max_words=20)

        # Exactly 20 words
        response = " ".join(["word"] * 20)

        result = validator.validate(response)

        # Should pass (<=20 words)
        assert ValidationIssue.TOO_LONG not in result.issues

    def test_validate_whitespace_only(self):
        """Test response with only whitespace"""
        validator = ResponseValidator()

        result = validator.validate("   \n\t   ")

        assert result.valid is False
        assert result.word_count == 0

    def test_validate_multiple_issues(self):
        """Test response with multiple validation issues"""
        validator = ResponseValidator()

        response = "yes"  # Too short, no opinion, poor language

        result = validator.validate(response)

        assert result.valid is False
        assert len(result.issues) > 1


# ============================================================================
# Test Paper Methodology Compliance
# ============================================================================


class TestPaperMethodology:
    """Test compliance with research paper methodology"""

    def test_default_word_limits_match_paper(self):
        """Test that default word limits match paper methodology (10-100 words)"""
        validator = ResponseValidator()

        assert validator.min_words == 10
        assert validator.max_words == 100

    def test_validates_typical_paper_response(self):
        """Test that typical paper responses pass validation"""
        validator = ResponseValidator()

        # Typical response from paper (1-3 sentences, expresses purchase intent)
        response = (
            "I find this fitness tracker appealing because it has all the features "
            "I need for my workouts. The price of $299 is reasonable for the quality. "
            "I would probably buy it."
        )

        result = validator.validate(response, product_name="fitness tracker")

        assert result.valid is True
        assert result.word_count >= 10
        assert result.word_count <= 100
        assert result.sentence_count >= 1
        assert result.sentence_count <= 5

    def test_detects_ai_meta_commentary(self):
        """Test that validator detects AI meta-commentary (critical for paper validity)"""
        validator = ResponseValidator()

        meta_responses = [
            "As an AI, I cannot provide personal opinions.",
            "I can't give a personal opinion on this product.",
            "I'm unable to express feelings about products.",
            "As a language model, I don't have preferences.",
        ]

        for response in meta_responses:
            result = validator.validate(response)
            assert ValidationIssue.META_COMMENTARY in result.issues

    def test_requires_purchase_intent_expression(self):
        """Test that validator requires purchase intent/opinion expression"""
        validator = ResponseValidator()

        # Valid responses with purchase intent
        valid_responses = [
            "I would definitely buy this product.",
            "I really like this and plan to purchase it.",
            "This appeals to me and I'd consider buying it.",
            "I'm not interested in purchasing this product.",
        ]

        for response in valid_responses:
            result = validator.validate(response)
            # Should have opinion indicators, so NO_OPINION should not be present
            assert ValidationIssue.NO_OPINION not in result.issues

"""
ABOUTME: Response validation and quality control for LLM-generated synthetic consumer responses
ABOUTME: Ensures responses meet quality standards before SSR processing with retry logic

This module validates that LLM responses are suitable for SSR processing by checking:
1. Length constraints (10-100 words, typically 1-3 sentences)
2. Content relevance (expresses purchase intent/opinion)
3. Sentiment coherence (no contradictions)
4. Language quality (grammatically correct, complete sentences)

Validation is critical because poor-quality responses lead to unreliable SSR distributions.
The retry strategy automatically improves responses that don't meet standards.
"""

import re
from typing import List, Optional
from dataclasses import dataclass
from enum import Enum


class ValidationIssue(Enum):
    """Types of validation issues"""

    TOO_SHORT = "too_short"
    TOO_LONG = "too_long"
    NO_OPINION = "no_opinion"
    META_COMMENTARY = "meta_commentary"
    OFF_TOPIC = "off_topic"
    CONTRADICTORY = "contradictory"
    INCOMPLETE = "incomplete"
    LANGUAGE_ERROR = "language_error"


@dataclass
class ValidationResult:
    """
    Result of response validation

    Attributes:
        valid: Whether response passed all checks
        issues: List of validation issues found
        word_count: Number of words in response
        sentence_count: Number of sentences
        confidence_score: Confidence in validation (0.0-1.0)
        suggestions: List of suggestions for improvement
    """

    valid: bool
    issues: List[ValidationIssue]
    word_count: int
    sentence_count: int
    confidence_score: float
    suggestions: List[str]

    def get_primary_issue(self) -> Optional[ValidationIssue]:
        """Get the most critical issue"""
        return self.issues[0] if self.issues else None


class ResponseValidator:
    """
    Validates LLM-generated synthetic consumer responses

    Features:
    - Multiple validation checks
    - Confidence scoring
    - Actionable suggestions for retry
    - Configurable thresholds
    """

    def __init__(
        self,
        min_words: int = 10,
        max_words: int = 100,
        min_sentences: int = 1,
        max_sentences: int = 5,
    ):
        """
        Initialize response validator

        Args:
            min_words: Minimum word count (default: 10)
            max_words: Maximum word count (default: 100)
            min_sentences: Minimum sentence count (default: 1)
            max_sentences: Maximum sentence count (default: 5)
        """
        self.min_words = min_words
        self.max_words = max_words
        self.min_sentences = min_sentences
        self.max_sentences = max_sentences

        # Patterns for meta-commentary detection
        self.meta_patterns = [
            r"as an ai",
            r"i cannot",
            r"i can'?t",
            r"i'm? unable to",
            r"i don'?t have",
            r"language model",
            r"artificial intelligence",
        ]

        # Opinion indicators (should be present)
        self.opinion_indicators = [
            r"\bi (?:would|wouldn't|might|could|should|think|feel|believe|like|love|hate|prefer)",
            r"(?:likely|unlikely|probable|possible)",
            r"(?:buy|purchase|get|acquire)",
            r"(?:interested|appeal|value|worth)",
            r"(?:good|bad|great|terrible|excellent|poor)",
        ]

    def validate(
        self, response_text: str, product_name: Optional[str] = None
    ) -> ValidationResult:
        """
        Validate response text

        Args:
            response_text: Response text to validate
            product_name: Optional product name for relevance checking

        Returns:
            ValidationResult with validation outcome and suggestions
        """
        if not response_text or not response_text.strip():
            return ValidationResult(
                valid=False,
                issues=[ValidationIssue.TOO_SHORT],
                word_count=0,
                sentence_count=0,
                confidence_score=1.0,
                suggestions=["Response is empty. Please provide a response."],
            )

        response_text = response_text.strip()

        # Count words and sentences
        word_count = len(response_text.split())
        sentences = self._split_sentences(response_text)
        sentence_count = len(sentences)

        issues = []
        suggestions = []
        confidence_score = 1.0

        # Check 1: Length validation
        if word_count < self.min_words:
            issues.append(ValidationIssue.TOO_SHORT)
            suggestions.append(
                f"Response is too short ({word_count} words). "
                f"Please elaborate briefly (aim for {self.min_words}-{self.max_words} words)."
            )
            confidence_score *= 0.3

        if word_count > self.max_words:
            issues.append(ValidationIssue.TOO_LONG)
            suggestions.append(
                f"Response is too long ({word_count} words). "
                f"Please be more concise (aim for {self.min_words}-{self.max_words} words)."
            )
            confidence_score *= 0.7

        # Check 2: Sentence count
        if sentence_count < self.min_sentences:
            issues.append(ValidationIssue.INCOMPLETE)
            suggestions.append("Please provide at least one complete sentence.")
            confidence_score *= 0.5

        # Check 3: Meta-commentary detection
        lower_text = response_text.lower()
        for pattern in self.meta_patterns:
            if re.search(pattern, lower_text):
                issues.append(ValidationIssue.META_COMMENTARY)
                suggestions.append(
                    "Please respond as a real consumer, not as an AI. "
                    "Express your personal opinion about the product."
                )
                confidence_score *= 0.1
                break

        # Check 4: Opinion detection
        has_opinion = any(
            re.search(pattern, lower_text, re.IGNORECASE)
            for pattern in self.opinion_indicators
        )
        if not has_opinion and word_count >= self.min_words:
            issues.append(ValidationIssue.NO_OPINION)
            suggestions.append(
                "Please express your opinion about purchasing the product. "
                "Include your likelihood of buying and reasons."
            )
            confidence_score *= 0.4

        # Check 5: Product relevance (if product name provided)
        if product_name and word_count >= self.min_words:
            # Very basic check - product name or category should be mentioned
            product_mentioned = product_name.lower() in lower_text or any(
                word in lower_text for word in ["product", "item", "this", "it"]
            )
            if not product_mentioned:
                issues.append(ValidationIssue.OFF_TOPIC)
                suggestions.append(
                    "Please focus your response on the product being evaluated."
                )
                confidence_score *= 0.6

        # Check 6: Contradiction detection (basic)
        contradictions = self._detect_contradictions(response_text)
        if contradictions:
            issues.append(ValidationIssue.CONTRADICTORY)
            suggestions.append(
                "Response contains contradictory statements. "
                "Please provide a consistent opinion."
            )
            confidence_score *= 0.5

        # Check 7: Language quality (basic checks)
        if not self._check_language_quality(response_text):
            issues.append(ValidationIssue.LANGUAGE_ERROR)
            suggestions.append("Please use complete, grammatically correct sentences.")
            confidence_score *= 0.6

        # Determine if valid
        valid = len(issues) == 0

        return ValidationResult(
            valid=valid,
            issues=issues,
            word_count=word_count,
            sentence_count=sentence_count,
            confidence_score=confidence_score,
            suggestions=suggestions,
        )

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Simple sentence splitter (periods, exclamation marks, question marks)
        sentences = re.split(r"[.!?]+", text)
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences

    def _detect_contradictions(self, text: str) -> bool:
        """
        Detect obvious contradictions in text

        Returns:
            True if contradictions detected, False otherwise
        """
        lower_text = text.lower()

        # Check for "yes but no" patterns
        positive_negative_patterns = [
            (
                r"\b(?:yes|like|love|want|would buy)\b",
                r"\b(?:no|don't|won't|wouldn't)\b",
            ),
            (r"\b(?:good|great|excellent)\b", r"\b(?:bad|terrible|poor)\b"),
            (r"\b(?:likely|probable)\b", r"\b(?:unlikely|improbable)\b"),
        ]

        for pos_pattern, neg_pattern in positive_negative_patterns:
            has_positive = re.search(pos_pattern, lower_text)
            has_negative = re.search(neg_pattern, lower_text)

            if has_positive and has_negative:
                # Check if they're in same sentence (contradiction)
                sentences = self._split_sentences(text)
                for sentence in sentences:
                    sent_lower = sentence.lower()
                    if re.search(pos_pattern, sent_lower) and re.search(
                        neg_pattern, sent_lower
                    ):
                        return True

        return False

    def _check_language_quality(self, text: str) -> bool:
        """
        Basic language quality checks

        Returns:
            True if language quality acceptable, False otherwise
        """
        # Check for complete sentences (should start with capital, end with punctuation)
        sentences = self._split_sentences(text)

        if not sentences:
            return False

        for sentence in sentences:
            # Should start with capital letter
            if not sentence[0].isupper():
                return False

            # Should have reasonable length (not just 1-2 words)
            if len(sentence.split()) < 3:
                return False

        return True

    def get_retry_prompt_adjustment(self, validation_result: ValidationResult) -> str:
        """
        Get prompt adjustment for retry based on validation issues

        Args:
            validation_result: ValidationResult from previous attempt

        Returns:
            String with additional instructions for retry
        """
        if validation_result.valid:
            return ""

        adjustments = []

        if ValidationIssue.TOO_SHORT in validation_result.issues:
            adjustments.append("Please provide more detail (at least 2-3 sentences).")

        if ValidationIssue.TOO_LONG in validation_result.issues:
            adjustments.append("Please be more concise (1-3 sentences maximum).")

        if ValidationIssue.META_COMMENTARY in validation_result.issues:
            adjustments.append(
                "IMPORTANT: Respond as a real consumer with personal opinions, "
                "not as an AI assistant."
            )

        if ValidationIssue.NO_OPINION in validation_result.issues:
            adjustments.append(
                "Express your clear opinion about whether you would buy this product "
                "and why."
            )

        if ValidationIssue.CONTRADICTORY in validation_result.issues:
            adjustments.append("Provide a consistent, non-contradictory opinion.")

        return " ".join(adjustments)


# Example usage and testing
if __name__ == "__main__":
    validator = ResponseValidator()

    print("Response Validator Testing")
    print("=" * 60)

    # Test cases
    test_responses = [
        # Valid responses
        (
            "I really like this fitness watch. It has all the features I need "
            "and the price seems reasonable. I'd probably buy it.",
            True,
            "Valid positive response",
        ),
        (
            "This product doesn't appeal to me. I already have a similar device "
            "and don't see enough value to upgrade.",
            True,
            "Valid negative response",
        ),
        # Invalid responses
        ("Yes.", False, "Too short"),
        (
            "As an AI language model, I cannot provide personal opinions about "
            "products. However, based on the features described, some consumers "
            "might find this appealing.",
            False,
            "Meta-commentary",
        ),
        (
            "I really love this product and think it's great! However, I hate it "
            "and would never buy it.",
            False,
            "Contradictory",
        ),
        (
            "this looks good maybe buy it yes",
            False,
            "Poor language quality",
        ),
    ]

    for i, (response, expected_valid, description) in enumerate(test_responses, 1):
        print(f"\nTest {i}: {description}")
        print(f'Response: "{response}"')

        result = validator.validate(response, product_name="Fitness Watch")

        print(f"Valid: {result.valid} (Expected: {expected_valid})")
        print(f"Word count: {result.word_count}")
        print(f"Sentence count: {result.sentence_count}")
        print(f"Confidence: {result.confidence_score:.2f}")

        if result.issues:
            print("Issues:")
            for issue in result.issues:
                print(f"  - {issue.value}")

        if result.suggestions:
            print("Suggestions:")
            for suggestion in result.suggestions:
                print(f"  - {suggestion}")

        # Test retry adjustment
        if not result.valid:
            adjustment = validator.get_retry_prompt_adjustment(result)
            if adjustment:
                print(f"Retry adjustment: {adjustment}")

        status = "✓" if result.valid == expected_valid else "✗"
        print(f"Test status: {status}")
        print("-" * 60)

    print("\n" + "=" * 60)
    print("Response Validator testing complete")

"""
ABOUTME: Sentiment Amplification Module for SSR Enhancement
ABOUTME: Detects strong sentiment in responses and adjusts distributions accordingly

This module solves the embedding similarity problem by using semantic cues
to amplify distributions when responses contain strong sentiment keywords.

Problem: Embeddings have 75% similarity between "absolutely not" and "definitely buy"
Solution: Detect strong keywords and shift distributions toward extremes
"""

import numpy as np
from typing import List, Tuple
from dataclasses import dataclass


@dataclass
class SentimentAnalysis:
    """Result of sentiment analysis on response text"""

    text: str
    sentiment_score: float  # -1.0 (very negative) to +1.0 (very positive)
    confidence: float  # 0.0 to 1.0
    strong_positive: bool
    strong_negative: bool
    keywords_found: List[str]


class SentimentAmplifier:
    """
    Analyzes response sentiment and amplifies SSR distributions accordingly

    Uses keyword detection to identify strong purchase intent signals
    that embeddings may not capture effectively due to high similarity.
    """

    # Strong positive purchase intent keywords
    STRONG_POSITIVE_KEYWORDS = {
        'definitely': 0.8,
        'absolutely': 0.9,
        'perfect': 0.7,
        'exactly': 0.7,
        'love': 0.6,
        'excellent': 0.7,
        'immediately': 0.8,
        'for sure': 0.7,
        'without hesitation': 0.9,
        'cant wait': 0.8,
        "can't wait": 0.8,
        'must have': 0.8,
        'need this': 0.7,
    }

    # Strong negative purchase intent keywords
    STRONG_NEGATIVE_KEYWORDS = {
        'never': 0.9,
        'absolutely not': 1.0,
        'no way': 0.9,
        'terrible': 0.8,
        'awful': 0.8,
        'horrible': 0.8,
        'hate': 0.7,
        'disgusting': 0.9,
        'ridiculous': 0.7,
        'waste': 0.7,
        'overpriced': 0.6,
        'too expensive': 0.6,
        'way too': 0.7,
        'not interested': 0.7,
        'doesnt interest': 0.7,
        "doesn't interest": 0.7,
        "don't want": 0.8,
        'dont want': 0.8,
        'wont buy': 0.9,
        "won't buy": 0.9,
        "wouldn't buy": 0.8,
        'wouldnt buy': 0.8,
    }

    # Hedging keywords that reduce confidence
    HEDGING_KEYWORDS = {
        'maybe', 'might', 'perhaps', 'possibly', 'consider',
        'thinking about', 'not sure', 'unsure', 'uncertain'
    }

    def __init__(
        self,
        amplification_strength: float = 0.3,
        min_confidence: float = 0.5
    ):
        """
        Initialize sentiment amplifier

        Args:
            amplification_strength: How much to shift distributions (0.0-1.0)
            min_confidence: Minimum confidence to apply amplification
        """
        self.amplification_strength = amplification_strength
        self.min_confidence = min_confidence

    def analyze_sentiment(self, text: str) -> SentimentAnalysis:
        """
        Analyze sentiment of response text

        Args:
            text: Response text to analyze

        Returns:
            SentimentAnalysis with score and confidence
        """
        text_lower = text.lower()

        # Find positive keywords
        positive_keywords = []
        positive_score = 0.0
        for keyword, weight in self.STRONG_POSITIVE_KEYWORDS.items():
            if keyword in text_lower:
                positive_keywords.append(keyword)
                positive_score += weight

        # Find negative keywords
        negative_keywords = []
        negative_score = 0.0
        for keyword, weight in self.STRONG_NEGATIVE_KEYWORDS.items():
            if keyword in text_lower:
                negative_keywords.append(keyword)
                negative_score += weight

        # Find hedging keywords (reduce confidence)
        hedging_count = sum(1 for keyword in self.HEDGING_KEYWORDS if keyword in text_lower)
        hedge_penalty = min(0.3, hedging_count * 0.1)

        # Calculate net sentiment score
        net_score = positive_score - negative_score

        # Normalize to -1.0 to +1.0
        max_score = 2.0  # Maximum possible positive or negative score
        sentiment_score = np.clip(net_score / max_score, -1.0, 1.0)

        # Calculate confidence (how strong the sentiment is)
        confidence = min(1.0, (positive_score + negative_score) / max_score)
        confidence = max(0.0, confidence - hedge_penalty)

        # Determine if strong sentiment
        strong_positive = positive_score > 0.6 and net_score > 0.5
        strong_negative = negative_score > 0.6 and net_score < -0.5

        return SentimentAnalysis(
            text=text,
            sentiment_score=sentiment_score,
            confidence=confidence,
            strong_positive=strong_positive,
            strong_negative=strong_negative,
            keywords_found=positive_keywords + negative_keywords
        )

    def amplify_distribution(
        self,
        original_distribution: np.ndarray,
        sentiment: SentimentAnalysis
    ) -> Tuple[np.ndarray, bool]:
        """
        Amplify distribution based on sentiment analysis

        Args:
            original_distribution: Original SSR probability distribution (5 elements)
            sentiment: Sentiment analysis result

        Returns:
            Tuple of (amplified_distribution, was_amplified)
        """
        # Check if amplification should be applied
        if sentiment.confidence < self.min_confidence:
            return original_distribution, False

        # Don't amplify if no strong sentiment
        if not sentiment.strong_positive and not sentiment.strong_negative:
            return original_distribution, False

        # Create amplification vector
        # Ratings: [1, 2, 3, 4, 5]
        amplification = np.zeros(5)

        if sentiment.strong_negative:
            # Shift toward rating 1-2
            strength = abs(sentiment.sentiment_score) * self.amplification_strength
            amplification[0] = strength * 0.6  # Rating 1
            amplification[1] = strength * 0.4  # Rating 2
            amplification[2] = -strength * 0.3  # Rating 3
            amplification[3] = -strength * 0.4  # Rating 4
            amplification[4] = -strength * 0.3  # Rating 5

        elif sentiment.strong_positive:
            # Shift toward rating 4-5
            strength = abs(sentiment.sentiment_score) * self.amplification_strength
            amplification[0] = -strength * 0.3  # Rating 1
            amplification[1] = -strength * 0.4  # Rating 2
            amplification[2] = -strength * 0.3  # Rating 3
            amplification[3] = strength * 0.4  # Rating 4
            amplification[4] = strength * 0.6  # Rating 5

        # Apply amplification
        amplified = original_distribution + amplification

        # Ensure non-negative
        amplified = np.maximum(amplified, 0.001)

        # Re-normalize to sum to 1.0
        amplified = amplified / amplified.sum()

        return amplified, True

    def process(
        self,
        text: str,
        original_distribution: np.ndarray
    ) -> Tuple[np.ndarray, SentimentAnalysis, bool]:
        """
        Complete sentiment amplification process

        Args:
            text: Response text
            original_distribution: Original SSR distribution

        Returns:
            Tuple of (final_distribution, sentiment_analysis, was_amplified)
        """
        sentiment = self.analyze_sentiment(text)
        amplified_dist, was_amplified = self.amplify_distribution(
            original_distribution, sentiment
        )

        return amplified_dist, sentiment, was_amplified


# Example usage and testing
if __name__ == "__main__":
    amplifier = SentimentAmplifier(
        amplification_strength=0.3,
        min_confidence=0.5
    )

    # Test cases
    test_cases = [
        "I would definitely buy this immediately! It's perfect for my needs.",
        "I absolutely would not buy this. It's way too expensive and terrible quality.",
        "I might consider it, but I'm not sure if it's right for me.",
        "This doesn't interest me at all. I would never purchase this product.",
        "I would likely purchase this product. It seems like a good value.",
    ]

    print("Sentiment Amplifier Test")
    print("=" * 70)

    for text in test_cases:
        sentiment = amplifier.analyze_sentiment(text)

        print(f"\nText: \"{text[:60]}...\"" if len(text) > 60 else f"\nText: \"{text}\"")
        print(f"Sentiment Score: {sentiment.sentiment_score:+.2f}")
        print(f"Confidence: {sentiment.confidence:.2f}")
        print(f"Strong Positive: {sentiment.strong_positive}")
        print(f"Strong Negative: {sentiment.strong_negative}")
        print(f"Keywords: {sentiment.keywords_found}")

        # Test distribution amplification
        uniform_dist = np.array([0.20, 0.20, 0.20, 0.20, 0.20])
        amplified, was_amplified = amplifier.amplify_distribution(uniform_dist, sentiment)

        if was_amplified:
            print(f"Original:  {[f'{p:.2f}' for p in uniform_dist]}")
            print(f"Amplified: {[f'{p:.2f}' for p in amplified]}")
            mean_original = sum((i+1) * p for i, p in enumerate(uniform_dist))
            mean_amplified = sum((i+1) * p for i, p in enumerate(amplified))
            print(f"Mean rating: {mean_original:.2f} → {mean_amplified:.2f} (Δ {mean_amplified-mean_original:+.2f})")
        else:
            print("Not amplified (confidence too low or no strong sentiment)")

    print("\n" + "=" * 70)

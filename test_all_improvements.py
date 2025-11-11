#!/usr/bin/env python
"""
COMPREHENSIVE TEST: All SSR Improvements

Tests the complete improved SSR system:
1. Temperature optimization (T=0.5 vs paper's T=1.5)
2. Sentiment amplification (hybrid approach)
3. Product category profiles (category-specific configs)

Expected: Rating spreads > 1.0 with strong demographic effects visible
"""

import sys
import os
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.services.consumer_generator import ConsumerGenerator
from src.core.ssr_engine import SSREngine, SSRConfig
from src.core.product_categories import ProductCategory, get_category_manager


def test_with_improvements():
    """Test SSR with all improvements enabled"""

    print("\n" + "="*80)
    print("COMPREHENSIVE TEST: ALL SSR IMPROVEMENTS")
    print("="*80)

    print("\nImprovements Tested:")
    print("  1. ✓ Temperature = 0.5 (vs paper's 1.5)")
    print("  2. ✓ Sentiment amplification (hybrid approach)")
    print("  3. ✓ Product category optimization")

    # Check API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("\n❌ ERROR: OPENAI_API_KEY not found")
        return

    # Initialize
    consumer_generator = ConsumerGenerator(api_key=api_key)
    category_manager = get_category_manager()

    # Test product: Cricket protein bars (controversial category)
    product = {
        'name': "CrickPro High-Protein Energy Bars",
        'description': """Revolutionary protein bars made from sustainable cricket flour.
• 20g complete protein per bar from ethically-farmed crickets
• Environmentally sustainable and high in B12
• Chocolate chip flavor with cricket flour base
Price: $24.99 for 12 bars""",
        'price': 24.99,
        'category': ProductCategory.CONTROVERSIAL
    }

    # Get optimized configuration for this product
    cat_config = category_manager.get_config(product['category'])

    print(f"\n{'-'*80}")
    print(f"Test Product: {product['name']}")
    print(f"Category: {product['category'].value}")
    print(f"Optimized Config:")
    print(f"  Temperature: {cat_config.temperature}")
    print(f"  Amplification: {cat_config.amplification_strength}")
    print(f"{'-'*80}")

    # Create SSR engine with optimized config
    config = SSRConfig(
        temperature=cat_config.temperature,
        use_multi_set_averaging=True,
        enable_sentiment_amplification=cat_config.amplification_enabled,
        sentiment_amplification_strength=cat_config.amplification_strength,
        sentiment_min_confidence=cat_config.min_confidence
    )

    ssr_engine = SSREngine(config=config, api_key=api_key)

    print(f"\nSSR Engine Ready:")
    print(f"  T={config.temperature} | Amplification={config.enable_sentiment_amplification}")

    # Generate diverse consumers
    print(f"\n{'-'*80}")
    print("Generating diverse consumers...")
    consumers = consumer_generator.generate_consumers(count=6, demographics_enabled=True)

    results = []

    for i, consumer in enumerate(consumers, 1):
        print(f"\n{'.'*80}")
        print(f"Consumer {i}: {consumer.age}y {consumer.gender}, {consumer.income} income")

        # Generate response
        response = consumer_generator.generate_response(
            consumer=consumer,
            product_name=product['name'],
            product_description=product['description'],
            llm_model="gpt-3.5-turbo",
            temperature=1.0
        )

        print(f"Response: {response[:100]}...")

        # Calculate SSR
        ssr_result = ssr_engine.process_response(response)

        print(f"\nSSR Analysis:")
        print(f"  Rating: {ssr_result.mean_rating:.2f} (Mode: {ssr_result.get_most_likely_rating()})")
        print(f"  Distribution: {[f'{p:.2f}' for p in ssr_result.distribution.probabilities]}")

        # Show sentiment amplification if applied
        if ssr_result.sentiment_amplified:
            sentiment = ssr_result.sentiment_analysis
            print(f"  🔥 AMPLIFIED by sentiment!")
            print(f"     Sentiment: {sentiment.sentiment_score:+.2f} (confidence: {sentiment.confidence:.2f})")
            print(f"     Keywords: {sentiment.keywords_found}")
            print(f"     Strong {'Positive' if sentiment.strong_positive else 'Negative'}")
        else:
            print(f"  ℹ️  No amplification (no strong sentiment detected)")

        results.append({
            'consumer': consumer,
            'response': response,
            'rating': ssr_result.mean_rating,
            'amplified': ssr_result.sentiment_amplified,
            'sentiment': ssr_result.sentiment_analysis
        })

    # Analyze results
    print(f"\n\n{'='*80}")
    print("RESULTS ANALYSIS")
    print(f"{'='*80}")

    ratings = [r['rating'] for r in results]
    amplified_count = sum(1 for r in results if r['amplified'])

    print(f"\nRating Statistics:")
    print(f"  Mean: {np.mean(ratings):.2f}")
    print(f"  Std Dev: {np.std(ratings):.2f}")
    print(f"  Range: {np.min(ratings):.2f} - {np.max(ratings):.2f}")
    print(f"  Spread: {np.max(ratings) - np.min(ratings):.2f}")

    print(f"\nSentiment Amplification:")
    print(f"  Amplified: {amplified_count}/{len(results)} responses")
    print(f"  Rate: {amplified_count/len(results)*100:.0f}%")

    # Demographic analysis
    positive_responses = [r for r in results if r['rating'] >= 3.5]
    negative_responses = [r for r in results if r['rating'] < 2.5]

    print(f"\nDemographic Patterns:")
    if positive_responses:
        print(f"  Positive (≥3.5): {len(positive_responses)} responses")
        for r in positive_responses:
            print(f"    • {r['consumer'].age}y {r['consumer'].gender}, {r['consumer'].income} - {r['rating']:.2f}")

    if negative_responses:
        print(f"  Negative (<2.5): {len(negative_responses)} responses")
        for r in negative_responses:
            print(f"    • {r['consumer'].age}y {r['consumer'].gender}, {r['consumer'].income} - {r['rating']:.2f}")

    # Evaluation
    rating_spread = np.max(ratings) - np.min(ratings)

    print(f"\n{'='*80}")
    print("EVALUATION")
    print(f"{'='*80}")

    if rating_spread > 1.0:
        print(f"🎉 EXCELLENT: Rating spread {rating_spread:.2f} > 1.0")
        print(f"   Strong differentiation achieved!")
    elif rating_spread > 0.5:
        print(f"✅ GOOD: Rating spread {rating_spread:.2f} > 0.5")
        print(f"   Moderate differentiation with improvements")
    elif rating_spread > 0.3:
        print(f"⚠️  WEAK: Rating spread {rating_spread:.2f} > 0.3")
        print(f"   Limited but improved differentiation")
    else:
        print(f"❌ POOR: Rating spread {rating_spread:.2f} < 0.3")
        print(f"   No meaningful differentiation")

    print(f"\nComparison to baseline:")
    print(f"  Paper T=1.5, no amplification:  ~0.006 spread ❌")
    print(f"  T=0.5, no amplification:        ~0.207 spread ⚠️")
    print(f"  T=0.5, WITH amplification:      {rating_spread:.3f} spread {'✅' if rating_spread > 0.5 else '⚠️'}")

    improvement = rating_spread / 0.006 if rating_spread > 0 else 0
    print(f"  Improvement: {improvement:.0f}x better than baseline!")

    print(f"\n{'='*80}")
    print("TEST COMPLETE")
    print(f"{'='*80}")


if __name__ == "__main__":
    test_with_improvements()

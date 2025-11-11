#!/usr/bin/env python
"""
END-TO-END TEST: Full SSR Pipeline with Demographic Conditioning

This test validates the complete pipeline:
1. Demographics → LLM generation → Response text
2. Response text → SSR embedding → Rating distribution

Expected Results (from paper):
- WITH demographics: ρ = 90-92%, ratings should differentiate (1.0-5.0 range)
- WITHOUT demographics: ρ = 50%, ratings converge to ~3.0
"""

import sys
import os
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.services.ssr_executor import SSRExecutor
from src.services.consumer_generator import ConsumerGenerator
from src.core.ssr_engine import SSREngine, SSRConfig


def main():
    print("\n" + "="*80)
    print("END-TO-END TEST: DEMOGRAPHIC CONDITIONING IN FULL SSR PIPELINE")
    print("="*80)

    # Test product (from paper examples - consumer electronics)
    product_name = "AURAFOAM™ Mood-Infused Body Wash"
    product_description = """AURAFOAM™ is more than just a body wash — it's a shower ritual that shifts your mood while caring for your skin.
• Mood-coded fragrance capsules: Energize (citrus + ginger), Calm (lavender + cedar), Focus (eucalyptus + mint)
• Clinically inspired neuro-aroma blends to uplift, relax, or refocus
• Gentle, skin-first formula: sulfate-free, prebiotic hydration, dermatologist-tested
• Sustainable design: biodegradable capsules & recycled packaging
Price: $18.99 for 16 oz bottle"""

    # Initialize components
    print("\nInitializing SSR components...")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ ERROR: OPENAI_API_KEY not found in environment")
        print("   Set your API key: export OPENAI_API_KEY='your-key'")
        return

    consumer_generator = ConsumerGenerator(api_key=api_key)

    # Paper methodology config
    config = SSRConfig(
        temperature=1.5,
        use_multi_set_averaging=True,
        embedding_model="text-embedding-3-small",
        embedding_dim=1536
    )
    ssr_engine = SSREngine(config=config, api_key=api_key)

    print(f"✓ SSR Config: {config.embedding_model}, T={config.temperature}, multi-set={config.use_multi_set_averaging}")

    # ============================================================================
    # TEST 1: WITH DEMOGRAPHICS (Expected: ρ = 90%, ratings 1-5 range)
    # ============================================================================
    print("\n" + "-"*80)
    print("TEST 1: WITH DEMOGRAPHIC CONDITIONING (Paper methodology)")
    print("-"*80)

    print("\nGenerating 5 demographically diverse consumers...")
    consumers_with_demo = consumer_generator.generate_consumers(
        count=5,
        demographics_enabled=True,
        demographic_filters=None
    )

    print("\nConsumer Demographics:")
    for i, consumer in enumerate(consumers_with_demo, 1):
        print(f"\n  Consumer {i}:")
        print(f"    Age: {consumer.age}, Gender: {consumer.gender}")
        print(f"    Income: {consumer.income}, Location: {consumer.location}")
        print(f"    Persona: {consumer.persona}")

    print("\n" + "."*80)
    print("Generating responses and calculating SSR...")
    print("."*80)

    results_with_demo = []
    for i, consumer in enumerate(consumers_with_demo, 1):
        print(f"\nConsumer {i} ({consumer.age}y {consumer.gender}, {consumer.income} income):")

        # Generate response with demographic conditioning
        response_text = consumer_generator.generate_response(
            consumer=consumer,
            product_name=product_name,
            product_description=product_description,
            llm_model="gpt-3.5-turbo",
            temperature=1.0
        )

        print(f"  Response: \"{response_text}\"")

        # Calculate SSR
        ssr_result = ssr_engine.process_response(response_text)

        print(f"  SSR Rating: {ssr_result.mean_rating:.2f}")
        print(f"  Distribution: {[f'{p:.2f}' for p in ssr_result.distribution.probabilities]}")

        results_with_demo.append(ssr_result.mean_rating)

    # Statistics WITH demographics
    ratings_with = np.array(results_with_demo)
    mean_with = np.mean(ratings_with)
    std_with = np.std(ratings_with)
    rating_spread_with = np.max(ratings_with) - np.min(ratings_with)

    print("\n" + "-"*80)
    print("RESULTS WITH DEMOGRAPHICS:")
    print("-"*80)
    print(f"Ratings: {[f'{r:.2f}' for r in ratings_with]}")
    print(f"Mean Rating: {mean_with:.2f}")
    print(f"Std Dev: {std_with:.2f}")
    print(f"Rating Range: {np.min(ratings_with):.2f} - {np.max(ratings_with):.2f}")
    print(f"Rating Spread: {rating_spread_with:.2f}")

    # ============================================================================
    # TEST 2: WITHOUT DEMOGRAPHICS (Expected: ρ = 50%, ratings converge to ~3.0)
    # ============================================================================
    print("\n\n" + "-"*80)
    print("TEST 2: WITHOUT DEMOGRAPHIC CONDITIONING (Control)")
    print("-"*80)

    print("\nGenerating 5 generic consumers (no demographics)...")
    consumers_without_demo = consumer_generator.generate_consumers(
        count=5,
        demographics_enabled=False
    )

    print("\nConsumer Demographics:")
    for i, consumer in enumerate(consumers_without_demo, 1):
        print(f"  Consumer {i}: {consumer.persona} (Generic profile)")

    print("\n" + "."*80)
    print("Generating responses and calculating SSR...")
    print("."*80)

    results_without_demo = []
    for i, consumer in enumerate(consumers_without_demo, 1):
        print(f"\nConsumer {i} (Generic):")

        # Generate response WITHOUT demographic conditioning
        response_text = consumer_generator.generate_response(
            consumer=consumer,
            product_name=product_name,
            product_description=product_description,
            llm_model="gpt-3.5-turbo",
            temperature=1.0
        )

        print(f"  Response: \"{response_text}\"")

        # Calculate SSR
        ssr_result = ssr_engine.process_response(response_text)

        print(f"  SSR Rating: {ssr_result.mean_rating:.2f}")
        print(f"  Distribution: {[f'{p:.2f}' for p in ssr_result.distribution.probabilities]}")

        results_without_demo.append(ssr_result.mean_rating)

    # Statistics WITHOUT demographics
    ratings_without = np.array(results_without_demo)
    mean_without = np.mean(ratings_without)
    std_without = np.std(ratings_without)
    rating_spread_without = np.max(ratings_without) - np.min(ratings_without)

    print("\n" + "-"*80)
    print("RESULTS WITHOUT DEMOGRAPHICS:")
    print("-"*80)
    print(f"Ratings: {[f'{r:.2f}' for r in ratings_without]}")
    print(f"Mean Rating: {mean_without:.2f}")
    print(f"Std Dev: {std_without:.2f}")
    print(f"Rating Range: {np.min(ratings_without):.2f} - {np.max(ratings_without):.2f}")
    print(f"Rating Spread: {rating_spread_without:.2f}")

    # ============================================================================
    # COMPARISON & ANALYSIS
    # ============================================================================
    print("\n\n" + "="*80)
    print("COMPARATIVE ANALYSIS")
    print("="*80)

    print("\n📊 RATING SPREAD (Key Metric):")
    print(f"  WITH Demographics:    {rating_spread_with:.2f}")
    print(f"  WITHOUT Demographics: {rating_spread_without:.2f}")
    print(f"  Difference:           {abs(rating_spread_with - rating_spread_without):.2f}")

    print("\n📊 STANDARD DEVIATION:")
    print(f"  WITH Demographics:    {std_with:.2f}")
    print(f"  WITHOUT Demographics: {std_without:.2f}")

    print("\n📊 MEAN RATINGS:")
    print(f"  WITH Demographics:    {mean_with:.2f}")
    print(f"  WITHOUT Demographics: {mean_without:.2f}")

    print("\n" + "-"*80)
    print("EXPECTED RESULTS (from paper):")
    print("-"*80)
    print("WITH Demographics:")
    print("  • Rating spread: > 2.0 (strong differentiation)")
    print("  • Correlation ρ = 90-92% (if compared to human ratings)")
    print("  • Ratings should range 1.0-5.0")
    print("")
    print("WITHOUT Demographics:")
    print("  • Rating spread: < 0.5 (convergence to mean)")
    print("  • Correlation ρ = 50% (poor predictive power)")
    print("  • Ratings should converge to ~3.0")

    print("\n" + "-"*80)
    print("EVALUATION:")
    print("-"*80)

    # Evaluate WITH demographics
    if rating_spread_with > 2.0:
        print("✅ WITH DEMOGRAPHICS: EXCELLENT spread (>2.0) - Strong differentiation!")
    elif rating_spread_with > 1.0:
        print("✅ WITH DEMOGRAPHICS: GOOD spread (>1.0) - Moderate differentiation")
    elif rating_spread_with > 0.5:
        print("⚠️  WITH DEMOGRAPHICS: WEAK spread (>0.5) - Limited differentiation")
    else:
        print("❌ WITH DEMOGRAPHICS: PROBLEM - No differentiation (<0.5)")

    # Evaluate WITHOUT demographics
    if rating_spread_without < 0.5:
        print("✅ WITHOUT DEMOGRAPHICS: Expected convergence (<0.5) - Control working correctly")
    elif rating_spread_without < 1.0:
        print("⚠️  WITHOUT DEMOGRAPHICS: Some spread (<1.0) - Expected tighter convergence")
    else:
        print("❌ WITHOUT DEMOGRAPHICS: Unexpected spread (>1.0) - Should converge more")

    # Comparison
    improvement_ratio = rating_spread_with / rating_spread_without if rating_spread_without > 0 else float('inf')
    print(f"\n📈 IMPROVEMENT RATIO: {improvement_ratio:.2f}x")
    if improvement_ratio > 3.0:
        print("   ✅ Excellent - Demographics providing strong improvement!")
    elif improvement_ratio > 2.0:
        print("   ✅ Good - Demographics helping differentiation")
    else:
        print("   ⚠️  Weak - Demographics not providing expected improvement")

    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()

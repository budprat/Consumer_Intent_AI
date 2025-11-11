#!/usr/bin/env python
"""
EXTREME PRODUCTS TEST SUITE

Tests SSR with products designed to show strong demographic differentiation:
1. Luxury items ($1000+) - Strong income effects
2. Controversial products - Strong opinion variance
3. Demographic-specific products - Age/gender target appeal

With T=0.5, we expect rating spreads > 0.5 (vs 0.006 at T=1.5)
"""

import sys
import os
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple

sys.path.insert(0, str(Path(__file__).parent))

from src.services.consumer_generator import ConsumerGenerator
from src.core.ssr_engine import SSREngine, SSRConfig


# ============================================================================
# EXTREME PRODUCT DEFINITIONS
# ============================================================================

EXTREME_PRODUCTS = {
    "luxury_watch": {
        "name": "Patek Philippe Grand Complications Watch",
        "description": """Ultra-luxury mechanical watch with perpetual calendar and moon phase.
• Handcrafted 18k white gold case with sapphire exhibition back
• Swiss-made automatic movement with 48-hour power reserve
• Limited edition (only 500 pieces worldwide)
• Includes certificate of authenticity and luxury presentation box
• Complimentary servicing for first 5 years
Price: $45,000 USD""",
        "category": "luxury",
        "expected_pattern": "Strong income effect: Low income rejects, High income considers"
    },

    "budget_smartphone": {
        "name": "EcoPhone Basic",
        "description": """Ultra-affordable smartphone with essential features only.
• 5-inch LCD display (480x800 resolution)
• 2GB RAM, 16GB storage
• 8MP rear camera, 2MP front camera
• 2-day battery life with 3000mAh battery
• Runs lightweight Android Go edition
• 1-year warranty
Price: $79 USD""",
        "category": "budget",
        "expected_pattern": "Inverted income effect: Low income appreciates value, High income wants better"
    },

    "cricket_protein_bars": {
        "name": "CrickPro High-Protein Energy Bars",
        "description": """Revolutionary protein bars made from sustainable cricket flour.
• 20g complete protein per bar (all 9 essential amino acids)
• Made from ethically-farmed crickets (environmentally sustainable)
• Chocolate chip flavor with cricket flour base
• Gluten-free, dairy-free, high in B12 and iron
• Certified organic and non-GMO
• Box of 12 bars
Price: $24.99 USD""",
        "category": "controversial",
        "expected_pattern": "Strong opinion variance: Eco-conscious open, traditional reject"
    },

    "vegan_leather_jacket": {
        "name": "VeganLux Premium Faux Leather Jacket",
        "description": """High-end vegan leather jacket made from innovative plant-based materials.
• Premium apple leather (made from apple waste)
• Italian design with classic motorcycle jacket style
• Fully lined with organic cotton
• Certified cruelty-free and vegan
• Water-resistant and durable
• Available in black and brown
Price: $299 USD""",
        "category": "ethical",
        "expected_pattern": "Age/values effect: Young progressive vs traditional preferences"
    },

    "gaming_chair": {
        "name": "ProGamer Elite Racing Chair",
        "description": """Professional esports gaming chair with advanced ergonomic design.
• RGB LED lighting with 16 million color options
• Premium PU leather with memory foam cushioning
• 4D adjustable armrests and lumbar support
• Reclines 90-180 degrees with locking mechanism
• Heavy-duty metal frame (330 lb capacity)
• 5-year manufacturer warranty
Price: $449 USD""",
        "category": "age_specific",
        "expected_pattern": "Strong age effect: Young gamers interested, seniors not interested"
    },

    "anti_aging_serum": {
        "name": "Youth Restore Advanced Anti-Aging Serum",
        "description": """Clinically-tested anti-aging facial serum with proven results.
• Contains retinol, hyaluronic acid, and peptides
• Reduces fine lines and wrinkles by up to 40% in 8 weeks
• Dermatologist-tested and recommended
• Suitable for all skin types
• Fragrance-free and non-comedogenic
• 1 oz bottle (2-month supply)
Price: $89 USD""",
        "category": "age_specific",
        "expected_pattern": "Age effect: Middle-aged+ interested, young adults not interested"
    },

    "cryptocurrency_course": {
        "name": "Crypto Mastery Complete Trading Course",
        "description": """Comprehensive online course for cryptocurrency trading and investing.
• 40 hours of video content from professional traders
• Live trading sessions and portfolio analysis
• Access to private Discord community
• Weekly market analysis and trade alerts
• Covers Bitcoin, Ethereum, DeFi, and NFTs
• Lifetime access with free updates
Price: $997 USD""",
        "category": "controversial",
        "expected_pattern": "Age/tech-savvy effect: Young tech-forward vs skeptical older"
    },
}


def test_product(
    product_id: str,
    product_info: Dict,
    consumer_generator: ConsumerGenerator,
    ssr_engine: SSREngine,
    cohort_size: int = 8
) -> Dict:
    """Test a single product with diverse demographic cohort"""

    print(f"\n{'='*80}")
    print(f"TESTING: {product_info['name']}")
    print(f"{'='*80}")
    print(f"Category: {product_info['category']}")
    print(f"Expected Pattern: {product_info['expected_pattern']}")
    print(f"\nProduct Description:")
    print(product_info['description'][:200] + "...")

    # Generate diverse consumers
    consumers = consumer_generator.generate_consumers(
        count=cohort_size,
        demographics_enabled=True
    )

    print(f"\n{'-'*80}")
    print(f"Testing with {cohort_size} diverse consumers...")
    print(f"{'-'*80}")

    results = []

    for i, consumer in enumerate(consumers, 1):
        # Generate response
        response = consumer_generator.generate_response(
            consumer=consumer,
            product_name=product_info['name'],
            product_description=product_info['description'],
            llm_model="gpt-3.5-turbo",
            temperature=1.0
        )

        # Calculate SSR
        ssr_result = ssr_engine.process_response(response)

        # Store result
        result = {
            'consumer_id': i,
            'age': consumer.age,
            'gender': consumer.gender,
            'income': consumer.income,
            'location': consumer.location,
            'response': response,
            'rating': ssr_result.mean_rating,
            'distribution': ssr_result.distribution.probabilities,
            'most_likely': ssr_result.get_most_likely_rating()
        }
        results.append(result)

        # Print result
        print(f"\nConsumer {i} ({consumer.age}y {consumer.gender}, {consumer.income} income):")
        print(f"  Response: {response[:100]}...")
        print(f"  Rating: {result['rating']:.2f} (Mode: {result['most_likely']})")
        print(f"  Distribution: {[f'{p:.2f}' for p in result['distribution']]}")

    # Calculate statistics
    ratings = [r['rating'] for r in results]
    rating_spread = np.max(ratings) - np.min(ratings)
    rating_std = np.std(ratings)
    rating_mean = np.mean(ratings)

    # Analyze by demographics
    low_income_ratings = [r['rating'] for r in results if r['income'] == 'Low']
    high_income_ratings = [r['rating'] for r in results if r['income'] == 'High']

    young_ratings = [r['rating'] for r in results if r['age'] < 35]
    old_ratings = [r['rating'] for r in results if r['age'] >= 55]

    print(f"\n{'-'*80}")
    print(f"RESULTS SUMMARY")
    print(f"{'-'*80}")
    print(f"Mean Rating: {rating_mean:.2f}")
    print(f"Std Dev: {rating_std:.2f}")
    print(f"Rating Range: {np.min(ratings):.2f} - {np.max(ratings):.2f}")
    print(f"Rating Spread: {rating_spread:.2f}")

    if low_income_ratings and high_income_ratings:
        income_diff = np.mean(high_income_ratings) - np.mean(low_income_ratings)
        print(f"\nIncome Effect:")
        print(f"  Low Income avg: {np.mean(low_income_ratings):.2f}")
        print(f"  High Income avg: {np.mean(high_income_ratings):.2f}")
        print(f"  Difference: {income_diff:+.2f}")

    if young_ratings and old_ratings:
        age_diff = np.mean(young_ratings) - np.mean(old_ratings)
        print(f"\nAge Effect:")
        print(f"  Young (<35) avg: {np.mean(young_ratings):.2f}")
        print(f"  Senior (55+) avg: {np.mean(old_ratings):.2f}")
        print(f"  Difference: {age_diff:+.2f}")

    # Evaluation
    print(f"\n{'-'*80}")
    print(f"EVALUATION:")
    print(f"{'-'*80}")
    if rating_spread > 1.0:
        print(f"✅ EXCELLENT: Rating spread > 1.0 - Strong differentiation!")
    elif rating_spread > 0.5:
        print(f"✅ GOOD: Rating spread > 0.5 - Moderate differentiation")
    elif rating_spread > 0.2:
        print(f"⚠️  WEAK: Rating spread > 0.2 - Limited differentiation")
    else:
        print(f"❌ POOR: Rating spread < 0.2 - No meaningful differentiation")

    return {
        'product_id': product_id,
        'product_name': product_info['name'],
        'category': product_info['category'],
        'rating_spread': rating_spread,
        'rating_std': rating_std,
        'rating_mean': rating_mean,
        'results': results
    }


def main():
    print("\n" + "="*80)
    print("EXTREME PRODUCTS TEST SUITE")
    print("="*80)
    print("\nTesting SSR with products designed for strong demographic effects")
    print(f"Temperature: 0.5 (optimized, paper used 1.5)")
    print(f"Expected: Rating spreads > 0.5 (vs ~0.006 with moderate products)")

    # Check API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("\n❌ ERROR: OPENAI_API_KEY not found in environment")
        return

    # Initialize components
    print("\nInitializing SSR components...")
    consumer_generator = ConsumerGenerator(api_key=api_key)
    config = SSRConfig(
        temperature=0.5,  # Optimized temperature
        use_multi_set_averaging=True
    )
    ssr_engine = SSREngine(config=config, api_key=api_key)

    print(f"✓ SSR Engine ready (T={config.temperature})")

    # Test products
    test_results = []

    # Test selection (use subset for speed)
    products_to_test = [
        'luxury_watch',
        'budget_smartphone',
        'cricket_protein_bars',
        'gaming_chair'
    ]

    for product_id in products_to_test:
        try:
            result = test_product(
                product_id=product_id,
                product_info=EXTREME_PRODUCTS[product_id],
                consumer_generator=consumer_generator,
                ssr_engine=ssr_engine,
                cohort_size=150  # Paper uses 150-400 for statistical power
            )
            test_results.append(result)
        except Exception as e:
            print(f"\n❌ ERROR testing {product_id}: {e}")
            import traceback
            traceback.print_exc()

    # Final comparison
    print("\n\n" + "="*80)
    print("FINAL COMPARISON ACROSS ALL PRODUCTS")
    print("="*80)

    for result in test_results:
        status = "✅ GOOD" if result['rating_spread'] > 0.5 else "⚠️ WEAK" if result['rating_spread'] > 0.2 else "❌ POOR"
        print(f"\n{result['product_name'][:40]:40} | Spread: {result['rating_spread']:.3f} {status}")
        print(f"  Category: {result['category']:15} | Mean: {result['rating_mean']:.2f} | Std: {result['rating_std']:.2f}")

    # Best performing product
    if test_results:
        best = max(test_results, key=lambda x: x['rating_spread'])
        print(f"\n🏆 Best Differentiation: {best['product_name']}")
        print(f"   Spread: {best['rating_spread']:.3f} | Category: {best['category']}")

    print("\n" + "="*80)
    print("TEST SUITE COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()

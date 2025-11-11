#!/usr/bin/env python
"""
Personal Care Products Test Suite

Tests SSR system with personal care products matching the paper's domain.
Paper tested on 57 personal care product surveys (body wash, deodorant, toothpaste, shampoo, etc.)

This test validates our implementation against the correct product domain.
"""

import sys
import os
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.services.consumer_generator import ConsumerGenerator
from src.core.ssr_engine import SSREngine, SSRConfig
from src.core.product_categories import ProductCategory


# Personal Care Products (matching paper's domain)
PERSONAL_CARE_PRODUCTS = {
    "body_wash_premium": {
        "name": "LuxeClean Moisturizing Body Wash",
        "description": """Premium body wash with natural botanical extracts and vitamins.
• pH-balanced formula with shea butter and vitamin E
• Gentle cleansing without stripping natural oils
• Fresh citrus blossom scent
• Dermatologist tested, paraben-free
Price: $12.99 for 18 oz""",
        "price": 12.99,
        "category": ProductCategory.HEALTH_WELLNESS,
        "expected_pattern": "Moderate price, appeals to health-conscious consumers"
    },

    "body_wash_budget": {
        "name": "FreshStart Daily Body Wash",
        "description": """Affordable everyday body wash for the whole family.
• Classic clean scent
• Gentle on sensitive skin
• Value size for daily use
• Trusted brand, dermatologist recommended
Price: $5.49 for 24 oz""",
        "price": 5.49,
        "category": ProductCategory.CONSUMER_GOODS,
        "expected_pattern": "Low price, broad appeal across income levels"
    },

    "deodorant_natural": {
        "name": "PureGuard Natural Deodorant",
        "description": """Aluminum-free natural deodorant with essential oils.
• Baking soda free, gentle formula
• Made with coconut oil and arrowroot powder
• Lavender & mint scent
• Vegan, cruelty-free, eco-friendly packaging
Price: $8.99""",
        "price": 8.99,
        "category": ProductCategory.ETHICAL,
        "expected_pattern": "Appeals to eco-conscious, younger demographics"
    },

    "deodorant_clinical": {
        "name": "MaxProtect Clinical Strength Deodorant",
        "description": """Maximum strength antiperspirant for all-day protection.
• Clinical strength formula, prescription-grade
• 48-hour protection guaranteed
• Unscented, hypoallergenic
• Recommended by dermatologists
Price: $9.99""",
        "price": 9.99,
        "category": ProductCategory.HEALTH_WELLNESS,
        "expected_pattern": "Practical appeal, broad demographic"
    },

    "toothpaste_whitening": {
        "name": "BrightSmile Professional Whitening Toothpaste",
        "description": """Advanced whitening toothpaste with professional results.
• Removes up to 95% of surface stains in 2 weeks
• Strengthens enamel with fluoride
• Fresh mint flavor
• ADA accepted, safe for daily use
Price: $6.99 for 4.2 oz""",
        "price": 6.99,
        "category": ProductCategory.HEALTH_WELLNESS,
        "expected_pattern": "Cosmetic benefit, appeals to image-conscious consumers"
    },

    "toothpaste_sensitive": {
        "name": "SensitiveShield Gentle Care Toothpaste",
        "description": """Specially formulated for sensitive teeth and gums.
• Clinically proven sensitivity relief
• Gentle whitening, low abrasion formula
• Soothing mint flavor
• Dentist recommended for sensitive teeth
Price: $7.49 for 4 oz""",
        "price": 7.49,
        "category": ProductCategory.HEALTH_WELLNESS,
        "expected_pattern": "Targeted need, appeals to older demographics"
    },

    "shampoo_premium": {
        "name": "SalonLux Keratin Repair Shampoo",
        "description": """Professional salon-quality shampoo for damaged hair.
• Infused with keratin proteins and argan oil
• Repairs and strengthens hair from within
• Sulfate-free, color-safe formula
• Luxurious fragrance, salon results at home
Price: $14.99 for 12 oz""",
        "price": 14.99,
        "category": ProductCategory.CONSUMER_GOODS,
        "expected_pattern": "Premium positioning, appeals to higher income"
    },

    "shampoo_family": {
        "name": "CleanEssentials Family Shampoo",
        "description": """Everyday shampoo for the whole family.
• Gentle formula safe for all hair types
• Classic clean scent
• Value size for daily use
• Trusted family brand
Price: $4.99 for 32 oz""",
        "price": 4.99,
        "category": ProductCategory.CONSUMER_GOODS,
        "expected_pattern": "Budget-friendly, broad family appeal"
    },

    "face_cream_antiaging": {
        "name": "YouthRevive Anti-Aging Face Cream",
        "description": """Advanced anti-aging cream with retinol and peptides.
• Reduces fine lines and wrinkles
• Firms and lifts sagging skin
• Hyaluronic acid for deep hydration
• Dermatologist developed, clinically tested
Price: $29.99 for 1.7 oz""",
        "price": 29.99,
        "category": ProductCategory.HEALTH_WELLNESS,
        "expected_pattern": "Age-specific (45+), strong income effect"
    },

    "face_cream_daily": {
        "name": "DailyGlow Moisturizing Face Cream",
        "description": """Lightweight daily moisturizer for all skin types.
• Non-greasy, fast-absorbing formula
• SPF 15 sun protection
• Vitamin C and E for healthy glow
• Fragrance-free, suitable for sensitive skin
Price: $12.99 for 2 oz""",
        "price": 12.99,
        "category": ProductCategory.HEALTH_WELLNESS,
        "expected_pattern": "Everyday product, broad appeal"
    },

    "hand_soap_luxury": {
        "name": "Aromatherapy Botanical Hand Soap",
        "description": """Luxurious hand soap with essential oils.
• French lavender and chamomile blend
• Moisturizing formula with aloe vera
• Elegant glass pump bottle
• Made with natural plant-based ingredients
Price: $8.99 for 10 oz""",
        "price": 8.99,
        "category": ProductCategory.CONSUMER_GOODS,
        "expected_pattern": "Small luxury, appeals to quality-conscious consumers"
    },

    "hand_soap_antibacterial": {
        "name": "GermGuard Antibacterial Hand Soap",
        "description": """Effective antibacterial hand soap for germ protection.
• Kills 99.9% of germs and bacteria
• Gentle on hands, moisturizing formula
• Fresh clean scent
• Trusted brand, pediatrician recommended
Price: $4.99 for 11.25 oz""",
        "price": 4.99,
        "category": ProductCategory.HEALTH_WELLNESS,
        "expected_pattern": "Practical need, family-focused"
    },
}


def test_product(
    product_id: str,
    product_info: dict,
    consumer_generator: ConsumerGenerator,
    ssr_engine: SSREngine,
    cohort_size: int = 150
) -> dict:
    """Test a single personal care product"""

    print(f"\n{'='*80}")
    print(f"Testing: {product_info['name']}")
    print(f"{'='*80}")
    print(f"Price: ${product_info['price']:.2f}")
    print(f"Category: {product_info['category'].value}")
    print(f"Expected Pattern: {product_info['expected_pattern']}")

    # Generate diverse consumer cohort
    print(f"\nGenerating {cohort_size} consumers...")
    consumers = consumer_generator.generate_consumers(
        count=cohort_size,
        demographics_enabled=True
    )

    # Collect results
    results = []
    ratings = []

    print(f"\nGenerating responses and calculating SSR ratings...")
    for i, consumer in enumerate(consumers, 1):
        if i % 25 == 0:
            print(f"  Progress: {i}/{cohort_size} consumers...")

        try:
            # Generate response
            response = consumer_generator.generate_response(
                consumer=consumer,
                product_name=product_info['name'],
                product_description=product_info['description'],
                llm_model="gpt-3.5-turbo"
                # temperature=0.5 is now default (paper-compliant)
            )

            # Calculate SSR
            ssr_result = ssr_engine.process_response(response)

            results.append({
                'consumer': consumer,
                'response': response,
                'rating': ssr_result.mean_rating,
                'distribution': ssr_result.distribution.probabilities
            })
            ratings.append(ssr_result.mean_rating)

        except Exception as e:
            print(f"\n⚠️  Warning: Error processing consumer {i}: {e}")
            continue

    # Analyze results
    ratings_array = np.array(ratings)

    # Calculate statistics
    mean_rating = np.mean(ratings_array)
    std_rating = np.std(ratings_array)
    min_rating = np.min(ratings_array)
    max_rating = np.max(ratings_array)
    spread = max_rating - min_rating

    # Demographic analysis
    age_groups = {
        'young (18-35)': [r['rating'] for r in results if r['consumer'].age <= 35],
        'middle (36-50)': [r['rating'] for r in results if 36 <= r['consumer'].age <= 50],
        'senior (51+)': [r['rating'] for r in results if r['consumer'].age > 50]
    }

    income_groups = {
        'low': [r['rating'] for r in results if r['consumer'].income == 'Low'],
        'middle': [r['rating'] for r in results if r['consumer'].income == 'Middle'],
        'high': [r['rating'] for r in results if r['consumer'].income == 'High']
    }

    # Print results
    print(f"\n{'-'*80}")
    print("RESULTS SUMMARY")
    print(f"{'-'*80}")
    print(f"Mean Rating: {mean_rating:.3f} (StdDev: {std_rating:.3f})")
    print(f"Range: {min_rating:.3f} - {max_rating:.3f}")
    print(f"Spread: {spread:.3f}")

    print(f"\nAge Group Effects:")
    for group, group_ratings in age_groups.items():
        if group_ratings:
            print(f"  {group:20s}: {np.mean(group_ratings):.3f} (n={len(group_ratings)})")

    print(f"\nIncome Effects:")
    for group, group_ratings in income_groups.items():
        if group_ratings:
            print(f"  {group:20s}: {np.mean(group_ratings):.3f} (n={len(group_ratings)})")

    # Age effect magnitude
    if age_groups['young (18-35)'] and age_groups['senior (51+)']:
        age_effect = abs(np.mean(age_groups['young (18-35)']) - np.mean(age_groups['senior (51+)']))
        print(f"\nAge Effect Magnitude: {age_effect:.3f}")

    # Income effect magnitude
    if income_groups['low'] and income_groups['high']:
        income_effect = abs(np.mean(income_groups['low']) - np.mean(income_groups['high']))
        print(f"Income Effect Magnitude: {income_effect:.3f}")

    return {
        'product_id': product_id,
        'product_name': product_info['name'],
        'price': product_info['price'],
        'mean_rating': mean_rating,
        'std_rating': std_rating,
        'spread': spread,
        'age_groups': {k: np.mean(v) if v else None for k, v in age_groups.items()},
        'income_groups': {k: np.mean(v) if v else None for k, v in income_groups.items()},
        'cohort_size': len(ratings)
    }


def main():
    """Run personal care products test suite"""

    print("\n" + "="*80)
    print("PERSONAL CARE PRODUCTS TEST SUITE")
    print("Testing SSR with Paper's Product Domain")
    print("="*80)

    print("\nConfiguration:")
    print("  Product Domain: Personal Care (matching paper)")
    print("  Cohort Size: 150 per product (paper minimum)")
    print("  LLM Temperature: 0.5 (paper-compliant)")
    print("  SSR Temperature: 1.0 (paper-compliant)")
    print("  Products: 12 personal care items")

    # Check API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("\n❌ ERROR: OPENAI_API_KEY not found in environment")
        print("Please set your OpenAI API key:")
        print("  export OPENAI_API_KEY='your-key-here'")
        return

    # Initialize
    print("\nInitializing SSR system...")
    consumer_generator = ConsumerGenerator(api_key=api_key)

    # Paper-compliant configuration
    config = SSRConfig(
        temperature=1.0,  # T_SSR (paper Equation 9)
        use_multi_set_averaging=True,
        enable_sentiment_amplification=False  # Paper doesn't use this
    )
    ssr_engine = SSREngine(config=config, api_key=api_key)

    print("✓ SSR Engine initialized with paper-compliant settings")

    # Test products (start with a subset for speed)
    products_to_test = [
        "body_wash_budget",
        "deodorant_natural",
        "toothpaste_whitening",
        "shampoo_family",
        "face_cream_daily"
    ]

    print(f"\nTesting {len(products_to_test)} products...")
    print("(Run with all 12 products for full validation)")

    test_results = []

    for product_id in products_to_test:
        try:
            result = test_product(
                product_id=product_id,
                product_info=PERSONAL_CARE_PRODUCTS[product_id],
                consumer_generator=consumer_generator,
                ssr_engine=ssr_engine,
                cohort_size=150  # Paper minimum
            )
            test_results.append(result)
        except Exception as e:
            print(f"\n❌ ERROR testing {product_id}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Cross-product analysis
    print(f"\n\n{'='*80}")
    print("CROSS-PRODUCT ANALYSIS")
    print(f"{'='*80}")

    if test_results:
        # Calculate correlation metrics
        mean_ratings = [r['mean_rating'] for r in test_results]
        spreads = [r['spread'] for r in test_results]

        print(f"\nProducts Tested: {len(test_results)}")
        print(f"Mean Rating Range: {min(mean_ratings):.3f} - {max(mean_ratings):.3f}")
        print(f"Cross-Product Spread: {max(mean_ratings) - min(mean_ratings):.3f}")
        print(f"Average Individual Spread: {np.mean(spreads):.3f} ± {np.std(spreads):.3f}")

        print(f"\nProduct Rankings (by mean rating):")
        sorted_results = sorted(test_results, key=lambda x: x['mean_rating'], reverse=True)
        for i, result in enumerate(sorted_results, 1):
            print(f"  {i}. {result['product_name']:45s} ${result['price']:6.2f}  →  {result['mean_rating']:.3f}")

        # Price correlation
        prices = [r['price'] for r in test_results]
        correlation = np.corrcoef(prices, mean_ratings)[0, 1]
        print(f"\nPrice-Rating Correlation: {correlation:.3f}")

        # Expected: Higher price → higher rating (quality perception)
        if correlation > 0.3:
            print("✓ Positive price-quality perception detected")
        elif correlation < -0.3:
            print("✓ Value-seeking behavior detected (prefer lower prices)")
        else:
            print("  Weak price correlation (mixed preferences)")

    print(f"\n{'='*80}")
    print("TEST SUITE COMPLETE")
    print(f"{'='*80}")

    if test_results:
        avg_spread = np.mean([r['spread'] for r in test_results])
        print(f"\n✓ Tested {len(test_results)} personal care products")
        print(f"✓ Average rating spread: {avg_spread:.3f}")
        print(f"✓ Configuration: Paper-compliant (T_LLM=0.5, T_SSR=1.0)")
        print(f"✓ Domain: Personal care (matching paper)")

        if avg_spread > 0.15:
            print(f"\n✅ GOOD: Rating differentiation achieved")
        else:
            print(f"\n⚠️  LIMITED: Rating spread still narrow (may be normal for personal care)")


if __name__ == "__main__":
    main()

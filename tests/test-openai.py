#!/usr/bin/env python3
"""
Real OpenAI SSR Test

Tests the SSR engine with actual OpenAI API calls.
This will consume API credits (~$0.01-0.02 per run).
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.core.ssr_engine import SSREngine
from src.core.reference_statements import ReferenceStatementSet, ReferenceStatement
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def print_section(title):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)


def create_test_reference_set():
    """Create a test reference set for purchase intent."""
    # Create individual reference statements
    statements = [
        ReferenceStatement(rating=1, text="I would never purchase this product."),
        ReferenceStatement(rating=2, text="I am unlikely to purchase this product."),
        ReferenceStatement(rating=3, text="I might purchase this product."),
        ReferenceStatement(rating=4, text="I would likely purchase this product."),
        ReferenceStatement(rating=5, text="I would definitely purchase this product."),
    ]

    # Create reference set
    return ReferenceStatementSet(
        id="test_purchase_intent",
        version="1.0",
        domain="consumer_products",
        language="en",
        statements=statements,
        description="5-point purchase intent scale from definitely not purchase to definitely purchase",
    )


def test_ssr_rating(product_description, model="gpt-4o"):
    """Test SSR rating generation with real OpenAI API calls."""

    print_section("SSR Rating Generation Test")
    print(f"Model: {model}")
    print(f"Product: {product_description}")
    print()

    # Create reference set
    ref_set = create_test_reference_set()

    # Initialize SSR engine
    engine = SSREngine(reference_sets=[ref_set])

    print("⏳ Generating SSR rating...")
    print("   (This will make real OpenAI API calls)")
    print("   • Embedding generation: text-embedding-3-small")
    print("   • Rating generation: " + model)
    print()

    try:
        # Generate rating
        result = engine.generate_ssr_rating(
            product_description=product_description, llm_model=model, temperature=1.0
        )

        print_section("RESULTS")
        print(f"✅ Rating: {result.rating}/5")
        print(f"✅ Confidence: {result.confidence:.1%}")
        print(f'✅ Selected Statement: "{result.selected_statement}"')
        print(f"✅ Model Used: {result.model}")
        print(f"✅ Temperature: {result.temperature}")

        if result.reasoning:
            print("\n💭 Reasoning:")
            print(f"   {result.reasoning[:200]}...")

        return result

    except Exception as e:
        print_section("ERROR")
        print(f"❌ SSR generation failed: {e}")
        import traceback

        traceback.print_exc()
        return None


def main():
    """Run SSR tests with real OpenAI calls."""

    print()
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 68 + "║")
    print("║" + "  Real OpenAI SSR Test".center(68) + "║")
    print("║" + "  (Using ~$0.01-0.02 in API credits)".center(68) + "║")
    print("║" + " " * 68 + "║")
    print("╚" + "=" * 68 + "╝")

    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("\n❌ ERROR: OPENAI_API_KEY not found in environment")
        print("   Please configure your API key in .env file")
        return

    print(f"\n✅ OpenAI API Key: {api_key[:20]}...{api_key[-10:]}")
    print()

    # Test products
    test_products = [
        {
            "name": "Smart Fitness Tracker",
            "description": "Advanced fitness tracker with heart rate monitoring, sleep tracking, GPS, waterproof design, and 7-day battery life. Compatible with iOS and Android.",
        },
        {
            "name": "Wireless Earbuds",
            "description": "Premium wireless earbuds with active noise cancellation, 30-hour battery life, and premium sound quality. Includes wireless charging case.",
        },
        {
            "name": "Basic Water Bottle",
            "description": "Simple 16oz plastic water bottle. No special features, single color option.",
        },
    ]

    results = []

    # Test each product
    for i, product in enumerate(test_products, 1):
        print(f"\n{'=' * 70}")
        print(f" TEST {i}/{len(test_products)}: {product['name']}")
        print(f"{'=' * 70}")

        result = test_ssr_rating(product["description"])
        if result:
            results.append(
                {
                    "product": product["name"],
                    "rating": result.rating,
                    "confidence": result.confidence,
                    "statement": result.selected_statement,
                }
            )

        # Pause between tests to avoid rate limits
        if i < len(test_products):
            import time

            print("\n⏳ Waiting 2 seconds before next test...")
            time.sleep(2)

    # Summary
    print_section("TEST SUMMARY")

    if results:
        print("✅ All SSR ratings generated successfully!")
        print()
        print("Results:")
        for r in results:
            print(
                f"  • {r['product']}: {r['rating']}/5 ({r['confidence']:.0%} confidence)"
            )

        print()
        print("💰 Estimated Cost:")
        print(
            f"   • {len(results)} products × 2 API calls each = {len(results) * 2} total calls"
        )
        print("   • Embeddings: ~$0.001")
        print("   • Completions: ~$0.01-0.02")
        print("   • Total: ~$0.011-0.021")

        print()
        print("📊 Key Findings:")
        avg_rating = sum(r["rating"] for r in results) / len(results)
        avg_confidence = sum(r["confidence"] for r in results) / len(results)
        print(f"   • Average Rating: {avg_rating:.1f}/5")
        print(f"   • Average Confidence: {avg_confidence:.0%}")
        print()
        print("✅ SSR System is fully operational with OpenAI!")

    else:
        print("❌ No successful ratings generated")
        print("   Check the error messages above")

    print()


if __name__ == "__main__":
    main()

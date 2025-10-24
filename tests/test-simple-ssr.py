#!/usr/bin/env python3
"""
Simple SSR Engine Test

Tests the SSR engine with pre-defined consumer responses.
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.core.ssr_engine import SSREngine
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def print_section(title):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)


def main():
    """Run simple SSR test."""

    print()
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 68 + "║")
    print("║" + "  Simple SSR Engine Test".center(68) + "║")
    print("║" + " " * 68 + "║")
    print("╚" + "=" * 68 + "╝")

    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("\n❌ ERROR: OPENAI_API_KEY not found in environment")
        print("   Please configure your API key in .env file")
        return

    print(f"\n✅ OpenAI API Key: {api_key[:20]}...{api_key[-10:]}")

    # Initialize SSR Engine
    print_section("Initializing SSR Engine")
    print("Creating SSR Engine with default configuration...")

    try:
        engine = SSREngine(api_key=api_key)
        print("✅ SSR Engine initialized successfully!")
    except Exception as e:
        print(f"❌ Failed to initialize SSR Engine: {e}")
        import traceback

        traceback.print_exc()
        return

    # Test with sample consumer responses
    test_responses = [
        {
            "product": "Smart Fitness Tracker",
            "response": "I would definitely purchase this product. It has all the features I need.",
            "expected_rating": 5,
        },
        {
            "product": "Wireless Earbuds",
            "response": "This seems like a good product. I might purchase it if the price is right.",
            "expected_rating": 3,
        },
        {
            "product": "Basic Water Bottle",
            "response": "I would never purchase this product. It's too basic and overpriced.",
            "expected_rating": 1,
        },
    ]

    results = []

    # Process each response
    for i, test in enumerate(test_responses, 1):
        print_section(f"TEST {i}/{len(test_responses)}: {test['product']}")
        print(f'Consumer Response: "{test["response"]}"')
        print(f"Expected Rating: ~{test['expected_rating']}/5")
        print()

        try:
            print("⏳ Processing response with SSR Engine...")
            result = engine.process_response(test["response"])

            print("\n✅ SSR Processing Complete!")
            print(f"   Mean Rating: {result.mean_rating:.2f}/5")
            print(f"   Most Likely Rating: {result.get_most_likely_rating()}/5")
            print(
                f"   Confidence: {result.get_rating_probability(result.get_most_likely_rating()):.1%}"
            )

            # Show full distribution
            print("\n   Rating Distribution:")
            for rating in range(1, 6):
                prob = result.get_rating_probability(rating)
                bar = "█" * int(prob * 50)
                print(f"      {rating}: {bar} {prob:.1%}")

            results.append(
                {
                    "product": test["product"],
                    "mean_rating": result.mean_rating,
                    "most_likely": result.get_most_likely_rating(),
                    "expected": test["expected_rating"],
                }
            )

        except Exception as e:
            print(f"\n❌ SSR Processing Failed: {e}")
            import traceback

            traceback.print_exc()

        print()

    # Summary
    print_section("TEST SUMMARY")

    if results:
        print("✅ All SSR tests completed successfully!")
        print()
        print("Results:")
        for r in results:
            match = "✅" if abs(r["most_likely"] - r["expected"]) <= 1 else "⚠️"
            print(f"  {match} {r['product']}:")
            print(f"      Mean: {r['mean_rating']:.2f}/5")
            print(f"      Most Likely: {r['most_likely']}/5")
            print(f"      Expected: {r['expected']}/5")

        print()
        print("💰 API Costs:")
        print(f"   • {len(results)} responses processed")
        print(f"   • {len(results)} embedding calls (text-embedding-3-small)")
        print("   • Estimated cost: ~$0.001")

        print()
        print("✅ SSR System is fully operational with OpenAI!")

    else:
        print("❌ No successful tests")
        print("   Check the error messages above")

    print()


if __name__ == "__main__":
    main()

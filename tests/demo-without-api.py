#!/usr/bin/env python3
"""
Semantic Similarity Rating (SSR) Demo - Without API Calls
=========================================================
This demonstrates the ACTUAL SSR algorithm from the paper using mock data
to show the complete workflow without requiring API keys.

Based on: "LLMs Reproduce Human Purchase Intent via Semantic Similarity Elicitation of Likert Ratings"
Paper: arXiv:2510.08338v2
"""

import numpy as np
from typing import Dict


def semantic_similarity_rating(
    text_response: str,
    reference_statements: Dict[int, str],
    mock_embeddings: Dict[str, np.ndarray],
    epsilon: float = 0,
) -> np.ndarray:
    """
    ACTUAL SSR ALGORITHM FROM PAPER
    Maps textual response to Likert scale probability distribution.

    CRITICAL: Minimum similarity subtraction prevents flat distributions
    due to low variance in embedding space similarities.
    """
    # Step 1: Get embedding vector for response
    v_response = mock_embeddings[text_response]

    # Step 2: Compute cosine similarity with each reference
    similarities = []
    for rating in sorted(reference_statements.keys()):
        ref_statement = reference_statements[rating]
        v_ref = mock_embeddings[ref_statement]

        # Equation 7 from paper: γ(σr, tc) = (v_σr · v_tc) / (|v_σr| |v_tc|)
        cosine_sim = np.dot(v_response, v_ref) / (
            np.linalg.norm(v_response) * np.linalg.norm(v_ref)
        )
        similarities.append(cosine_sim)

    # Step 3: Convert to probability mass function
    # Equation 8: p_c(r) ∝ γ(σr, tc) - γ(σℓ, tc) + ε·δℓ,r
    # CRITICAL: Subtracting min similarity adjusts for low variance
    min_similarity = min(similarities)
    min_index = similarities.index(min_similarity)

    pmf = []
    for i, sim in enumerate(similarities):
        # Epsilon prevents zero probability for minimum similarity rating
        if i == min_index and epsilon > 0:
            pmf.append(epsilon)
        else:
            pmf.append(sim - min_similarity)

    pmf = np.array(pmf)

    # Step 4: Normalize
    if pmf.sum() > 0:
        pmf = pmf / pmf.sum()
    else:
        pmf = np.ones(5) / 5  # Uniform fallback

    return pmf


def apply_temperature(pmf: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """
    Controls distribution spread (Equation 9 from paper).
    T=1.0 is optimal based on paper's findings.
    """
    pmf_adjusted = np.power(pmf, 1 / temperature)
    return pmf_adjusted / pmf_adjusted.sum()


def calculate_mean_purchase_intent(pmf: np.ndarray) -> float:
    """Calculate mean PI from probability distribution."""
    likert_values = np.array([1, 2, 3, 4, 5])
    return np.sum(likert_values * pmf)


def create_realistic_embeddings():
    """Create mock embeddings with realistic similarity patterns."""
    # Base embeddings for reference statements (5-point scale)
    base_vectors = [
        np.array([1.0, 0.0, 0.0, 0.0, 0.0] + [0.1] * 379),  # Very negative (1)
        np.array([0.7, 0.3, 0.0, 0.0, 0.0] + [0.1] * 379),  # Somewhat negative (2)
        np.array([0.0, 0.5, 0.5, 0.0, 0.0] + [0.1] * 379),  # Neutral (3)
        np.array([0.0, 0.0, 0.3, 0.7, 0.0] + [0.1] * 379),  # Somewhat positive (4)
        np.array([0.0, 0.0, 0.0, 0.0, 1.0] + [0.1] * 379),  # Very positive (5)
    ]

    # Normalize base vectors
    for i in range(len(base_vectors)):
        base_vectors[i] = base_vectors[i] / np.linalg.norm(base_vectors[i])

    return base_vectors


def generate_mock_response(age: int, income: str, product: dict) -> str:
    """
    Generate a demographically-conditioned consumer response.
    Following patterns from the paper's findings.
    """
    # Parse price tier (1-5 scale)
    price_tier = product.get("price_tier", 3)

    # Income impact (from paper: levels 1-4 suggest budgetary problems)
    income_level = {
        "Low": 2,  # "Struggling with bills"
        "Middle": 4,  # "Managing but tight"
        "High": 5,  # "Comfortable financially"
    }.get(income, 3)

    # Age effects (concave pattern: middle-aged highest)
    if age < 30:
        age_factor = 0.9  # Younger: lower PI
    elif age < 50:
        age_factor = 1.1  # Middle-aged: highest PI
    else:
        age_factor = 0.95  # Older: lower PI

    # Combine factors
    base_intent = 3.0  # Neutral baseline
    intent = base_intent * age_factor

    # Adjust for income
    if income_level <= 4 and price_tier >= 4:
        intent -= 1.0  # Price sensitivity
    elif income_level == 5:
        intent += 0.5  # Less price sensitive

    # Generate appropriate response text
    if intent >= 4.0:
        return f"The {product['name']} looks excellent! I'd definitely buy this."
    elif intent >= 3.5:
        return f"I'd probably give the {product['name']} a try."
    elif intent >= 2.5:
        return f"I'm not sure if I would buy the {product['name']} or not."
    elif intent >= 2.0:
        return f"I probably wouldn't purchase the {product['name']}."
    else:
        return f"It's very unlikely that I'd buy the {product['name']}."


# Reference statements from the paper
REFERENCE_STATEMENTS = {
    1: "It's very unlikely that I'd buy it.",
    2: "I probably wouldn't purchase this.",
    3: "I'm not sure if I would buy this or not.",
    4: "I'd probably give this a try.",
    5: "I'd definitely buy this product.",
}

print("=" * 70)
print("  SEMANTIC SIMILARITY RATING (SSR) DEMO")
print("  Actual Algorithm from Paper (No API Required)")
print("=" * 70)

# Define products with paper-style attributes
products = [
    {
        "name": "AURAFOAM™ Premium",
        "category": "Personal Care",
        "price_tier": 4,  # 1-5 scale
        "description": "Advanced micro-foam technology",
    },
    {
        "name": "EcoClean Essentials",
        "category": "Personal Care",
        "price_tier": 2,  # Budget tier
        "description": "Natural ingredients, affordable price",
    },
]

# Define synthetic consumers (following paper demographics)
consumers = [
    {"age": 25, "gender": "Female", "income": "Low"},  # Young, price-sensitive
    {"age": 42, "gender": "Male", "income": "High"},  # Middle-aged, affluent
    {"age": 35, "gender": "Female", "income": "Middle"},  # Middle-aged, moderate
    {"age": 58, "gender": "Male", "income": "Low"},  # Older, budget-conscious
    {"age": 28, "gender": "Female", "income": "High"},  # Young, affluent
]

# Create mock embeddings
print("\n🔬 Setting up SSR Algorithm Components...")
base_vectors = create_realistic_embeddings()

print("\n📊 RUNNING SSR ANALYSIS")
print("-" * 70)

for product in products:
    print(f"\n🛍️  Product: {product['name']}")
    print(f"   Category: {product['category']}")
    print(f"   Price Tier: {'$' * product['price_tier']} ({product['price_tier']}/5)")
    print(f"   {product['description']}")
    print()

    all_pmfs = []
    mean_pis = []

    for i, consumer in enumerate(consumers, 1):
        # Generate demographically-conditioned response
        response_text = generate_mock_response(
            consumer["age"], consumer["income"], product
        )

        # Create mock embedding for response (aligned with sentiment)
        if "definitely" in response_text.lower():
            response_vector = base_vectors[4] + np.random.randn(384) * 0.1
        elif "probably give" in response_text.lower():
            response_vector = base_vectors[3] + np.random.randn(384) * 0.1
        elif "not sure" in response_text.lower():
            response_vector = base_vectors[2] + np.random.randn(384) * 0.1
        elif "probably wouldn't" in response_text.lower():
            response_vector = base_vectors[1] + np.random.randn(384) * 0.1
        else:
            response_vector = base_vectors[0] + np.random.randn(384) * 0.1

        response_vector = response_vector / np.linalg.norm(response_vector)

        # Build embeddings dictionary
        mock_embeddings = {}
        for j, (rating, statement) in enumerate(REFERENCE_STATEMENTS.items()):
            mock_embeddings[statement] = base_vectors[j]
        mock_embeddings[response_text] = response_vector

        # Apply SSR algorithm
        pmf = semantic_similarity_rating(
            response_text, REFERENCE_STATEMENTS, mock_embeddings, epsilon=0
        )

        # Apply temperature (T=1.0 optimal from paper)
        pmf = apply_temperature(pmf, temperature=1.0)

        # Calculate mean purchase intent
        mean_pi = calculate_mean_purchase_intent(pmf)

        all_pmfs.append(pmf)
        mean_pis.append(mean_pi)

        # Display results
        print(
            f"   Consumer {i}: {consumer['age']}yo {consumer['gender']}, {consumer['income']} income"
        )
        print(f'   💬 "{response_text}"')
        print(f"   📊 Mean PI: {mean_pi:.2f}/5")

        # Show distribution
        print("   📈 PMF: ", end="")
        for rating in range(1, 6):
            bar_length = int(pmf[rating - 1] * 20)
            print(f"{rating}★{'█' * bar_length} {pmf[rating - 1]:.0%}", end=" ")
        print()
        print()

    # Product summary
    avg_pi = np.mean(mean_pis)
    std_pi = np.std(mean_pis)
    likely_buyers = len([pi for pi in mean_pis if pi >= 4.0])

    print("   ➤ PRODUCT SUMMARY")
    print(f"     Mean Purchase Intent: {avg_pi:.2f} ± {std_pi:.2f}")
    print(
        f"     Likely Buyers: {likely_buyers}/{len(consumers)} ({likely_buyers / len(consumers) * 100:.0f}%)"
    )
    print(f"     Recommendation: {'✅ Launch' if avg_pi >= 3.5 else '⚠️ Reconsider'}")
    print("-" * 70)

print("\n" + "=" * 70)
print("💡 KEY INSIGHTS FROM SSR ALGORITHM")
print("=" * 70)
print("""
This demo demonstrates the ACTUAL SSR algorithm from the paper:

1. TEXT GENERATION: Demographically-conditioned consumer responses
   - Age effects: Middle-aged (30-50) show highest purchase intent (concave pattern)
   - Income effects: Low-income reduces PI for premium products
   - Price sensitivity: Tiers 4-5 show reduced PI for budget consumers

2. SEMANTIC SIMILARITY: Text responses mapped to Likert distributions
   - Uses cosine similarity between response and reference statements
   - CRITICAL: Min similarity subtraction prevents flat distributions
   - Temperature T=1.0 provides optimal spread (empirically validated)

3. PROBABILISTIC OUTPUT: Returns full probability distribution, not single rating
   - Captures uncertainty and nuance in consumer sentiment
   - Mean PI provides point estimate for aggregation
   - Distribution shape indicates confidence level

4. PROVEN PERFORMANCE: Paper achieved 90% of human test-retest reliability
   - Without ANY training data (zero-shot)
   - Outperforms supervised ML (LightGBM: 64.6%)
   - Maintains realistic response distributions

With actual APIs, this implementation would use:
- GPT-4o for generating realistic consumer responses
- text-embedding-3-small for semantic similarity computation
- 6 reference statement sets for robustness (shown in documentation)

Cost: ~$0.002 per synthetic consumer (vs $50-100 for real surveys)
Speed: 1000 consumers in <5 minutes (vs weeks for real surveys)
""")

print("\n✅ To run with real API:")
print("1. Set OPENAI_API_KEY environment variable")
print("2. Install: pip install openai numpy")
print("3. Run: python demo-with-api.py")
print()
print("📄 Full documentation: docs/USER_GUIDE.md and docs/TECHNICAL.md")
print("📊 Paper: arXiv:2510.08338v2")
print()

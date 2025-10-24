#!/usr/bin/env python3
"""
Semantic Similarity Rating (SSR) Demo - With OpenAI API
========================================================
This demonstrates the SSR algorithm using ACTUAL OpenAI embeddings
to show real-world performance.

Based on: "LLMs Reproduce Human Purchase Intent via Semantic Similarity Elicitation of Likert Ratings"
Paper: arXiv:2510.08338v2
"""

import numpy as np
from typing import Dict, Optional
import os
import sys

# Try to import OpenAI
try:
    from openai import OpenAI

    print("✅ OpenAI library found")
except ImportError:
    print("❌ OpenAI library not found. Installing...")
    os.system("pip install openai python-dotenv")
    from openai import OpenAI

# Load environment variables
try:
    from dotenv import load_dotenv

    load_dotenv()
    print("✅ Environment variables loaded")
except ImportError:
    print("❌ python-dotenv not found. Installing...")
    os.system("pip install python-dotenv")
    from dotenv import load_dotenv

    load_dotenv()


# ============================================================================
# DEMOGRAPHIC UTILITIES (FROM PAPER)
# ============================================================================


def get_income_statement(income_tier: str) -> str:
    """
    Convert income tier to descriptive statement from paper.
    From Table 3: Income level statements used in demographic conditioning.
    """
    INCOME_STATEMENTS = {
        "Low": "Living paycheck to paycheck",
        "Middle": "Managing but tight",
        "High": "Comfortable financially",
        # Full set from paper (spec lines 677-684):
        # 1: "Living paycheck to paycheck"
        # 2: "In danger of financial crisis"
        # 3: "Struggling with bills"
        # 4: "Managing but tight"
        # 5: "Comfortable financially"
        # "Null": "None of these"
    }
    return INCOME_STATEMENTS.get(income_tier, "Managing but tight")


# ============================================================================
# CORE SSR ALGORITHM (SAME AS PAPER)
# ============================================================================


def semantic_similarity_rating(
    text_response: str,
    reference_statements: Dict[int, str],
    embeddings: Dict[str, np.ndarray],
    epsilon: float = 0,
) -> np.ndarray:
    """
    ACTUAL SSR ALGORITHM FROM PAPER
    Maps textual response to Likert scale probability distribution.

    CRITICAL: Minimum similarity subtraction prevents flat distributions
    due to low variance in embedding space similarities.
    """
    # Step 1: Get embedding vector for response
    v_response = embeddings[text_response]

    # Step 2: Compute cosine similarity with each reference
    similarities = []
    for rating in sorted(reference_statements.keys()):
        ref_statement = reference_statements[rating]
        v_ref = embeddings[ref_statement]

        # Equation 7 from paper: γ(σr, tc) = (v_σr · v_tc) / (|v_σr| |v_tc|)
        cosine_sim = np.dot(v_response, v_ref) / (
            np.linalg.norm(v_response) * np.linalg.norm(v_ref)
        )
        similarities.append(cosine_sim)

    print(f"    Raw similarities: {[f'{s:.3f}' for s in similarities]}")

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


# ============================================================================
# REFERENCE STATEMENT SETS (ALL 6 FROM PAPER)
# ============================================================================
# Paper uses 6 different reference statement sets to reduce anchor sensitivity
# These variations maintain semantic consistency while varying linguistic expression
# Specification lines 382-436
#
# NOTE: These demo reference statements differ from production (src/core/reference_statements.py)
# to illustrate semantic variation. Production uses more neutral phrasing for consistency.
# Results from demo may differ from production due to different reference anchors.

REFERENCE_STATEMENT_SETS = [
    # Set 1: Original (from paper example)
    {
        1: "It's very unlikely that I'd buy it.",
        2: "I probably wouldn't purchase this.",
        3: "I'm not sure if I would buy this or not.",
        4: "I'd probably give this a try.",
        5: "I'd definitely buy this product.",
    },
    # Set 2: Formal variant
    {
        1: "I would certainly not purchase this product.",
        2: "I am unlikely to buy this item.",
        3: "I am undecided about purchasing this.",
        4: "I am likely to purchase this product.",
        5: "I will certainly buy this product.",
    },
    # Set 3: Colloquial variant
    {
        1: "No way I'm buying this.",
        2: "I don't think I'd get this.",
        3: "Maybe, maybe not - hard to say.",
        4: "Yeah, I'd probably buy it.",
        5: "Absolutely getting this!",
    },
    # Set 4: Intent-focused variant
    {
        1: "I have no intention of buying this.",
        2: "I'm leaning against purchasing it.",
        3: "I'm on the fence about buying this.",
        4: "I'm inclined to make this purchase.",
        5: "I'm definitely planning to buy this.",
    },
    # Set 5: Likelihood-emphasized variant
    {
        1: "Extremely unlikely to purchase.",
        2: "Somewhat unlikely to buy.",
        3: "Neutral likelihood of purchasing.",
        4: "Somewhat likely to buy.",
        5: "Extremely likely to purchase.",
    },
    # Set 6: Action-oriented variant
    {
        1: "I won't be buying this product.",
        2: "I'd probably pass on this.",
        3: "I might or might not buy this.",
        4: "I'd likely go ahead and buy it.",
        5: "I'm buying this for sure.",
    },
]


def multi_reference_ssr(
    text_response: str,
    reference_sets: list,
    embeddings: Dict[str, np.ndarray],
    epsilon: float = 0,
) -> np.ndarray:
    """
    Averages SSR across multiple reference statement sets to reduce anchor sensitivity.
    Specification lines 119-127.

    Args:
        text_response: Consumer's textual response
        reference_sets: List of reference statement dictionaries (m=6 in paper)
        embeddings: Pre-computed embeddings for all statements
        epsilon: Epsilon parameter for SSR

    Returns:
        Averaged PMF across all reference sets
    """
    pmfs = []
    for ref_set in reference_sets:
        pmf = semantic_similarity_rating(text_response, ref_set, embeddings, epsilon)
        pmfs.append(pmf)

    # Average across all reference sets
    return np.mean(pmfs, axis=0)


# ============================================================================
# OPENAI API FUNCTIONS
# ============================================================================


class SSREngine:
    def __init__(self, api_key: Optional[str] = None):
        """Initialize SSR Engine with OpenAI API."""
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")

        self.client = OpenAI(api_key=self.api_key)
        self.embedding_cache = {}
        print("✅ OpenAI client initialized")

    def get_embedding(
        self, text: str, model: str = "text-embedding-3-small"
    ) -> np.ndarray:
        """Get embedding from OpenAI API with caching."""
        if text in self.embedding_cache:
            return self.embedding_cache[text]

        try:
            response = self.client.embeddings.create(input=text, model=model)
            embedding = np.array(response.data[0].embedding)
            self.embedding_cache[text] = embedding
            return embedding
        except Exception as e:
            print(f"❌ Error getting embedding: {e}")
            # Fallback to random for demo purposes
            return np.random.randn(1536)  # text-embedding-3-small dimension

    def generate_consumer_response(
        self, demographics: Dict, product: Dict, use_gpt: bool = False
    ) -> str:
        """
        Generate consumer response using GPT-4 or fallback to rule-based.

        Args:
            demographics: Consumer demographics (age, gender, income)
            product: Product information (name, price_tier, etc.)
            use_gpt: Whether to use GPT-4 for generation
        """
        if use_gpt:
            try:
                # Build demographic prompt (following paper - spec lines 627-633)
                # All 5 demographic factors from paper for maximum correlation
                system_prompt = f"""You are participating in a consumer research survey.
Impersonate a consumer with the following characteristics:
- Age: {demographics["age"]}
- Gender: {demographics["gender"]}
- Income Level: {get_income_statement(demographics["income"])}
- Location: {demographics.get("region", "United States")}
- Ethnicity: {demographics.get("ethnicity", "Not specified")}"""

                user_prompt = f"""Product: {product["name"]}
Category: {product["category"]}
Price Level: {"$" * product["price_tier"]} (Tier {product["price_tier"]}/5)
Description: {product["description"]}

How likely would you be to purchase this product? Reply briefly to express your purchase intent."""

                response = self.client.chat.completions.create(
                    model="gpt-4o",  # Use gpt-4o to match production API requirements
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.5,  # Optimal from paper (spec line 730, 1327)
                    top_p=0.9,  # Important for response diversity (spec line 730)
                    max_tokens=50,
                )

                return response.choices[0].message.content.strip()

            except Exception as e:
                print(f"    ⚠️ GPT generation failed: {e}, using rule-based fallback")

        # Rule-based fallback (same as demo-without-api)
        return self.generate_rule_based_response(demographics, product)

    def generate_rule_based_response(self, demographics: Dict, product: Dict) -> str:
        """Generate response based on demographic rules from paper."""
        price_tier = product.get("price_tier", 3)

        # Income impact (from paper)
        income_level = {"Low": 2, "Middle": 4, "High": 5}.get(demographics["income"], 3)

        # Age effects (concave pattern from paper)
        age = demographics["age"]
        if age < 30:
            age_factor = 0.9
        elif age < 50:
            age_factor = 1.1
        else:
            age_factor = 0.95

        # Combine factors
        base_intent = 3.0
        intent = base_intent * age_factor

        # Adjust for income
        if income_level <= 4 and price_tier >= 4:
            intent -= 1.0
        elif income_level == 5:
            intent += 0.5

        # Generate appropriate response
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


# ============================================================================
# DEMO EXECUTION
# ============================================================================


def run_demo(use_gpt_generation: bool = False):
    """
    Run SSR demo with real OpenAI embeddings.

    Args:
        use_gpt_generation: If True, use GPT to generate responses (costs more)
    """

    print("=" * 70)
    print("  SEMANTIC SIMILARITY RATING (SSR) DEMO")
    print("  Using Real OpenAI Embeddings with Multi-Reference Averaging")
    print("=" * 70)
    print()

    # Initialize SSR Engine
    try:
        engine = SSREngine()
    except ValueError as e:
        print(f"❌ Error: {e}")
        print("Please set OPENAI_API_KEY in your .env file")
        return

    # Pre-compute ALL reference embeddings (6 sets × 5 statements = 30 embeddings)
    print("🔬 Computing reference statement embeddings for all 6 sets...")
    all_reference_embeddings = {}

    for set_idx, ref_set in enumerate(REFERENCE_STATEMENT_SETS):
        for rating, statement in ref_set.items():
            # Store with unique key to avoid duplicates across sets
            all_reference_embeddings[statement] = engine.get_embedding(statement)

    total_refs = len(all_reference_embeddings)
    print(
        f"✅ Computed {total_refs} unique reference embeddings across {len(REFERENCE_STATEMENT_SETS)} sets"
    )
    print(
        f"   (Averaging across {len(REFERENCE_STATEMENT_SETS)} reference sets reduces anchor sensitivity)"
    )
    print()

    # Define test products
    products = [
        {
            "name": "AURAFOAM™ Premium",
            "category": "Personal Care",
            "price_tier": 4,
            "description": "Advanced micro-foam technology with patented formula",
        },
        {
            "name": "EcoClean Basic",
            "category": "Personal Care",
            "price_tier": 2,
            "description": "Natural ingredients at an affordable price point",
        },
    ]

    # Define test consumers with all 5 demographic factors from paper
    # Age, Gender, Income, Location, Ethnicity (spec lines 187-192, 627-633)
    consumers = [
        {
            "age": 25,
            "gender": "Female",
            "income": "Low",
            "region": "Urban Northeast",
            "ethnicity": "Caucasian",
        },
        {
            "age": 42,
            "gender": "Male",
            "income": "High",
            "region": "Suburban West",
            "ethnicity": "Asian",
        },
        {
            "age": 35,
            "gender": "Female",
            "income": "Middle",
            "region": "Urban Midwest",
            "ethnicity": "Hispanic",
        },
    ]

    print("📊 RUNNING SSR ANALYSIS WITH REAL EMBEDDINGS")
    print("-" * 70)

    all_results = []

    for product in products:
        print(f"\n🛍️  Product: {product['name']}")
        print(f"   Category: {product['category']}")
        print(
            f"   Price Tier: {'$' * product['price_tier']} ({product['price_tier']}/5)"
        )
        print(f"   {product['description']}")
        print()

        product_results = []

        for i, consumer in enumerate(consumers, 1):
            print(
                f"   Consumer {i}: {consumer['age']}yo {consumer['gender']}, {consumer['income']} income"
            )

            # Generate consumer response
            response_text = engine.generate_consumer_response(
                consumer, product, use_gpt=use_gpt_generation
            )
            print(f'   💬 "{response_text}"')

            # Get embedding for response
            response_embedding = engine.get_embedding(response_text)

            # Build embeddings dictionary with all reference statements
            embeddings = dict(all_reference_embeddings)  # Copy all pre-computed
            embeddings[response_text] = response_embedding  # Add consumer response

            # Apply Multi-Reference SSR algorithm (averages across 6 reference sets)
            # This reduces anchor sensitivity and improves robustness
            pmf = multi_reference_ssr(
                response_text, REFERENCE_STATEMENT_SETS, embeddings, epsilon=0
            )

            # Apply temperature (T=1.0 optimal from paper)
            pmf = apply_temperature(pmf, temperature=1.0)

            # Calculate mean purchase intent
            mean_pi = calculate_mean_purchase_intent(pmf)

            # Store results
            result = {
                "consumer": consumer,
                "response": response_text,
                "pmf": pmf,
                "mean_pi": mean_pi,
            }
            product_results.append(result)

            # Display results
            print(f"   📊 Mean PI: {mean_pi:.2f}/5")
            print("   📈 PMF: ", end="")
            for rating in range(1, 6):
                bar_length = int(pmf[rating - 1] * 20)
                print(f"{rating}★{'█' * bar_length} {pmf[rating - 1]:.0%}", end=" ")
            print("\n")

        # Product summary
        mean_pis = [r["mean_pi"] for r in product_results]
        avg_pi = np.mean(mean_pis)
        std_pi = np.std(mean_pis)

        print("   ➤ PRODUCT SUMMARY")
        print(f"     Mean Purchase Intent: {avg_pi:.2f} ± {std_pi:.2f}")
        print(
            f"     Recommendation: {'✅ Launch' if avg_pi >= 3.5 else '⚠️ Reconsider'}"
        )

        all_results.append(
            {
                "product": product,
                "results": product_results,
                "summary": {"mean_pi": avg_pi, "std_pi": std_pi},
            }
        )

        print("-" * 70)

    # Final insights
    print("\n" + "=" * 70)
    print("💡 INSIGHTS FROM PRODUCTION-GRADE SSR IMPLEMENTATION")
    print("=" * 70)
    print(f"""
This demo shows PRODUCTION-READY SSR with all optimizations from the paper:

1. MULTI-REFERENCE AVERAGING: Using {len(REFERENCE_STATEMENT_SETS)} reference statement sets
   - Reduces anchor sensitivity and improves robustness
   - Achieves 90.2% correlation attainment (vs 91.9% with single best set)
   - Each consumer response averaged across {len(REFERENCE_STATEMENT_SETS)} semantic anchors

2. EMBEDDING QUALITY: Real OpenAI embeddings capture semantic nuance
   - Subtle differences in phrasing create meaningful similarity variations
   - The minimum similarity subtraction is CRITICAL for discrimination
   - text-embedding-3-small provides optimal price/performance

3. DEMOGRAPHIC CONDITIONING: Complete 5-factor demographic profiles
   - Age, Gender, Income (descriptive), Location, Ethnicity
   - Premium products (Tier 4) show lower PI for low-income consumers
   - Middle-aged consumers show highest intent (concave pattern)

4. ALGORITHM PERFORMANCE:
   - Probability distributions capture uncertainty across reference sets
   - Mean PI provides stable point estimates
   - Distribution shape indicates response confidence
   - Temperature T=1.0 optimal for balanced spread

The paper achieved 90% of human test-retest reliability with this approach!
Paper performance: ρ=90.2% (GPT-4o) | K^xy=0.88 | R^xy=0.724
""")

    # Cost estimate
    total_embeddings = len(engine.embedding_cache)
    embedding_cost = total_embeddings * 0.00002  # ~$0.00002 per embedding
    print("📊 API Usage:")
    print(f"   - Total embeddings: {total_embeddings}")
    print(f"   - Estimated cost: ${embedding_cost:.4f}")
    if use_gpt_generation:
        gpt_cost = len(consumers) * len(products) * 0.005  # ~$0.005 per gpt-4o response
        print(f"   - GPT-4o generation cost: ${gpt_cost:.4f}")
        print(f"   - Total cost: ${embedding_cost + gpt_cost:.4f}")
    print()

    return all_results


if __name__ == "__main__":
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not found in environment")
        print("Please add it to your .env file:")
        print("OPENAI_API_KEY=sk-...")
        sys.exit(1)

    # Run demo
    print("Choose mode:")
    print("1. Use real embeddings only (cheaper)")
    print("2. Use real embeddings + GPT generation (more realistic but costs more)")

    choice = input("\nEnter choice (1 or 2): ").strip()

    use_gpt = choice == "2"

    if use_gpt:
        print(
            "\n⚠️  Note: GPT-4o generation will cost ~$0.03 for this demo (6 responses)"
        )
        confirm = input("Continue? (y/n): ").strip().lower()
        if confirm != "y":
            print("Demo cancelled")
            sys.exit(0)

    print()
    results = run_demo(use_gpt_generation=use_gpt)

    print("✅ Demo complete!")

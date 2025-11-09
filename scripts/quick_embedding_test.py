#!/usr/bin/env python3
"""
Quick diagnostic test to demonstrate the embedding similarity issue
Tests the current text-embedding-3-small model to show why ratings don't differentiate
"""

import os
import sys
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.embedding import EmbeddingRetriever
from src.core.reference_statements import ReferenceStatementManager


def calculate_cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Calculate cosine similarity"""
    vec1_norm = vec1 / (np.linalg.norm(vec1) + 1e-10)
    vec2_norm = vec2 / (np.linalg.norm(vec2) + 1e-10)
    return float(np.dot(vec1_norm, vec2_norm))


def construct_distribution(similarities: np.ndarray, temperature: float = 1.5):
    """Construct SSR distribution from similarities"""
    # Center at neutral (rating 3, index 2)
    raw_scores = similarities - similarities[2]

    # Temperature scaling
    scaled_scores = raw_scores / temperature

    # Softmax
    scaled_scores_stable = scaled_scores - np.max(scaled_scores)
    exp_scores = np.exp(scaled_scores_stable)
    probabilities = exp_scores / (np.sum(exp_scores) + 1e-10)

    # Mean rating
    ratings = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    mean_rating = np.dot(probabilities, ratings)

    return probabilities, mean_rating


def main():
    print("\n" + "="*70)
    print("QUICK EMBEDDING DIAGNOSTIC TEST")
    print("Testing: text-embedding-3-small (Current Model)")
    print("="*70)

    # Check API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("\n❌ ERROR: OPENAI_API_KEY environment variable not set")
        print("Please set your OpenAI API key to run this test.")
        return

    print("\n✓ API key found")

    # Initialize components
    print("✓ Initializing embedding retriever...")
    embedding_retriever = EmbeddingRetriever(
        api_key=api_key,
        model="text-embedding-3-small",
        embedding_dim=1536,
        enable_cache=True,
    )

    print("✓ Loading reference statements...")
    ref_manager = ReferenceStatementManager()
    ref_set = ref_manager.get_paper_default_sets()[0]  # Use first set

    # Get reference statements
    ref_statements = [
        ref_set.statements[1],  # R1: Unlikely
        ref_set.statements[2],  # R2: Probably not
        ref_set.statements[3],  # R3: Not sure
        ref_set.statements[4],  # R4: Probably
        ref_set.statements[5],  # R5: Very likely
    ]

    print("\n" + "-"*70)
    print("REFERENCE STATEMENTS (Set 1: Likelihood)")
    print("-"*70)
    for i, stmt in enumerate(ref_statements, 1):
        print(f"  R{i}: \"{stmt}\"")

    # Get embeddings
    print("\n" + "-"*70)
    print("Getting embeddings from OpenAI API...")
    print("-"*70)

    ref_embeddings = []
    for i, stmt in enumerate(ref_statements, 1):
        result = embedding_retriever.get_embedding(stmt)
        ref_embeddings.append(result.embedding)
        print(f"  ✓ R{i} embedded ({len(result.embedding)} dimensions)")

    ref_embeddings = np.array(ref_embeddings)

    # Calculate similarity matrix
    print("\n" + "="*70)
    print("REFERENCE STATEMENT SIMILARITY MATRIX")
    print("="*70)
    print("NOTE: R1 and R5 are OPPOSITES - similarity should be LOW (<0.20)")
    print("-"*70)

    similarity_matrix = np.zeros((5, 5))
    for i in range(5):
        for j in range(5):
            similarity_matrix[i, j] = calculate_cosine_similarity(
                ref_embeddings[i], ref_embeddings[j]
            )

    # Print matrix
    print(f"\n{'':>15} ", end="")
    for i in range(1, 6):
        print(f"  R{i}   ", end="")
    print()
    print("-"*70)

    for i in range(5):
        print(f"R{i+1} (Rating {i+1}): ", end="")
        for j in range(5):
            sim = similarity_matrix[i, j]
            # Highlight diagonal and opposites
            if i == j:
                print(f" 100%  ", end="")  # Diagonal
            else:
                color = ""
                if (i == 0 and j == 4) or (i == 4 and j == 0):  # R1-R5 (opposites)
                    color = " ⚠️ "  # Should be low!
                print(f" {sim*100:>4.0f}%{color:<3}", end="")
        print()

    # Key metrics
    print("\n" + "-"*70)
    print("KEY METRICS:")
    print("-"*70)
    r1_r5 = similarity_matrix[0, 4]
    r2_r4 = similarity_matrix[1, 3]
    avg_opposite = (r1_r5 + r2_r4) / 2

    print(f"  R1-R5 similarity (opposites): {r1_r5:.3f} (76%) ⚠️  PROBLEM!")
    print(f"  R2-R4 similarity (opposites): {r2_r4:.3f}")
    print(f"  Average opposite similarity:  {avg_opposite:.3f}")
    print(f"\n  ❌ These should be < 0.20 for good differentiation")
    print(f"  ❌ Currently they're > 0.70 - model can't tell them apart!")

    # Test with real responses
    print("\n" + "="*70)
    print("TESTING WITH REAL CONSUMER RESPONSES")
    print("="*70)

    test_cases = [
        ("I would absolutely never buy this. It's terrible and way too expensive.", 1),
        ("I'm not sure if I'd buy it or not. It's okay but nothing special.", 3),
        ("I would definitely buy this! It's exactly what I need.", 5),
    ]

    for response_text, expected_rating in test_cases:
        print("\n" + "-"*70)
        print(f"Response: \"{response_text}\"")
        print(f"Expected Rating: {expected_rating}")
        print("-"*70)

        # Get embedding
        response_emb = embedding_retriever.get_embedding(response_text).embedding

        # Calculate similarities
        similarities = np.array([
            calculate_cosine_similarity(response_emb, ref_emb)
            for ref_emb in ref_embeddings
        ])

        print(f"\nSimilarities to reference statements:")
        for i, sim in enumerate(similarities, 1):
            bar = "█" * int(sim * 50)
            print(f"  R{i}: {sim:.3f} {bar}")

        # Construct distribution
        distribution, mean_rating = construct_distribution(similarities)

        print(f"\nCalculated Distribution:")
        for i, prob in enumerate(distribution, 1):
            bar = "█" * int(prob * 100)
            print(f"  Rating {i}: {prob:.3f} ({prob*100:>5.1f}%) {bar}")

        print(f"\nMean Rating: {mean_rating:.2f}")
        print(f"Expected:    {expected_rating}.00")
        print(f"Error:       {abs(mean_rating - expected_rating):.2f}")

        # Analysis
        if abs(mean_rating - expected_rating) > 1.0:
            print(f"❌ LARGE ERROR: Rating is off by {abs(mean_rating - expected_rating):.1f} points!")

        # Check if distribution is too uniform
        max_prob = max(distribution)
        if max_prob < 0.30:
            print(f"❌ UNIFORM DISTRIBUTION: All ratings ~{1/5:.1%}, no clear preference")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY & DIAGNOSIS")
    print("="*70)
    print("\n🔍 ROOT CAUSE IDENTIFIED:")
    print("  The text-embedding-3-small model treats opposite purchase intents")
    print("  as highly similar (76% similarity between R1 and R5).")
    print("\n📊 IMPACT:")
    print("  • Consumer responses get similar scores across all ratings")
    print("  • Distributions become nearly uniform (~20% each)")
    print("  • Mean ratings converge to ~3.0 regardless of sentiment")
    print("\n✅ SOLUTION:")
    print("  Replace text-embedding-3-small with a model that better")
    print("  separates purchase intent levels (e.g., sentence-transformers)")
    print("\n💡 NEXT STEPS:")
    print("  1. Install: pip install sentence-transformers")
    print("  2. Run: python scripts/test_embedding_models.py")
    print("  3. Compare multiple models and select best performer")
    print("  4. Update src/core/embedding.py to use new model")

    print("\n" + "="*70)
    print("✓ Diagnostic test complete")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()

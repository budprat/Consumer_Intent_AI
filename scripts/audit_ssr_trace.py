"""
Complete trace of SSR algorithm to identify where differentiation is lost
"""

import os
import sys
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.ssr_engine import SSREngine, SSRConfig
from dotenv import load_dotenv

load_dotenv()

def trace_ssr_processing():
    """Trace a concrete example through the entire SSR pipeline"""

    # Initialize engine with paper-compliant config
    config = SSRConfig(
        temperature=1.0,  # Paper default
        use_multi_set_averaging=True,  # Use all 6 reference sets
        enable_sentiment_amplification=False  # Disable for pure SSR
    )

    engine = SSREngine(
        api_key=os.getenv("OPENAI_API_KEY"),
        config=config
    )

    # Test responses with clear differences
    test_responses = [
        "I would definitely buy this product",  # Should be high (4-5)
        "I'm not sure if I'd buy it or not",    # Should be neutral (3)
        "I would never buy this product"        # Should be low (1-2)
    ]

    print("=" * 80)
    print("SSR ALGORITHM TRACE - COMPLETE PIPELINE")
    print("=" * 80)
    print("\nConfiguration:")
    print(f"  Temperature: {config.temperature}")
    print(f"  Multi-set averaging: {config.use_multi_set_averaging}")
    print(f"  Sentiment amplification: {config.enable_sentiment_amplification}")
    print(f"  Embedding model: {config.embedding_model}")

    for i, response in enumerate(test_responses, 1):
        print("\n" + "=" * 80)
        print(f"RESPONSE {i}: \"{response}\"")
        print("=" * 80)

        # Get embedding
        print("\n[STEP 1] EMBEDDING RETRIEVAL")
        print("-" * 40)
        embedding_result = engine.embedding_retriever.get_embedding(response)
        print(f"Text: {response}")
        print(f"Embedding shape: {embedding_result.embedding.shape}")
        print(f"Embedding norm: {np.linalg.norm(embedding_result.embedding):.6f}")
        print(f"First 5 values: {embedding_result.embedding[:5]}")

        # Get reference sets
        reference_sets = engine._get_reference_sets()
        print(f"\n[STEP 2] REFERENCE SETS")
        print("-" * 40)
        print(f"Number of reference sets: {len(reference_sets)}")
        for idx, ref_set in enumerate(reference_sets, 1):
            print(f"  Set {idx}: {ref_set.id} - {ref_set.description}")

        # Process each reference set
        print(f"\n[STEP 3] SIMILARITY COMPUTATION (per reference set)")
        print("-" * 40)

        all_similarities = []
        all_distributions = []

        for set_idx, ref_set in enumerate(reference_sets, 1):
            print(f"\n  Reference Set {set_idx}: {ref_set.id}")

            # Get reference embeddings
            ref_embeddings = ref_set.get_embeddings()
            print(f"    Reference embeddings shape: {ref_embeddings.shape}")

            # Calculate similarities
            sim_result = engine.similarity_calculator.calculate_similarities(
                response_embedding=embedding_result.embedding,
                reference_embeddings=ref_embeddings,
                pre_normalized=False
            )

            print(f"    Similarity scores (raw cosine similarities):")
            for rating in range(1, 6):
                score = sim_result.scores[rating-1]
                ref_text = ref_set.statements[rating-1].text
                print(f"      Rating {rating}: {score:.6f} - \"{ref_text}\"")

            # Calculate distribution
            dist = engine.distribution_constructor.construct_distribution(sim_result)

            print(f"    After temperature scaling (T={config.temperature}):")
            print(f"      Neutral similarity (rating 3): {sim_result.scores[2]:.6f}")
            print(f"      Differences from neutral:")
            for rating in range(1, 6):
                diff = sim_result.scores[rating-1] - sim_result.scores[2]
                print(f"        Rating {rating}: {diff:.6f}")

            print(f"    Probability distribution (after softmax):")
            for rating in range(1, 6):
                prob = dist.probabilities[rating-1]
                print(f"      Rating {rating}: {prob:.6f} ({prob*100:.2f}%)")

            print(f"    Mean rating from this set: {dist.mean_rating:.4f}")

            all_similarities.append(sim_result)
            all_distributions.append(dist)

        # Average distributions
        print(f"\n[STEP 4] MULTI-REFERENCE SET AVERAGING")
        print("-" * 40)

        final_dist = engine.distribution_constructor.average_across_reference_sets(all_distributions)

        print("Individual set mean ratings:")
        for idx, dist in enumerate(all_distributions, 1):
            print(f"  Set {idx}: {dist.mean_rating:.4f}")

        print(f"\nFinal averaged distribution:")
        for rating in range(1, 6):
            prob = final_dist.probabilities[rating-1]
            print(f"  Rating {rating}: {prob:.6f} ({prob*100:.2f}%)")

        print(f"\nFINAL MEAN RATING: {final_dist.mean_rating:.4f}")

        # Analyze the issue
        print(f"\n[ANALYSIS]")
        print("-" * 40)

        # Check similarity spread
        avg_similarities = np.mean([s.scores for s in all_similarities], axis=0)
        sim_range = np.max(avg_similarities) - np.min(avg_similarities)
        sim_std = np.std(avg_similarities)

        print(f"Average similarity scores across all sets:")
        for rating in range(1, 6):
            print(f"  Rating {rating}: {avg_similarities[rating-1]:.6f}")

        print(f"\nSimilarity spread metrics:")
        print(f"  Range (max - min): {sim_range:.6f}")
        print(f"  Standard deviation: {sim_std:.6f}")
        print(f"  Max similarity: {np.max(avg_similarities):.6f} (rating {np.argmax(avg_similarities)+1})")
        print(f"  Min similarity: {np.min(avg_similarities):.6f} (rating {np.argmin(avg_similarities)+1})")

        if sim_range < 0.1:
            print("\n  ⚠️  WARNING: Very small similarity range!")
            print("     All reference statements are nearly equally similar to response.")
            print("     This causes the distribution to collapse toward uniform (all ~3.0)")

        # Check distribution entropy
        entropy = -np.sum(final_dist.probabilities * np.log(final_dist.probabilities + 1e-10))
        max_entropy = np.log(5)  # Maximum entropy for 5 categories

        print(f"\nDistribution entropy:")
        print(f"  Actual entropy: {entropy:.4f}")
        print(f"  Maximum entropy: {max_entropy:.4f}")
        print(f"  Normalized entropy: {entropy/max_entropy:.2%}")

        if entropy/max_entropy > 0.95:
            print("\n  ⚠️  WARNING: Distribution is nearly uniform!")
            print("     High entropy indicates lack of differentiation.")

if __name__ == "__main__":
    trace_ssr_processing()

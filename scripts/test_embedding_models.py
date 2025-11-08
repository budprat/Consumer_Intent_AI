#!/usr/bin/env python3
"""
ABOUTME: Comprehensive embedding model comparison for SSR rating differentiation
ABOUTME: Tests multiple embedding models to find best alternative to text-embedding-3-small

This script compares different embedding models to identify which one provides
the best separation between opposing purchase intent statements.

Tests performed:
1. Reference statement similarity matrices (should have low R1-R5 similarity)
2. Real response differentiation (positive vs negative)
3. Distribution construction and rating calculation
4. Quality metrics (separation, variance, reliability)

Models tested:
- text-embedding-3-small (baseline - current issue)
- text-embedding-3-large (OpenAI alternative)
- sentence-transformers/all-MiniLM-L6-v2 (efficient)
- sentence-transformers/all-mpnet-base-v2 (best overall)
- sentence-transformers/all-MiniLM-L12-v2 (balanced)
- sentence-transformers/paraphrase-multilingual-mpnet-base-v2 (robust)
"""

import os
import sys
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import json
from pathlib import Path
from datetime import datetime
import warnings

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.reference_statements import ReferenceStatementManager


@dataclass
class EmbeddingModelConfig:
    """Configuration for embedding model testing"""
    name: str
    provider: str  # "openai" or "sentence-transformers"
    model_id: str
    dimension: int
    requires_api_key: bool = False


@dataclass
class SeparationMetrics:
    """Metrics for measuring reference statement separation"""
    r1_r5_similarity: float  # Should be LOW (opposites)
    r1_r3_similarity: float  # Should be moderate
    r3_r5_similarity: float  # Should be moderate
    avg_opposite_similarity: float  # Average similarity between opposites (R1-R5, R2-R4)
    avg_adjacent_similarity: float  # Average similarity between adjacent ratings
    separation_score: float  # Overall quality score (0-1, higher = better separation)


@dataclass
class ResponseTestResult:
    """Result from testing a single response"""
    response_text: str
    expected_rating: int  # Ground truth (1-5)
    calculated_rating: float  # From SSR
    distribution: List[float]
    rating_error: float  # Absolute error from expected
    confidence: float  # Distribution confidence


@dataclass
class ModelTestResult:
    """Complete test results for one embedding model"""
    model_name: str
    model_config: EmbeddingModelConfig
    separation_metrics: SeparationMetrics
    response_tests: List[ResponseTestResult]
    avg_rating_error: float
    rating_spread: float  # Range of ratings produced (max - min)
    overall_score: float  # Combined quality score
    test_duration_seconds: float


class EmbeddingModelTester:
    """
    Comprehensive embedding model testing framework

    Tests multiple embedding models to identify which provides best
    purchase intent differentiation for SSR methodology.
    """

    def __init__(self, data_dir: Optional[Path] = None):
        """Initialize tester with reference statements"""
        self.data_dir = data_dir or Path(__file__).parent.parent / "data"
        self.reference_manager = ReferenceStatementManager(data_dir=self.data_dir)

        # Load reference statements
        self.reference_sets = self.reference_manager.get_paper_default_sets()[:6]

        # Test responses with ground truth ratings
        self.test_responses = [
            # Rating 1: Very negative
            ("I would absolutely never buy this. It's terrible and way too expensive.", 1),
            ("This is horrible. I can't imagine anyone wanting this product.", 1),
            ("Definitely not buying. This looks like a complete waste of money.", 1),

            # Rating 2: Negative
            ("I probably wouldn't buy it. It doesn't seem worth the price.", 2),
            ("I'm leaning towards not buying this. There are better options.", 2),
            ("It's rather unlikely I'd purchase this product.", 2),

            # Rating 3: Neutral
            ("I'm not sure if I'd buy it or not. It's okay but nothing special.", 3),
            ("I'm on the fence about this. Could go either way.", 3),
            ("I'm undecided. It has some good points and some bad points.", 3),

            # Rating 4: Positive
            ("I'd probably buy it. It looks pretty good for the price.", 4),
            ("I'm leaning towards buying this. It seems like a solid choice.", 4),
            ("I think I would buy this. It meets my needs well.", 4),

            # Rating 5: Very positive
            ("I would definitely buy this! It's exactly what I need.", 5),
            ("This is amazing! I'd absolutely purchase this right away.", 5),
            ("It's very likely I'd buy it. This is a fantastic product.", 5),
        ]

        # Define models to test
        self.models_to_test = [
            # OpenAI models (require API key)
            EmbeddingModelConfig(
                name="OpenAI text-embedding-3-small (CURRENT)",
                provider="openai",
                model_id="text-embedding-3-small",
                dimension=1536,
                requires_api_key=True,
            ),
            EmbeddingModelConfig(
                name="OpenAI text-embedding-3-large",
                provider="openai",
                model_id="text-embedding-3-large",
                dimension=3072,
                requires_api_key=True,
            ),

            # Sentence-transformers models (local, no API key needed)
            EmbeddingModelConfig(
                name="all-MiniLM-L6-v2 (Fast & Efficient)",
                provider="sentence-transformers",
                model_id="sentence-transformers/all-MiniLM-L6-v2",
                dimension=384,
                requires_api_key=False,
            ),
            EmbeddingModelConfig(
                name="all-mpnet-base-v2 (Best Overall)",
                provider="sentence-transformers",
                model_id="sentence-transformers/all-mpnet-base-v2",
                dimension=768,
                requires_api_key=False,
            ),
            EmbeddingModelConfig(
                name="all-MiniLM-L12-v2 (Balanced)",
                provider="sentence-transformers",
                model_id="sentence-transformers/all-MiniLM-L12-v2",
                dimension=384,
                requires_api_key=False,
            ),
            EmbeddingModelConfig(
                name="paraphrase-mpnet-base-v2 (Paraphrase Detection)",
                provider="sentence-transformers",
                model_id="sentence-transformers/paraphrase-mpnet-base-v2",
                dimension=768,
                requires_api_key=False,
            ),
        ]

    def get_embeddings_openai(
        self, texts: List[str], model_id: str
    ) -> np.ndarray:
        """Get embeddings using OpenAI API"""
        try:
            import openai

            # Check for API key
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable not set")

            client = openai.OpenAI(api_key=api_key)

            # Get embeddings
            response = client.embeddings.create(
                input=texts,
                model=model_id,
            )

            embeddings = np.array([item.embedding for item in response.data])
            return embeddings

        except Exception as e:
            print(f"❌ Error getting OpenAI embeddings: {e}")
            return None

    def get_embeddings_sentence_transformers(
        self, texts: List[str], model_id: str
    ) -> np.ndarray:
        """Get embeddings using sentence-transformers"""
        try:
            from sentence_transformers import SentenceTransformer

            # Load model (downloads if not cached)
            print(f"   Loading model {model_id}...")
            model = SentenceTransformer(model_id)

            # Get embeddings
            embeddings = model.encode(texts, show_progress_bar=False)
            return np.array(embeddings)

        except ImportError:
            print("❌ sentence-transformers not installed. Install with:")
            print("   pip install sentence-transformers")
            return None
        except Exception as e:
            print(f"❌ Error getting sentence-transformers embeddings: {e}")
            return None

    def get_embeddings(
        self, texts: List[str], config: EmbeddingModelConfig
    ) -> Optional[np.ndarray]:
        """Get embeddings using specified model"""
        if config.provider == "openai":
            return self.get_embeddings_openai(texts, config.model_id)
        elif config.provider == "sentence-transformers":
            return self.get_embeddings_sentence_transformers(texts, config.model_id)
        else:
            raise ValueError(f"Unknown provider: {config.provider}")

    def calculate_cosine_similarity(
        self, vec1: np.ndarray, vec2: np.ndarray
    ) -> float:
        """Calculate cosine similarity between two vectors"""
        # Normalize vectors
        vec1_norm = vec1 / (np.linalg.norm(vec1) + 1e-10)
        vec2_norm = vec2 / (np.linalg.norm(vec2) + 1e-10)

        # Cosine similarity
        similarity = np.dot(vec1_norm, vec2_norm)
        return float(similarity)

    def calculate_separation_metrics(
        self, reference_embeddings: np.ndarray
    ) -> SeparationMetrics:
        """
        Calculate separation metrics for reference statements

        Args:
            reference_embeddings: Array of shape (5, embedding_dim) for ratings 1-5

        Returns:
            SeparationMetrics with quality scores
        """
        # Calculate all pairwise similarities
        similarities = np.zeros((5, 5))
        for i in range(5):
            for j in range(5):
                similarities[i, j] = self.calculate_cosine_similarity(
                    reference_embeddings[i], reference_embeddings[j]
                )

        # Key similarities
        r1_r5 = similarities[0, 4]  # Opposites (should be LOW)
        r1_r3 = similarities[0, 2]  # Negative to neutral
        r3_r5 = similarities[2, 4]  # Neutral to positive

        # Average opposite similarity (should be LOW)
        opposite_pairs = [
            (0, 4),  # R1-R5
            (1, 3),  # R2-R4
        ]
        avg_opposite = np.mean([similarities[i, j] for i, j in opposite_pairs])

        # Average adjacent similarity (should be MODERATE-HIGH)
        adjacent_pairs = [
            (0, 1),  # R1-R2
            (1, 2),  # R2-R3
            (2, 3),  # R3-R4
            (3, 4),  # R4-R5
        ]
        avg_adjacent = np.mean([similarities[i, j] for i, j in adjacent_pairs])

        # Separation score: higher is better
        # Perfect separation: opposites = 0, adjacent = 1
        # Score = (1 - avg_opposite) * avg_adjacent
        separation_score = (1.0 - avg_opposite) * avg_adjacent

        return SeparationMetrics(
            r1_r5_similarity=r1_r5,
            r1_r3_similarity=r1_r3,
            r3_r5_similarity=r3_r5,
            avg_opposite_similarity=avg_opposite,
            avg_adjacent_similarity=avg_adjacent,
            separation_score=separation_score,
        )

    def construct_distribution_from_similarities(
        self, similarities: np.ndarray, temperature: float = 1.5
    ) -> Tuple[np.ndarray, float]:
        """
        Construct probability distribution from similarity scores

        Uses SSR methodology from paper.

        Args:
            similarities: Array of 5 similarity scores (ratings 1-5)
            temperature: Temperature parameter (default: 1.5 from paper)

        Returns:
            Tuple of (probabilities, mean_rating)
        """
        # Center at neutral (rating 3, index 2)
        neutral_idx = 2
        raw_scores = similarities - similarities[neutral_idx]

        # Temperature scaling
        scaled_scores = raw_scores / temperature

        # Softmax normalization
        scaled_scores_stable = scaled_scores - np.max(scaled_scores)
        exp_scores = np.exp(scaled_scores_stable)
        probabilities = exp_scores / (np.sum(exp_scores) + 1e-10)

        # Mean rating
        ratings = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        mean_rating = np.dot(probabilities, ratings)

        return probabilities, mean_rating

    def calculate_confidence(self, distribution: np.ndarray) -> float:
        """Calculate confidence from distribution (inverse of normalized entropy)"""
        epsilon = 1e-10
        safe_dist = distribution + epsilon
        safe_dist = safe_dist / safe_dist.sum()

        entropy = -np.sum(safe_dist * np.log(safe_dist))
        max_entropy = np.log(5)
        confidence = 1.0 - (entropy / max_entropy)

        return float(confidence)

    def test_response(
        self,
        response_text: str,
        expected_rating: int,
        response_embedding: np.ndarray,
        reference_embeddings: np.ndarray,
    ) -> ResponseTestResult:
        """
        Test a single response against reference statements

        Args:
            response_text: Consumer response text
            expected_rating: Ground truth rating (1-5)
            response_embedding: Embedding of response
            reference_embeddings: Embeddings of 5 reference statements

        Returns:
            ResponseTestResult with calculated rating and metrics
        """
        # Calculate similarities to all 5 reference statements
        similarities = np.array([
            self.calculate_cosine_similarity(response_embedding, ref_emb)
            for ref_emb in reference_embeddings
        ])

        # Construct distribution
        distribution, mean_rating = self.construct_distribution_from_similarities(
            similarities, temperature=1.5
        )

        # Calculate metrics
        rating_error = abs(mean_rating - expected_rating)
        confidence = self.calculate_confidence(distribution)

        return ResponseTestResult(
            response_text=response_text,
            expected_rating=expected_rating,
            calculated_rating=mean_rating,
            distribution=list(distribution),
            rating_error=rating_error,
            confidence=confidence,
        )

    def test_model(self, config: EmbeddingModelConfig) -> Optional[ModelTestResult]:
        """
        Test a single embedding model

        Args:
            config: Model configuration

        Returns:
            ModelTestResult or None if test failed
        """
        print(f"\n{'='*70}")
        print(f"Testing: {config.name}")
        print(f"Provider: {config.provider} | Model: {config.model_id}")
        print(f"Dimension: {config.dimension}")
        print(f"{'='*70}")

        start_time = datetime.now()

        try:
            # Step 1: Get reference statement embeddings (use first reference set)
            ref_set = self.reference_sets[0]
            ref_statements = [
                ref_set.statements[1],  # Rating 1
                ref_set.statements[2],  # Rating 2
                ref_set.statements[3],  # Rating 3
                ref_set.statements[4],  # Rating 4
                ref_set.statements[5],  # Rating 5
            ]

            print(f"\n1. Getting reference statement embeddings...")
            ref_embeddings = self.get_embeddings(ref_statements, config)

            if ref_embeddings is None:
                print("❌ Failed to get reference embeddings")
                return None

            print(f"   ✓ Got {len(ref_embeddings)} embeddings")

            # Step 2: Calculate separation metrics
            print(f"\n2. Calculating separation metrics...")
            separation_metrics = self.calculate_separation_metrics(ref_embeddings)

            print(f"   R1-R5 similarity (opposites): {separation_metrics.r1_r5_similarity:.3f}")
            print(f"   Avg opposite similarity: {separation_metrics.avg_opposite_similarity:.3f}")
            print(f"   Avg adjacent similarity: {separation_metrics.avg_adjacent_similarity:.3f}")
            print(f"   Separation score: {separation_metrics.separation_score:.3f}")

            # Step 3: Test with real responses
            print(f"\n3. Testing with {len(self.test_responses)} real responses...")

            response_texts = [text for text, _ in self.test_responses]
            response_embeddings = self.get_embeddings(response_texts, config)

            if response_embeddings is None:
                print("❌ Failed to get response embeddings")
                return None

            response_tests = []
            for i, ((text, expected_rating), embedding) in enumerate(
                zip(self.test_responses, response_embeddings)
            ):
                result = self.test_response(
                    text, expected_rating, embedding, ref_embeddings
                )
                response_tests.append(result)

                # Print sample results
                if i % 5 == 0:  # Print every 5th result
                    print(f"   Expected: {expected_rating} → Calculated: {result.calculated_rating:.2f} "
                          f"(error: {result.rating_error:.2f})")

            # Step 4: Calculate aggregate metrics
            avg_error = np.mean([r.rating_error for r in response_tests])
            all_ratings = [r.calculated_rating for r in response_tests]
            rating_spread = max(all_ratings) - min(all_ratings)

            # Overall score: combination of separation and accuracy
            # Perfect: low error, high spread, high separation
            accuracy_score = 1.0 - (avg_error / 4.0)  # Normalize by max possible error
            spread_score = rating_spread / 4.0  # Normalize by max possible spread
            overall_score = (
                0.4 * separation_metrics.separation_score +
                0.3 * accuracy_score +
                0.3 * spread_score
            )

            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            print(f"\n4. Summary:")
            print(f"   Average rating error: {avg_error:.3f}")
            print(f"   Rating spread: {rating_spread:.3f}")
            print(f"   Overall score: {overall_score:.3f}")
            print(f"   Test duration: {duration:.1f}s")

            return ModelTestResult(
                model_name=config.name,
                model_config=config,
                separation_metrics=separation_metrics,
                response_tests=response_tests,
                avg_rating_error=avg_error,
                rating_spread=rating_spread,
                overall_score=overall_score,
                test_duration_seconds=duration,
            )

        except Exception as e:
            print(f"\n❌ Error testing model: {e}")
            import traceback
            traceback.print_exc()
            return None

    def run_all_tests(self) -> List[ModelTestResult]:
        """Run tests on all configured models"""
        print("\n" + "="*70)
        print("EMBEDDING MODEL COMPARISON TEST")
        print("="*70)
        print(f"Testing {len(self.models_to_test)} embedding models")
        print(f"Reference statements: {len(self.reference_sets)} sets")
        print(f"Test responses: {len(self.test_responses)} responses")
        print("="*70)

        results = []

        for config in self.models_to_test:
            result = self.test_model(config)
            if result is not None:
                results.append(result)

        return results

    def generate_comparison_report(
        self, results: List[ModelTestResult]
    ) -> str:
        """Generate comprehensive comparison report"""
        report = []

        report.append("\n" + "="*70)
        report.append("EMBEDDING MODEL COMPARISON REPORT")
        report.append("="*70)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Models tested: {len(results)}")
        report.append("")

        # Sort by overall score (descending)
        sorted_results = sorted(results, key=lambda r: r.overall_score, reverse=True)

        # Ranking table
        report.append("\n" + "-"*70)
        report.append("OVERALL RANKING (by overall score)")
        report.append("-"*70)
        report.append(f"{'Rank':<6} {'Model':<40} {'Score':<8} {'Status'}")
        report.append("-"*70)

        for i, result in enumerate(sorted_results, 1):
            status = "🏆 WINNER" if i == 1 else ("✓ Good" if result.overall_score > 0.5 else "⚠ Poor")
            report.append(
                f"{i:<6} {result.model_name:<40} {result.overall_score:.3f}    {status}"
            )

        # Detailed metrics comparison
        report.append("\n" + "-"*70)
        report.append("SEPARATION METRICS (Lower opposite similarity = Better)")
        report.append("-"*70)
        report.append(f"{'Model':<40} {'R1-R5':<10} {'Avg Opp':<10} {'Sep Score':<10}")
        report.append("-"*70)

        for result in sorted_results:
            sep = result.separation_metrics
            report.append(
                f"{result.model_name:<40} "
                f"{sep.r1_r5_similarity:.3f}      "
                f"{sep.avg_opposite_similarity:.3f}      "
                f"{sep.separation_score:.3f}"
            )

        # Rating accuracy comparison
        report.append("\n" + "-"*70)
        report.append("RATING ACCURACY (Lower error & Higher spread = Better)")
        report.append("-"*70)
        report.append(f"{'Model':<40} {'Avg Error':<12} {'Spread':<10}")
        report.append("-"*70)

        for result in sorted_results:
            report.append(
                f"{result.model_name:<40} "
                f"{result.avg_rating_error:.3f}        "
                f"{result.rating_spread:.3f}"
            )

        # Best model detailed analysis
        if sorted_results:
            best = sorted_results[0]
            worst = sorted_results[-1] if len(sorted_results) > 1 else None

            report.append("\n" + "="*70)
            report.append("WINNER: " + best.model_name)
            report.append("="*70)

            report.append(f"\nOverall Score: {best.overall_score:.3f}")
            report.append(f"\nSeparation Metrics:")
            report.append(f"  • R1-R5 similarity (opposites): {best.separation_metrics.r1_r5_similarity:.3f}")
            report.append(f"  • Average opposite similarity: {best.separation_metrics.avg_opposite_similarity:.3f}")
            report.append(f"  • Separation score: {best.separation_metrics.separation_score:.3f}")

            report.append(f"\nRating Performance:")
            report.append(f"  • Average rating error: {best.avg_rating_error:.3f}")
            report.append(f"  • Rating spread: {best.rating_spread:.3f}")

            report.append(f"\nConfiguration:")
            report.append(f"  • Provider: {best.model_config.provider}")
            report.append(f"  • Model ID: {best.model_config.model_id}")
            report.append(f"  • Dimension: {best.model_config.dimension}")
            report.append(f"  • Requires API key: {best.model_config.requires_api_key}")

            # Compare to worst/baseline
            if worst and worst != best:
                report.append(f"\n" + "-"*70)
                report.append(f"IMPROVEMENT OVER BASELINE ({worst.model_name}):")
                report.append("-"*70)

                sep_improvement = (
                    (worst.separation_metrics.avg_opposite_similarity -
                     best.separation_metrics.avg_opposite_similarity) /
                    worst.separation_metrics.avg_opposite_similarity * 100
                )

                spread_improvement = (
                    (best.rating_spread - worst.rating_spread) /
                    (worst.rating_spread + 0.01) * 100
                )

                report.append(f"  • Opposite similarity reduced by: {sep_improvement:.1f}%")
                report.append(f"  • Rating spread increased by: {spread_improvement:.1f}%")

        # Recommendations
        report.append("\n" + "="*70)
        report.append("RECOMMENDATIONS")
        report.append("="*70)

        if sorted_results:
            best = sorted_results[0]

            # Check if improvement is significant
            baseline = next((r for r in results if "CURRENT" in r.model_name), None)

            if baseline and best != baseline:
                report.append(f"\n✅ RECOMMENDED: Switch to {best.model_name}")
                report.append(f"\nReasons:")
                report.append(f"  1. Better separation of opposite intents")
                report.append(f"  2. Wider rating spread ({best.rating_spread:.2f} vs {baseline.rating_spread:.2f})")
                report.append(f"  3. Lower average error ({best.avg_rating_error:.2f} vs {baseline.avg_rating_error:.2f})")

                report.append(f"\nImplementation:")
                report.append(f"  • Update src/core/embedding.py to support {best.model_config.provider}")
                report.append(f"  • Set embedding_model='{best.model_config.model_id}'")
                if not best.model_config.requires_api_key:
                    report.append(f"  • No API key required (runs locally)")
                    report.append(f"  • Install: pip install sentence-transformers")
            else:
                report.append(f"\n⚠ WARNING: No significant improvement found")
                report.append(f"All models show similar limitations with rating differentiation.")
                report.append(f"Consider:")
                report.append(f"  1. Fine-tuning a custom model on purchase intent data")
                report.append(f"  2. Using supervised projection layer")
                report.append(f"  3. Implementing learned similarity function")

        report.append("\n" + "="*70)

        return "\n".join(report)

    def save_results(
        self, results: List[ModelTestResult], output_dir: Path
    ):
        """Save detailed results to JSON file"""
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"embedding_model_comparison_{timestamp}.json"

        # Convert to serializable format
        data = {
            "test_date": datetime.now().isoformat(),
            "num_models_tested": len(results),
            "num_test_responses": len(self.test_responses),
            "models": [],
        }

        for result in results:
            model_data = {
                "name": result.model_name,
                "config": {
                    "provider": result.model_config.provider,
                    "model_id": result.model_config.model_id,
                    "dimension": result.model_config.dimension,
                    "requires_api_key": result.model_config.requires_api_key,
                },
                "separation_metrics": {
                    "r1_r5_similarity": result.separation_metrics.r1_r5_similarity,
                    "r1_r3_similarity": result.separation_metrics.r1_r3_similarity,
                    "r3_r5_similarity": result.separation_metrics.r3_r5_similarity,
                    "avg_opposite_similarity": result.separation_metrics.avg_opposite_similarity,
                    "avg_adjacent_similarity": result.separation_metrics.avg_adjacent_similarity,
                    "separation_score": result.separation_metrics.separation_score,
                },
                "performance": {
                    "avg_rating_error": result.avg_rating_error,
                    "rating_spread": result.rating_spread,
                    "overall_score": result.overall_score,
                    "test_duration_seconds": result.test_duration_seconds,
                },
                "response_tests": [
                    {
                        "response": r.response_text,
                        "expected_rating": r.expected_rating,
                        "calculated_rating": r.calculated_rating,
                        "distribution": r.distribution,
                        "rating_error": r.rating_error,
                        "confidence": r.confidence,
                    }
                    for r in result.response_tests
                ],
            }
            data["models"].append(model_data)

        with open(output_file, "w") as f:
            json.dump(data, f, indent=2)

        print(f"\n✓ Detailed results saved to: {output_file}")


def main():
    """Main execution function"""
    print("\n" + "="*70)
    print("EMBEDDING MODEL TESTING FOR SSR RATING DIFFERENTIATION")
    print("="*70)

    # Check dependencies
    print("\nChecking dependencies...")

    # Check OpenAI
    try:
        import openai
        has_openai = True
        print("  ✓ openai package installed")
    except ImportError:
        has_openai = False
        print("  ⚠ openai package not installed (OpenAI models will be skipped)")

    # Check sentence-transformers
    try:
        import sentence_transformers
        has_st = True
        print("  ✓ sentence-transformers package installed")
    except ImportError:
        has_st = False
        print("  ⚠ sentence-transformers not installed")
        print("    Install with: pip install sentence-transformers")

    # Check OpenAI API key
    has_api_key = bool(os.getenv("OPENAI_API_KEY"))
    if has_api_key:
        print("  ✓ OPENAI_API_KEY environment variable set")
    else:
        print("  ⚠ OPENAI_API_KEY not set (OpenAI models will be skipped)")

    if not (has_openai and has_api_key) and not has_st:
        print("\n❌ ERROR: No embedding models available to test!")
        print("Please install at least one of:")
        print("  1. pip install openai && set OPENAI_API_KEY")
        print("  2. pip install sentence-transformers")
        return

    # Initialize tester
    print("\nInitializing tester...")
    tester = EmbeddingModelTester()

    # Run tests
    results = tester.run_all_tests()

    if not results:
        print("\n❌ No models were successfully tested!")
        return

    # Generate report
    print("\n" + "="*70)
    print("Generating comparison report...")
    report = tester.generate_comparison_report(results)
    print(report)

    # Save results
    output_dir = Path(__file__).parent.parent / "test_results"
    tester.save_results(results, output_dir)

    print("\n✓ Testing complete!")
    print("\nNext steps:")
    print("  1. Review the comparison report above")
    print("  2. Check detailed results in test_results/ directory")
    print("  3. Implement the recommended embedding model")
    print("  4. Re-run SSR tests to verify improvement")


if __name__ == "__main__":
    main()

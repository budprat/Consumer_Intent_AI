"""
ABOUTME: Distribution construction from similarity scores using SSR methodology
ABOUTME: Implements temperature scaling, softmax normalization, and multi-set averaging

This module constructs probability distributions over Likert ratings (1-5) from
similarity scores using the Semantic Similarity Rating (SSR) formula from the paper.

Mathematical Formula:
    p_c,i(r) ∝ γ(σ_r,i, t_c) - γ(σ_ℓ,i, t_c) + ε·δ_ℓ,r

Where:
    - p_c,i(r) = probability of rating r for concept c, reference set i
    - γ(σ_r, t_c) = cosine_similarity(embedding(σ_r), embedding(t_c))
    - σ_r = reference statement for rating r ∈ {1,2,3,4,5}
    - t_c = synthetic consumer response text for concept c
    - ℓ = 3 (neutral reference point)
    - ε = offset parameter (default 0)
    - δ_ℓ,r = Kronecker delta (1 if ℓ=r, else 0)

Temperature Scaling:
    scaled_score(r) = [γ(σ_r, t_c) - γ(σ_3, t_c)] / T
    probability(r) = softmax(scaled_score(r))
"""

import numpy as np
from typing import List, Dict
from dataclasses import dataclass
from .similarity import SimilarityResult


@dataclass
class DistributionResult:
    """
    Result of distribution construction

    Attributes:
        probabilities: Probability distribution over ratings 1-5 (sums to 1.0)
        mean_rating: Expected rating (weighted average)
        similarity_scores: Original similarity scores used
        temperature: Temperature parameter used
        offset: Offset parameter used
    """

    probabilities: np.ndarray  # Shape: (5,) for ratings 1-5
    mean_rating: float
    similarity_scores: np.ndarray
    temperature: float
    offset: float

    def get_rating_probability(self, rating: int) -> float:
        """Get probability for specific rating (1-5)"""
        if not 1 <= rating <= 5:
            raise ValueError(f"Rating must be 1-5, got {rating}")
        return float(self.probabilities[rating - 1])


class DistributionConstructor:
    """
    Constructs probability distributions from similarity scores using SSR methodology

    Features:
    - Temperature scaling for distribution spread control
    - Softmax normalization ensures valid probability distribution
    - Offset parameter for distribution centering
    - Multi-reference set averaging for robustness
    - Numerical stability handling
    """

    def __init__(
        self,
        temperature: float = 1.0,
        offset: float = 0.0,
        neutral_rating: int = 3,
    ):
        """
        Initialize distribution constructor

        Args:
            temperature: Controls distribution spread (T)
                - Lower T: Sharper distribution (more confident)
                - Higher T: Smoother distribution (more uncertain)
                - Paper tested: 0.5, 1.0, 1.5
                - Default: 1.0 (per paper's best results)
            offset: Offset parameter (ε) for distribution centering
                - Default: 0 (no offset)
            neutral_rating: Neutral reference point (ℓ)
                - Default: 3 (middle Likert rating)
        """
        if temperature <= 0:
            raise ValueError(f"Temperature must be positive, got {temperature}")
        if not 1 <= neutral_rating <= 5:
            raise ValueError(f"Neutral rating must be 1-5, got {neutral_rating}")

        self.temperature = temperature
        self.offset = offset
        self.neutral_rating = neutral_rating
        self._eps = 1e-10  # Small epsilon for numerical stability

    def construct_distribution(
        self, similarity_result: SimilarityResult
    ) -> DistributionResult:
        """
        Construct probability distribution from similarity scores

        Args:
            similarity_result: Similarity scores for ratings 1-5

        Returns:
            DistributionResult containing probability distribution

        Algorithm:
            1. Raw scores: score[r] = similarity[r] - similarity[neutral]
            2. Add offset: score[r] += offset * δ_neutral,r
            3. Temperature scale: scaled[r] = score[r] / T
            4. Softmax normalize: prob[r] = exp(scaled[r]) / sum(exp(scaled))
        """
        scores = similarity_result.scores

        if len(scores) != 5:
            raise ValueError(f"Expected 5 scores (ratings 1-5), got {len(scores)}")

        # Step 1: Calculate raw scores by differencing from neutral
        neutral_idx = self.neutral_rating - 1
        neutral_similarity = scores[neutral_idx]
        raw_scores = scores - neutral_similarity

        # Step 2: Add offset for neutral rating
        if self.offset != 0:
            raw_scores[neutral_idx] += self.offset

        # Step 3: Temperature scaling
        scaled_scores = raw_scores / self.temperature

        # Step 4: Softmax normalization
        probabilities = self._softmax(scaled_scores)

        # Calculate mean rating
        ratings = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        mean_rating = np.dot(probabilities, ratings)

        return DistributionResult(
            probabilities=probabilities,
            mean_rating=float(mean_rating),
            similarity_scores=scores,
            temperature=self.temperature,
            offset=self.offset,
        )

    def _softmax(self, scores: np.ndarray) -> np.ndarray:
        """
        Softmax normalization with numerical stability

        Formula:
            softmax(x_i) = exp(x_i) / sum(exp(x_j))

        Numerical stability trick:
            softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_j - max(x)))

        Args:
            scores: Input scores

        Returns:
            Normalized probabilities (sum to 1.0)
        """
        # Subtract max for numerical stability
        scores_stable = scores - np.max(scores)

        # Compute exponentials
        exp_scores = np.exp(scores_stable)

        # Normalize
        probabilities = exp_scores / (np.sum(exp_scores) + self._eps)

        return probabilities

    def average_across_reference_sets(
        self, distributions: List[DistributionResult]
    ) -> DistributionResult:
        """
        Average distributions from multiple reference sets

        Args:
            distributions: List of distributions from different reference sets

        Returns:
            Averaged distribution

        Formula:
            P_final(r) = (1/N) Σ_i P_i(r)

        Where:
            - N = number of reference sets
            - P_i(r) = probability of rating r from reference set i
        """
        if not distributions:
            raise ValueError("No distributions provided for averaging")

        # Extract probability arrays
        prob_arrays = [d.probabilities for d in distributions]

        # Average across reference sets
        averaged_probs = np.mean(prob_arrays, axis=0)

        # Ensure normalization (should already be normalized, but verify)
        averaged_probs = averaged_probs / (np.sum(averaged_probs) + self._eps)

        # Calculate mean rating from averaged distribution
        ratings = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        mean_rating = np.dot(averaged_probs, ratings)

        # Use first distribution's metadata (they should all have same parameters)
        first_dist = distributions[0]

        return DistributionResult(
            probabilities=averaged_probs,
            mean_rating=float(mean_rating),
            similarity_scores=first_dist.similarity_scores,
            temperature=first_dist.temperature,
            offset=first_dist.offset,
        )

    def weighted_average_across_reference_sets(
        self, distributions: List[DistributionResult], weights: List[float]
    ) -> DistributionResult:
        """
        Weighted average of distributions from multiple reference sets

        Args:
            distributions: List of distributions from different reference sets
            weights: Weights for each reference set (must sum to 1.0)

        Returns:
            Weighted averaged distribution

        Formula:
            P_final(r) = Σ_i w_i * P_i(r)

        Where:
            - w_i = weight for reference set i
            - P_i(r) = probability of rating r from reference set i
        """
        if not distributions:
            raise ValueError("No distributions provided for averaging")

        if len(distributions) != len(weights):
            raise ValueError(
                f"Number of distributions ({len(distributions)}) "
                f"must match number of weights ({len(weights)})"
            )

        # Validate weights
        weight_sum = sum(weights)
        if not np.isclose(weight_sum, 1.0):
            raise ValueError(f"Weights must sum to 1.0, got {weight_sum}")

        # Extract probability arrays
        prob_arrays = [d.probabilities for d in distributions]

        # Weighted average
        weighted_probs = np.zeros(5)
        for prob_array, weight in zip(prob_arrays, weights):
            weighted_probs += weight * prob_array

        # Ensure normalization
        weighted_probs = weighted_probs / (np.sum(weighted_probs) + self._eps)

        # Calculate mean rating
        ratings = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        mean_rating = np.dot(weighted_probs, ratings)

        first_dist = distributions[0]

        return DistributionResult(
            probabilities=weighted_probs,
            mean_rating=float(mean_rating),
            similarity_scores=first_dist.similarity_scores,
            temperature=first_dist.temperature,
            offset=first_dist.offset,
        )

    def validate_distribution(self, distribution: DistributionResult) -> bool:
        """
        Validate that distribution is valid probability distribution

        Checks:
        1. All probabilities in [0, 1]
        2. Probabilities sum to 1.0 (within tolerance)
        3. Mean rating in [1, 5]

        Args:
            distribution: Distribution to validate

        Returns:
            True if valid, False otherwise
        """
        probs = distribution.probabilities

        # Check range
        if not np.all((probs >= 0) & (probs <= 1)):
            return False

        # Check sum
        if not np.isclose(np.sum(probs), 1.0, atol=1e-6):
            return False

        # Check mean rating
        if not (1.0 <= distribution.mean_rating <= 5.0):
            return False

        return True

    def get_distribution_statistics(
        self, distribution: DistributionResult
    ) -> Dict[str, float]:
        """
        Calculate statistics for distribution

        Returns:
            Dictionary containing:
            - mean: Expected rating
            - variance: Distribution variance
            - std: Standard deviation
            - mode: Most probable rating
            - entropy: Shannon entropy (measure of uncertainty)
        """
        probs = distribution.probabilities
        ratings = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        mean = distribution.mean_rating
        variance = np.dot(probs, (ratings - mean) ** 2)
        std = np.sqrt(variance)
        mode = int(np.argmax(probs) + 1)  # Rating with highest probability

        # Shannon entropy: -Σ p(x) log(p(x))
        # Higher entropy = more uncertain distribution
        entropy = -np.sum(probs * np.log(probs + self._eps))

        return {
            "mean": mean,
            "variance": float(variance),
            "std": float(std),
            "mode": mode,
            "entropy": float(entropy),
        }


# Example usage and testing
if __name__ == "__main__":
    # Test distribution constructor
    constructor = DistributionConstructor(temperature=1.0)

    # Create test similarity scores
    # Simulating a positive response (higher similarity to rating 5)
    test_scores = np.array([0.3, 0.4, 0.5, 0.7, 0.85])  # Ratings 1-5

    # Create SimilarityResult
    sim_result = SimilarityResult(
        scores=test_scores,
        response_embedding=np.random.randn(1536),
        reference_embeddings=np.random.randn(5, 1536),
    )

    # Construct distribution
    dist = constructor.construct_distribution(sim_result)

    print("Distribution for positive response:")
    print("Probabilities:")
    for i, prob in enumerate(dist.probabilities, 1):
        print(f"  Rating {i}: {prob:.4f} ({prob * 100:.1f}%)")

    print(f"\nMean rating: {dist.mean_rating:.2f}")
    print(f"Validation: {constructor.validate_distribution(dist)}")

    # Get statistics
    stats = constructor.get_distribution_statistics(dist)
    print("\nDistribution statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value:.4f}")

    # Test multi-reference set averaging
    print("\n" + "=" * 50)
    print("Testing multi-reference set averaging:")

    # Create multiple distributions (simulating different reference sets)
    distributions = []
    for _ in range(6):  # Paper used 6 reference sets
        # Add small variations to scores
        varied_scores = test_scores + np.random.randn(5) * 0.05
        sim_result = SimilarityResult(
            scores=varied_scores,
            response_embedding=np.random.randn(1536),
            reference_embeddings=np.random.randn(5, 1536),
        )
        dist = constructor.construct_distribution(sim_result)
        distributions.append(dist)

    # Average distributions
    averaged_dist = constructor.average_across_reference_sets(distributions)

    print("Averaged distribution across 6 reference sets:")
    for i, prob in enumerate(averaged_dist.probabilities, 1):
        print(f"  Rating {i}: {prob:.4f} ({prob * 100:.1f}%)")

    print(f"\nMean rating: {averaged_dist.mean_rating:.2f}")

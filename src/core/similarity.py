"""
ABOUTME: Semantic similarity calculation using cosine similarity for embeddings
ABOUTME: Provides vectorized operations for efficient batch processing

This module implements the core similarity computation for SSR methodology,
calculating cosine similarity between response embeddings and reference statement embeddings.

Mathematical formula:
    cosine_similarity(a, b) = (a · b) / (||a|| × ||b||)

Where:
    - a, b are embedding vectors (1536 dimensions for text-embedding-3-small)
    - · is dot product
    - ||·|| is L2 norm
"""

import numpy as np
from typing import List, Union
from dataclasses import dataclass


@dataclass
class SimilarityResult:
    """
    Result of similarity calculation

    Attributes:
        scores: Similarity scores for each reference statement (ratings 1-5)
        response_embedding: The response embedding vector used
        reference_embeddings: The reference embeddings used for comparison
    """

    scores: np.ndarray  # Shape: (5,) for 5 Likert ratings
    response_embedding: np.ndarray
    reference_embeddings: np.ndarray  # Shape: (5, embedding_dim)


class SimilarityCalculator:
    """
    Calculates cosine similarity between embeddings with optimization for batch processing

    Features:
    - Vectorized operations using numpy
    - Pre-normalized embeddings for efficiency
    - Batch processing support
    - Numerical stability handling
    """

    def __init__(self, embedding_dim: int = 1536):
        """
        Initialize similarity calculator

        Args:
            embedding_dim: Dimension of embedding vectors (default: 1536 for text-embedding-3-small)
        """
        self.embedding_dim = embedding_dim
        self._eps = 1e-8  # Small epsilon for numerical stability

    def normalize_embedding(self, embedding: np.ndarray) -> np.ndarray:
        """
        Normalize embedding vector to unit length

        Args:
            embedding: Embedding vector(s) to normalize

        Returns:
            Normalized embedding vector(s)

        Mathematical operation:
            normalized = embedding / ||embedding||
        """
        if embedding.ndim == 1:
            # Single vector
            norm = np.linalg.norm(embedding) + self._eps
            return embedding / norm
        else:
            # Multiple vectors
            norms = np.linalg.norm(embedding, axis=1, keepdims=True) + self._eps
            return embedding / norms

    def cosine_similarity(
        self,
        embedding_a: np.ndarray,
        embedding_b: np.ndarray,
        pre_normalized: bool = False,
    ) -> Union[float, np.ndarray]:
        """
        Calculate cosine similarity between two embeddings or sets of embeddings

        Args:
            embedding_a: First embedding vector or matrix
            embedding_b: Second embedding vector or matrix
            pre_normalized: Whether embeddings are already normalized

        Returns:
            Cosine similarity score(s) in range [-1, 1]

        Formula:
            similarity = (a · b) / (||a|| × ||b||)

        If pre_normalized=True:
            similarity = a · b  (since ||a|| = ||b|| = 1)
        """
        if not pre_normalized:
            embedding_a = self.normalize_embedding(embedding_a)
            embedding_b = self.normalize_embedding(embedding_b)

        # Compute dot product
        if embedding_a.ndim == 1 and embedding_b.ndim == 1:
            # Single vector to single vector
            return np.dot(embedding_a, embedding_b)
        elif embedding_a.ndim == 1 and embedding_b.ndim == 2:
            # Single vector to multiple vectors
            return np.dot(embedding_b, embedding_a)
        elif embedding_a.ndim == 2 and embedding_b.ndim == 1:
            # Multiple vectors to single vector
            return np.dot(embedding_a, embedding_b)
        else:
            # Multiple vectors to multiple vectors
            return np.dot(embedding_a, embedding_b.T)

    def calculate_similarities(
        self,
        response_embedding: np.ndarray,
        reference_embeddings: np.ndarray,
        pre_normalized: bool = False,
    ) -> SimilarityResult:
        """
        Calculate similarities between response and all reference statements

        Args:
            response_embedding: Embedding of synthetic consumer response (1536,)
            reference_embeddings: Embeddings of 5 reference statements (5, 1536)
            pre_normalized: Whether embeddings are already normalized

        Returns:
            SimilarityResult containing similarity scores for each rating

        Note:
            Reference embeddings should be ordered by rating (1, 2, 3, 4, 5)
        """
        if reference_embeddings.shape[0] != 5:
            raise ValueError(
                f"Expected 5 reference embeddings (for ratings 1-5), "
                f"got {reference_embeddings.shape[0]}"
            )

        if response_embedding.shape[0] != self.embedding_dim:
            raise ValueError(
                f"Response embedding has wrong dimension: "
                f"expected {self.embedding_dim}, got {response_embedding.shape[0]}"
            )

        if reference_embeddings.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Reference embeddings have wrong dimension: "
                f"expected {self.embedding_dim}, got {reference_embeddings.shape[1]}"
            )

        # Calculate similarities
        scores = self.cosine_similarity(
            response_embedding, reference_embeddings, pre_normalized=pre_normalized
        )

        return SimilarityResult(
            scores=scores,
            response_embedding=response_embedding,
            reference_embeddings=reference_embeddings,
        )

    def batch_calculate_similarities(
        self,
        response_embeddings: np.ndarray,
        reference_embeddings: np.ndarray,
        pre_normalized: bool = False,
    ) -> List[SimilarityResult]:
        """
        Calculate similarities for multiple responses in batch

        Args:
            response_embeddings: Multiple response embeddings (N, 1536)
            reference_embeddings: Embeddings of 5 reference statements (5, 1536)
            pre_normalized: Whether embeddings are already normalized

        Returns:
            List of SimilarityResult for each response

        Optimization:
            Uses vectorized operations for efficient batch processing
        """
        if reference_embeddings.shape[0] != 5:
            raise ValueError(
                f"Expected 5 reference embeddings (for ratings 1-5), "
                f"got {reference_embeddings.shape[0]}"
            )

        # Normalize if needed
        if not pre_normalized:
            response_embeddings = self.normalize_embedding(response_embeddings)
            reference_embeddings = self.normalize_embedding(reference_embeddings)

        # Calculate all similarities at once (N, 5)
        scores_matrix = np.dot(response_embeddings, reference_embeddings.T)

        # Create SimilarityResult for each response
        results = []
        for i, scores in enumerate(scores_matrix):
            results.append(
                SimilarityResult(
                    scores=scores,
                    response_embedding=response_embeddings[i],
                    reference_embeddings=reference_embeddings,
                )
            )

        return results

    def validate_similarity_scores(self, scores: np.ndarray) -> bool:
        """
        Validate that similarity scores are in valid range [-1, 1]

        Args:
            scores: Similarity scores to validate

        Returns:
            True if all scores are valid, False otherwise
        """
        return np.all((scores >= -1.0 - self._eps) & (scores <= 1.0 + self._eps))


# Example usage and testing
if __name__ == "__main__":
    # Test similarity calculator
    calc = SimilarityCalculator(embedding_dim=1536)

    # Create random embeddings for testing
    response_emb = np.random.randn(1536)
    reference_embs = np.random.randn(5, 1536)

    # Calculate similarities
    result = calc.calculate_similarities(response_emb, reference_embs)

    print("Similarity scores for ratings 1-5:")
    for i, score in enumerate(result.scores, 1):
        print(f"  Rating {i}: {score:.4f}")

    print(f"\nValidation: {calc.validate_similarity_scores(result.scores)}")

    # Test batch processing
    batch_response_embs = np.random.randn(10, 1536)
    batch_results = calc.batch_calculate_similarities(
        batch_response_embs, reference_embs
    )

    print(f"\nBatch processing: Processed {len(batch_results)} responses")

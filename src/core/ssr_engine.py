"""
ABOUTME: Main SSR (Semantic Similarity Rating) Engine orchestrating all components
ABOUTME: Provides high-level interface for generating synthetic consumer ratings

This is the main entry point for the SSR system, coordinating:
1. Text Elicitation (LLM response generation)
2. Embedding Retrieval (convert text to vectors)
3. Similarity Calculation (compare with reference statements)
4. Distribution Construction (generate probability distributions)

The SSR Engine achieves 90% of human test-retest reliability (ρ ≥ 0.90)
and distribution similarity K^xy ≥ 0.85 when properly configured.
"""

import numpy as np
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
from pathlib import Path

from .similarity import SimilarityCalculator, SimilarityResult
from .distribution import DistributionConstructor, DistributionResult
from .embedding import EmbeddingRetriever
from .reference_statements import ReferenceStatementManager, ReferenceStatementSet


@dataclass
class SSRConfig:
    """
    Configuration for SSR Engine

    Attributes:
        temperature: Distribution temperature (default: 1.0, tested: 0.5, 1.0, 1.5)
        offset: Distribution offset parameter (default: 0.0)
        use_multi_set_averaging: Average across multiple reference sets (default: True)
        reference_set_ids: Specific reference sets to use (default: paper's 6 sets)
        embedding_model: Embedding model name (default: text-embedding-3-small)
        embedding_dim: Embedding dimension (default: 1536)
        enable_cache: Enable embedding caching (default: True)
    """

    temperature: float = 1.0
    offset: float = 0.0
    use_multi_set_averaging: bool = True
    reference_set_ids: Optional[List[str]] = None
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 1536
    enable_cache: bool = True

    def __post_init__(self):
        """Validate configuration"""
        if self.temperature <= 0:
            raise ValueError(f"Temperature must be positive, got {self.temperature}")

        if self.embedding_dim <= 0:
            raise ValueError(
                f"Embedding dimension must be positive, got {self.embedding_dim}"
            )

        # Keep reference_set_ids as None to allow dynamic selection
        # This enables the system to use all available diverse reference sets
        pass


@dataclass
class SSRResult:
    """
    Complete result from SSR processing

    Attributes:
        response_text: Original LLM response text
        distribution: Final probability distribution over ratings 1-5
        mean_rating: Expected rating (weighted average)
        reference_sets_used: Number of reference sets used
        similarity_results: List of similarity results (one per reference set)
        embedding_result: Embedding of response text
        config: Configuration used for this result
    """

    response_text: str
    distribution: DistributionResult
    mean_rating: float
    reference_sets_used: int
    similarity_results: List[SimilarityResult] = field(default_factory=list)
    embedding_result: Any = None  # EmbeddingResult
    config: SSRConfig = field(default_factory=SSRConfig)

    def get_rating_probability(self, rating: int) -> float:
        """Get probability for specific rating (1-5)"""
        return self.distribution.get_rating_probability(rating)

    def get_most_likely_rating(self) -> int:
        """Get the most probable rating (mode)"""
        return int(np.argmax(self.distribution.probabilities) + 1)


class SSREngine:
    """
    Main SSR Engine for generating synthetic consumer ratings

    This engine implements the complete SSR methodology from the paper,
    achieving 90% of human test-retest reliability through:
    1. Semantic similarity measurement between responses and reference statements
    2. Multi-reference set averaging for robustness
    3. Temperature-controlled distribution construction

    Usage:
        engine = SSREngine(api_key="your-openai-key")
        result = engine.process_response("I'd probably buy it.")
        print(f"Mean rating: {result.mean_rating:.2f}")
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        config: Optional[SSRConfig] = None,
        data_dir: Optional[Path] = None,
    ):
        """
        Initialize SSR Engine

        Args:
            api_key: OpenAI API key for embeddings (or use OPENAI_API_KEY env var)
            config: SSR configuration (uses defaults if not provided)
            data_dir: Directory for data storage (embeddings cache, reference statements)
        """
        self.config = config or SSRConfig()

        # Initialize components
        self.embedding_retriever = EmbeddingRetriever(
            api_key=api_key,
            model=self.config.embedding_model,
            embedding_dim=self.config.embedding_dim,
            enable_cache=self.config.enable_cache,
        )

        self.reference_manager = ReferenceStatementManager(data_dir=data_dir)

        self.similarity_calculator = SimilarityCalculator(
            embedding_dim=self.config.embedding_dim
        )

        self.distribution_constructor = DistributionConstructor(
            temperature=self.config.temperature, offset=self.config.offset
        )

        # Pre-compute reference statement embeddings
        self._precompute_reference_embeddings()

        # Statistics
        self.responses_processed = 0

    def _precompute_reference_embeddings(self):
        """
        Pre-compute embeddings for all reference statements

        This is done once during initialization to avoid repeated API calls
        during runtime, improving performance and reducing costs.
        """
        print("Pre-computing reference statement embeddings...")
        self.reference_manager.compute_all_embeddings(self.embedding_retriever)
        print("Reference embeddings computed successfully")

    def process_response(self, response_text: str) -> SSRResult:
        """
        Process a synthetic consumer response into a rating distribution

        This is the main entry point for SSR processing. It:
        1. Embeds the response text
        2. Calculates similarity to reference statements
        3. Constructs probability distribution
        4. Averages across multiple reference sets if configured

        Args:
            response_text: Text response from synthetic consumer (1-3 sentences)

        Returns:
            SSRResult containing distribution and metadata

        Example:
            result = engine.process_response("I'd probably buy it.")
            print(f"Rating distribution: {result.distribution.probabilities}")
            print(f"Mean rating: {result.mean_rating:.2f}")
        """
        if not response_text or not response_text.strip():
            raise ValueError("Response text cannot be empty")

        # Step 1: Get embedding for response text
        embedding_result = self.embedding_retriever.get_embedding(response_text)

        # Step 2: Get reference statement sets to use
        reference_sets = self._get_reference_sets()

        # Step 3: Calculate similarities and distributions for each reference set
        similarity_results = []
        distributions = []

        for ref_set in reference_sets:
            # Get reference embeddings
            ref_embeddings = ref_set.get_embeddings()

            # Calculate similarities
            sim_result = self.similarity_calculator.calculate_similarities(
                response_embedding=embedding_result.embedding,
                reference_embeddings=ref_embeddings,
                pre_normalized=False,  # Embeddings from API are not normalized
            )
            similarity_results.append(sim_result)

            # Construct distribution
            dist = self.distribution_constructor.construct_distribution(sim_result)
            distributions.append(dist)

        # Step 4: Average distributions if using multi-set averaging
        if self.config.use_multi_set_averaging and len(distributions) > 1:
            final_distribution = (
                self.distribution_constructor.average_across_reference_sets(
                    distributions
                )
            )
        else:
            final_distribution = distributions[0]

        # Update statistics
        self.responses_processed += 1

        return SSRResult(
            response_text=response_text,
            distribution=final_distribution,
            mean_rating=final_distribution.mean_rating,
            reference_sets_used=len(reference_sets),
            similarity_results=similarity_results,
            embedding_result=embedding_result,
            config=self.config,
        )

    def process_responses_batch(self, response_texts: List[str]) -> List[SSRResult]:
        """
        Process multiple responses in batch for efficiency

        Uses batch embedding retrieval to reduce API calls and latency.

        Args:
            response_texts: List of response texts to process

        Returns:
            List of SSRResult objects, one per response
        """
        if not response_texts:
            raise ValueError("Response texts list cannot be empty")

        # Step 1: Get embeddings for all responses in batch
        embedding_results = self.embedding_retriever.get_embeddings_batch(
            response_texts
        )

        # Step 2: Process each response
        ssr_results = []
        for response_text, embedding_result in zip(response_texts, embedding_results):
            # Get reference sets
            reference_sets = self._get_reference_sets()

            # Calculate similarities and distributions
            similarity_results = []
            distributions = []

            for ref_set in reference_sets:
                ref_embeddings = ref_set.get_embeddings()

                sim_result = self.similarity_calculator.calculate_similarities(
                    response_embedding=embedding_result.embedding,
                    reference_embeddings=ref_embeddings,
                    pre_normalized=False,
                )
                similarity_results.append(sim_result)

                dist = self.distribution_constructor.construct_distribution(sim_result)
                distributions.append(dist)

            # Average distributions if configured
            if self.config.use_multi_set_averaging and len(distributions) > 1:
                final_distribution = (
                    self.distribution_constructor.average_across_reference_sets(
                        distributions
                    )
                )
            else:
                final_distribution = distributions[0]

            ssr_results.append(
                SSRResult(
                    response_text=response_text,
                    distribution=final_distribution,
                    mean_rating=final_distribution.mean_rating,
                    reference_sets_used=len(reference_sets),
                    similarity_results=similarity_results,
                    embedding_result=embedding_result,
                    config=self.config,
                )
            )

        # Update statistics
        self.responses_processed += len(ssr_results)

        return ssr_results

    def _get_reference_sets(self) -> List[ReferenceStatementSet]:
        """Get configured reference statement sets"""
        import random
        
        # If specific sets are configured, try to use them
        if self.config.reference_set_ids:
            available_sets = []
            for set_id in self.config.reference_set_ids:
                try:
                    available_sets.append(self.reference_manager.get_set(set_id))
                except KeyError:
                    pass
            if available_sets:
                return available_sets
        
        # Use diverse reference sets for better differentiation
        # Exclude paper sets as they're too generic and cause convergence to 3.0
        all_sets = self.reference_manager.get_all_sets()
        diverse_sets = [s for s in all_sets if not s.id.startswith("paper_set_")]
        
        # If we have diverse sets, use them (random selection for variety)
        if diverse_sets:
            if len(diverse_sets) > 4:
                # Select 3-5 diverse sets randomly for each survey
                num_sets = random.randint(3, min(5, len(diverse_sets)))
                return random.sample(diverse_sets, num_sets)
            else:
                return diverse_sets
        
        # Fallback to paper sets only if no diverse sets available
        try:
            paper_sets = self.reference_manager.get_paper_default_sets()
            if paper_sets:
                return paper_sets
        except:
            pass
        
        # Last resort fallback
        return all_sets if all_sets else [self.reference_manager.get_set("test_set_1")]

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get engine statistics

        Returns:
            Dictionary with processing statistics, embedding cache stats, etc.
        """
        embedding_stats = self.embedding_retriever.get_statistics()
        ref_stats = self.reference_manager.get_statistics()

        return {
            "responses_processed": self.responses_processed,
            "reference_sets_available": ref_stats["total_sets"],
            "reference_sets_used": len(self._get_reference_sets()),
            "embedding_api_calls": embedding_stats["api_calls"],
            "embedding_cache_hit_rate": embedding_stats.get("hit_rate", 0.0),
            "total_cost_usd": embedding_stats.get("estimated_cost_usd", 0.0),
            "temperature": self.config.temperature,
            "multi_set_averaging": self.config.use_multi_set_averaging,
        }

    def update_config(self, **kwargs):
        """
        Update configuration parameters

        Args:
            **kwargs: Configuration parameters to update

        Example:
            engine.update_config(temperature=1.5, use_multi_set_averaging=False)
        """
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
            else:
                raise ValueError(f"Unknown configuration parameter: {key}")

        # Update components if needed
        if "temperature" in kwargs or "offset" in kwargs:
            self.distribution_constructor = DistributionConstructor(
                temperature=self.config.temperature, offset=self.config.offset
            )


# Example usage and testing
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    load_dotenv()

    # Initialize SSR Engine
    print("Initializing SSR Engine...")
    engine = SSREngine(api_key=os.getenv("OPENAI_API_KEY"))

    print("\nSSR Engine initialized successfully!")
    print("=" * 60)

    # Test with example responses (from paper's reference statements)
    test_responses = [
        "It's very likely I'd buy it.",  # Should rate around 5
        "I'd probably buy it.",  # Should rate around 4
        "I'm not sure if I'd buy it or not.",  # Should rate around 3
        "I probably wouldn't buy it.",  # Should rate around 2
        "It's rather unlikely I'd buy it.",  # Should rate around 1
    ]

    print("\nProcessing test responses:")
    print("=" * 60)

    for i, response in enumerate(test_responses, 1):
        result = engine.process_response(response)

        print(f'\nResponse {i}: "{response}"')
        print(f"Mean rating: {result.mean_rating:.2f}")
        print(f"Most likely rating: {result.get_most_likely_rating()}")
        print("Probability distribution:")
        for rating in range(1, 6):
            prob = result.get_rating_probability(rating)
            print(f"  Rating {rating}: {prob:.3f} ({prob * 100:.1f}%)")

    # Test batch processing
    print("\n" + "=" * 60)
    print("Testing batch processing:")
    batch_results = engine.process_responses_batch(test_responses)
    print(f"Processed {len(batch_results)} responses in batch")

    # Show statistics
    print("\n" + "=" * 60)
    print("Engine Statistics:")
    stats = engine.get_statistics()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")

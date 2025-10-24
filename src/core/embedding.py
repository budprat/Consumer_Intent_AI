"""
ABOUTME: Embedding retrieval using OpenAI text-embedding-3-small model
ABOUTME: Provides caching, batch processing, and error handling for embedding operations

This module handles all embedding operations for the SSR system, including:
- Text-to-embedding conversion using OpenAI API
- Batch processing for efficiency
- Caching to reduce API calls and costs
- Retry logic for reliability

Embedding Model: text-embedding-3-small
- Dimensions: 1536
- Cost-effective and performant
- Used in the original research paper
"""

import numpy as np
import hashlib
import time
from typing import List, Dict, Optional, Union
from dataclasses import dataclass
from pathlib import Path
import pickle


from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

try:
    from openai import OpenAI, OpenAIError
except ImportError:
    OpenAI = None
    OpenAIError = Exception


@dataclass
class EmbeddingResult:
    """
    Result of embedding operation

    Attributes:
        embedding: The embedding vector (1536 dimensions)
        text: Original text that was embedded
        model: Model used for embedding
        cached: Whether result was retrieved from cache
        token_count: Number of tokens in the text (approximate)
    """

    embedding: np.ndarray
    text: str
    model: str
    cached: bool = False
    token_count: Optional[int] = None


class EmbeddingCache:
    """
    Persistent cache for embeddings to reduce API calls

    Features:
    - Hash-based key generation (content-addressable)
    - Persistent storage using pickle
    - Thread-safe operations
    - Cache statistics tracking
    """

    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize embedding cache

        Args:
            cache_dir: Directory for cache storage (default: data/cache)
        """
        if cache_dir is None:
            cache_dir = Path(__file__).parent.parent.parent / "data" / "cache"

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.cache_file = self.cache_dir / "embeddings.pkl"
        self._cache: Dict[str, Dict] = self._load_cache()

        # Statistics
        self.hits = 0
        self.misses = 0

    def _load_cache(self) -> Dict[str, Dict]:
        """Load cache from disk"""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, "rb") as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"Warning: Failed to load cache: {e}")
                return {}
        return {}

    def _save_cache(self):
        """Save cache to disk"""
        try:
            with open(self.cache_file, "wb") as f:
                pickle.dump(self._cache, f)
        except Exception as e:
            print(f"Warning: Failed to save cache: {e}")

    def _generate_key(self, text: str, model: str) -> str:
        """
        Generate cache key from text and model

        Uses SHA256 hash of text and model for content-addressable caching

        Args:
            text: Input text
            model: Embedding model name

        Returns:
            Hex digest of hash
        """
        content = f"{model}|{text}"
        return hashlib.sha256(content.encode()).hexdigest()

    def get(self, text: str, model: str) -> Optional[np.ndarray]:
        """
        Retrieve embedding from cache

        Args:
            text: Input text
            model: Embedding model name

        Returns:
            Cached embedding or None if not found
        """
        key = self._generate_key(text, model)

        if key in self._cache:
            self.hits += 1
            cached_data = self._cache[key]
            # Convert list back to numpy array
            return np.array(cached_data["embedding"])
        else:
            self.misses += 1
            return None

    def set(self, text: str, model: str, embedding: np.ndarray):
        """
        Store embedding in cache

        Args:
            text: Input text
            model: Embedding model name
            embedding: Embedding vector to cache
        """
        key = self._generate_key(text, model)

        self._cache[key] = {
            "embedding": embedding.tolist(),  # Convert numpy to list for JSON
            "text": text,
            "model": model,
            "timestamp": time.time(),
        }

        # Save cache periodically (every 10 new entries)
        if len(self._cache) % 10 == 0:
            self._save_cache()

    def clear(self):
        """Clear all cached embeddings"""
        self._cache = {}
        self._save_cache()

    def get_statistics(self) -> Dict[str, Union[int, float]]:
        """
        Get cache statistics

        Returns:
            Dictionary with hits, misses, hit_rate, size
        """
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0.0

        return {
            "hits": self.hits,
            "misses": self.misses,
            "total_requests": total,
            "hit_rate": hit_rate,
            "cache_size": len(self._cache),
        }


class EmbeddingRetriever:
    """
    Retrieves embeddings from OpenAI API with caching and batch processing

    Features:
    - OpenAI text-embedding-3-small integration
    - Automatic caching to reduce API calls
    - Batch processing up to 2048 texts
    - Retry logic for reliability
    - Cost tracking
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "text-embedding-3-small",
        embedding_dim: int = 1536,
        cache_dir: Optional[Path] = None,
        enable_cache: bool = True,
    ):
        """
        Initialize embedding retriever

        Args:
            api_key: OpenAI API key (or use OPENAI_API_KEY env var)
            model: Embedding model name (default: text-embedding-3-small)
            embedding_dim: Embedding dimension (default: 1536)
            cache_dir: Directory for cache storage
            enable_cache: Whether to use caching
        """
        if OpenAI is None:
            raise ImportError(
                "OpenAI library not installed. Install with: pip install openai"
            )

        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.embedding_dim = embedding_dim

        self.enable_cache = enable_cache
        self.cache = EmbeddingCache(cache_dir) if enable_cache else None

        # Statistics
        self.api_calls = 0
        self.total_tokens = 0

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(OpenAIError),
    )
    def get_embedding(self, text: str) -> EmbeddingResult:
        """
        Get embedding for single text

        Args:
            text: Text to embed

        Returns:
            EmbeddingResult with embedding vector

        Raises:
            OpenAIError: If API call fails after retries
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")

        # Check cache first
        if self.enable_cache:
            cached_embedding = self.cache.get(text, self.model)
            if cached_embedding is not None:
                return EmbeddingResult(
                    embedding=cached_embedding,
                    text=text,
                    model=self.model,
                    cached=True,
                )

        # Call API
        try:
            response = self.client.embeddings.create(
                input=text, model=self.model, encoding_format="float"
            )

            embedding = np.array(response.data[0].embedding, dtype=np.float32)
            token_count = response.usage.total_tokens

            # Update statistics
            self.api_calls += 1
            self.total_tokens += token_count

            # Cache result
            if self.enable_cache:
                self.cache.set(text, self.model, embedding)

            return EmbeddingResult(
                embedding=embedding,
                text=text,
                model=self.model,
                cached=False,
                token_count=token_count,
            )

        except Exception as e:
            raise OpenAIError(f"Failed to get embedding: {str(e)}")

    def get_embeddings_batch(self, texts: List[str]) -> List[EmbeddingResult]:
        """
        Get embeddings for multiple texts in batch

        Args:
            texts: List of texts to embed (max 2048)

        Returns:
            List of EmbeddingResult objects

        Note:
            OpenAI API supports batch sizes up to 2048 texts
            This method handles caching and only calls API for non-cached texts
        """
        if not texts:
            raise ValueError("Texts list cannot be empty")

        if len(texts) > 2048:
            raise ValueError(
                f"Batch size exceeds OpenAI limit: {len(texts)} > 2048. "
                "Consider splitting into smaller batches."
            )

        results = []
        texts_to_fetch = []
        fetch_indices = []

        # Check cache and identify texts that need to be fetched
        for i, text in enumerate(texts):
            if not text or not text.strip():
                raise ValueError(f"Text at index {i} is empty")

            if self.enable_cache:
                cached_embedding = self.cache.get(text, self.model)
                if cached_embedding is not None:
                    results.append(
                        EmbeddingResult(
                            embedding=cached_embedding,
                            text=text,
                            model=self.model,
                            cached=True,
                        )
                    )
                    continue

            # Need to fetch this text
            texts_to_fetch.append(text)
            fetch_indices.append(i)

        # Fetch uncached texts from API
        if texts_to_fetch:
            try:
                response = self.client.embeddings.create(
                    input=texts_to_fetch, model=self.model, encoding_format="float"
                )

                # Process results
                for text, embedding_data in zip(texts_to_fetch, response.data):
                    embedding = np.array(embedding_data.embedding, dtype=np.float32)

                    # Cache result
                    if self.enable_cache:
                        self.cache.set(text, self.model, embedding)

                    results.append(
                        EmbeddingResult(
                            embedding=embedding,
                            text=text,
                            model=self.model,
                            cached=False,
                        )
                    )

                # Update statistics
                self.api_calls += 1
                self.total_tokens += response.usage.total_tokens

            except Exception as e:
                raise OpenAIError(f"Failed to get batch embeddings: {str(e)}")

        # Sort results back to original order
        results.sort(key=lambda r: texts.index(r.text))

        return results

    def get_statistics(self) -> Dict[str, Union[int, float]]:
        """
        Get retriever statistics

        Returns:
            Dictionary with API calls, tokens, costs, and cache stats
        """
        stats = {
            "api_calls": self.api_calls,
            "total_tokens": self.total_tokens,
            "estimated_cost_usd": self.total_tokens
            * 0.00002
            / 1000,  # $0.02 per 1M tokens
        }

        if self.enable_cache:
            cache_stats = self.cache.get_statistics()
            stats.update(cache_stats)

        return stats


# Example usage and testing
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    load_dotenv()

    # Initialize retriever
    retriever = EmbeddingRetriever(
        api_key=os.getenv("OPENAI_API_KEY"), enable_cache=True
    )

    # Test single embedding
    print("Testing single embedding:")
    text = "It's very likely I'd buy it."
    result = retriever.get_embedding(text)
    print(f"Text: {text}")
    print(f"Embedding shape: {result.embedding.shape}")
    print(f"Cached: {result.cached}")
    print(f"Tokens: {result.token_count}")

    # Test caching (should be cached on second call)
    print("\nTesting cache:")
    result2 = retriever.get_embedding(text)
    print(f"Second call cached: {result2.cached}")

    # Test batch processing
    print("\nTesting batch processing:")
    texts = [
        "It's rather unlikely I'd buy it.",
        "I probably wouldn't buy it.",
        "I'm not sure if I'd buy it or not.",
        "I'd probably buy it.",
        "It's very likely I'd buy it.",
    ]

    batch_results = retriever.get_embeddings_batch(texts)
    print(f"Batch size: {len(batch_results)}")
    for i, res in enumerate(batch_results, 1):
        print(f"  {i}. Cached: {res.cached}, Shape: {res.embedding.shape}")

    # Print statistics
    print("\nStatistics:")
    stats = retriever.get_statistics()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")

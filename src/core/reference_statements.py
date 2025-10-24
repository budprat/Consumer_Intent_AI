"""
ABOUTME: Reference statement management system for SSR methodology
ABOUTME: Handles storage, retrieval, and versioning of reference statements with embedding caching

This module manages reference statements used in the SSR methodology.
The paper uses 6 different reference statement sets, each expressing
the same sentiment gradient with different phrasing for robustness.

Reference Statement Structure:
- 5 statements per set (ratings 1-5)
- 6 sets total (as used in the paper)
- Each statement has pre-computed embedding for efficiency
"""

import numpy as np
import yaml
from typing import List, Dict, Optional
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime


@dataclass
class ReferenceStatement:
    """
    Single reference statement with metadata

    Attributes:
        rating: Likert rating (1-5)
        text: Reference statement text
        embedding: Pre-computed embedding vector (1536 dimensions)
        embedding_cached: Whether embedding has been computed
    """

    rating: int
    text: str
    embedding: Optional[np.ndarray] = None
    embedding_cached: bool = False

    def __post_init__(self):
        """Validate rating is 1-5"""
        if not 1 <= self.rating <= 5:
            raise ValueError(f"Rating must be 1-5, got {self.rating}")


@dataclass
class ReferenceStatementSet:
    """
    Set of 5 reference statements (one for each rating)

    Attributes:
        id: Unique identifier for this set
        version: Version string (e.g., "1.0")
        domain: Domain/category (e.g., "consumer_products")
        language: Language code (e.g., "en")
        statements: List of 5 ReferenceStatement objects
        description: Optional description of this set
        created_at: Creation timestamp
    """

    id: str
    version: str
    domain: str
    language: str
    statements: List[ReferenceStatement]
    description: Optional[str] = None
    created_at: Optional[str] = None

    def __post_init__(self):
        """Validate that we have exactly 5 statements"""
        if len(self.statements) != 5:
            raise ValueError(
                f"Reference set must have exactly 5 statements, got {len(self.statements)}"
            )

        # Validate statements are ordered by rating
        for i, stmt in enumerate(self.statements, 1):
            if stmt.rating != i:
                raise ValueError(
                    f"Statements must be ordered by rating 1-5. "
                    f"Expected rating {i} at position {i - 1}, got {stmt.rating}"
                )

        if self.created_at is None:
            self.created_at = datetime.now().isoformat()

    def get_embeddings(self) -> np.ndarray:
        """
        Get embeddings for all statements in order

        Returns:
            Array of shape (5, 1536) with embeddings for ratings 1-5

        Raises:
            ValueError: If any embeddings are not computed yet
        """
        if not all(stmt.embedding_cached for stmt in self.statements):
            raise ValueError(
                "Not all embeddings are computed. Call compute_embeddings() first."
            )

        return np.array([stmt.embedding for stmt in self.statements])

    def compute_embeddings(self, embedding_retriever):
        """
        Compute embeddings for all statements using embedding retriever

        Args:
            embedding_retriever: EmbeddingRetriever instance to use for embedding

        Note:
            This method caches embeddings in the statements themselves
        """

        texts = [stmt.text for stmt in self.statements]
        results = embedding_retriever.get_embeddings_batch(texts)

        for stmt, result in zip(self.statements, results):
            stmt.embedding = result.embedding
            stmt.embedding_cached = True


class ReferenceStatementManager:
    """
    Manages reference statement sets with persistence and caching

    Features:
    - Store and retrieve multiple reference statement sets
    - Pre-compute and cache embeddings
    - Load from YAML configuration files
    - Version control and validation
    """

    def __init__(self, data_dir: Optional[Path] = None):
        """
        Initialize reference statement manager

        Args:
            data_dir: Directory for storing reference statements
        """
        if data_dir is None:
            data_dir = (
                Path(__file__).parent.parent.parent / "data" / "reference_statements"
            )

        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self._sets: Dict[str, ReferenceStatementSet] = {}
        self._load_default_sets()

    def _load_default_sets(self):
        """Load default reference statement sets from data directory"""
        # Try to load from YAML files
        for yaml_file in self.data_dir.glob("*.yaml"):
            try:
                ref_set = self.load_from_yaml(yaml_file)
                self._sets[ref_set.id] = ref_set
            except Exception as e:
                print(f"Warning: Failed to load {yaml_file}: {e}")
        
        # Try to load from JSON files (validated_sets.json)
        json_path = Path(__file__).parent.parent.parent / "data" / "reference_sets" / "validated_sets.json"
        if json_path.exists():
            try:
                import json
                with open(json_path) as f:
                    data = json.load(f)
                    for ref_set_data in data.get("reference_sets", []):
                        statements = []
                        for rating_str, text in ref_set_data["statements"].items():
                            statements.append(ReferenceStatement(int(rating_str), text))
                        
                        ref_set = ReferenceStatementSet(
                            id=ref_set_data["set_id"],
                            version="1.0",
                            domain="consumer_products",
                            language="en",
                            description=ref_set_data.get("description", ref_set_data["name"]),
                            statements=statements
                        )
                        self._sets[ref_set.id] = ref_set
                print(f"Loaded {len(self._sets)} reference sets from JSON")
            except Exception as e:
                print(f"Warning: Failed to load JSON reference sets: {e}")

        # If no sets loaded, create paper's default sets
        if not self._sets:
            self._create_paper_default_sets()

    def _create_paper_default_sets(self):
        """
        Create the 6 reference statement sets from the paper

        These are the reference statements used in the original research paper
        "Using LLMs as Synthetic Consumers for Purchase Intent Surveys"
        """
        # Set 1: Direct likelihood statements
        set1 = ReferenceStatementSet(
            id="paper_set_1",
            version="1.0",
            domain="consumer_products",
            language="en",
            description="Direct likelihood statements (paper reference set 1)",
            statements=[
                ReferenceStatement(1, "It's rather unlikely I'd buy it."),
                ReferenceStatement(2, "I probably wouldn't buy it."),
                ReferenceStatement(3, "I'm not sure if I'd buy it or not."),
                ReferenceStatement(4, "I'd probably buy it."),
                ReferenceStatement(5, "It's very likely I'd buy it."),
            ],
        )

        # Set 2: Interest-based statements
        set2 = ReferenceStatementSet(
            id="paper_set_2",
            version="1.0",
            domain="consumer_products",
            language="en",
            description="Interest-based statements (paper reference set 2)",
            statements=[
                ReferenceStatement(1, "I'm not interested in this at all."),
                ReferenceStatement(2, "I don't think this would interest me."),
                ReferenceStatement(3, "I'm somewhat unsure about my interest."),
                ReferenceStatement(4, "This seems interesting to me."),
                ReferenceStatement(5, "I'm very interested in this product."),
            ],
        )

        # Set 3: Appeal-based statements
        set3 = ReferenceStatementSet(
            id="paper_set_3",
            version="1.0",
            domain="consumer_products",
            language="en",
            description="Appeal-based statements (paper reference set 3)",
            statements=[
                ReferenceStatement(1, "This doesn't appeal to me."),
                ReferenceStatement(2, "This product has limited appeal."),
                ReferenceStatement(3, "I have mixed feelings about this."),
                ReferenceStatement(4, "This product appeals to me."),
                ReferenceStatement(5, "This really appeals to me."),
            ],
        )

        # Set 4: Consideration-based statements
        set4 = ReferenceStatementSet(
            id="paper_set_4",
            version="1.0",
            domain="consumer_products",
            language="en",
            description="Consideration-based statements (paper reference set 4)",
            statements=[
                ReferenceStatement(1, "I wouldn't consider buying this."),
                ReferenceStatement(2, "I'd be hesitant to purchase this."),
                ReferenceStatement(3, "I might consider this product."),
                ReferenceStatement(4, "I would consider purchasing this."),
                ReferenceStatement(5, "I would definitely consider buying this."),
            ],
        )

        # Set 5: Value-based statements
        set5 = ReferenceStatementSet(
            id="paper_set_5",
            version="1.0",
            domain="consumer_products",
            language="en",
            description="Value-based statements (paper reference set 5)",
            statements=[
                ReferenceStatement(1, "This doesn't seem worth it to me."),
                ReferenceStatement(2, "I question the value of this product."),
                ReferenceStatement(3, "I'm uncertain about the value here."),
                ReferenceStatement(4, "This seems like a good value."),
                ReferenceStatement(5, "This is excellent value for me."),
            ],
        )

        # Set 6: Decision-based statements
        set6 = ReferenceStatementSet(
            id="paper_set_6",
            version="1.0",
            domain="consumer_products",
            language="en",
            description="Decision-based statements (paper reference set 6)",
            statements=[
                ReferenceStatement(1, "I would pass on this product."),
                ReferenceStatement(2, "I'm inclined to skip this."),
                ReferenceStatement(3, "I'm on the fence about this."),
                ReferenceStatement(4, "I'm leaning toward getting this."),
                ReferenceStatement(5, "I would choose to buy this."),
            ],
        )

        # Store all sets
        for ref_set in [set1, set2, set3, set4, set5, set6]:
            self._sets[ref_set.id] = ref_set
            # Save to YAML for persistence
            self.save_to_yaml(ref_set)

    def add_set(self, ref_set: ReferenceStatementSet):
        """
        Add new reference statement set

        Args:
            ref_set: ReferenceStatementSet to add

        Raises:
            ValueError: If set ID already exists
        """
        if ref_set.id in self._sets:
            raise ValueError(f"Reference set with ID '{ref_set.id}' already exists")

        self._sets[ref_set.id] = ref_set
        self.save_to_yaml(ref_set)

    def get_set(self, set_id: str) -> ReferenceStatementSet:
        """
        Get reference statement set by ID

        Args:
            set_id: Unique identifier for the set

        Returns:
            ReferenceStatementSet

        Raises:
            KeyError: If set ID not found
        """
        if set_id not in self._sets:
            raise KeyError(f"Reference set '{set_id}' not found")

        return self._sets[set_id]

    def get_all_sets(self) -> List[ReferenceStatementSet]:
        """Get all available reference statement sets"""
        return list(self._sets.values())

    def get_paper_default_sets(self) -> List[ReferenceStatementSet]:
        """Get the 6 default reference statement sets from the paper"""
        paper_sets = [
            self._sets[f"paper_set_{i}"]
            for i in range(1, 7)
            if f"paper_set_{i}" in self._sets
        ]
        return paper_sets if paper_sets else []

    def compute_all_embeddings(self, embedding_retriever):
        """
        Compute embeddings for all reference statement sets

        Args:
            embedding_retriever: EmbeddingRetriever instance

        Note:
            This should be called once during initialization to pre-compute
            all embeddings for efficient runtime performance
        """
        for ref_set in self._sets.values():
            if not all(stmt.embedding_cached for stmt in ref_set.statements):
                ref_set.compute_embeddings(embedding_retriever)

        print(f"Computed embeddings for {len(self._sets)} reference statement sets")

    def save_to_yaml(self, ref_set: ReferenceStatementSet):
        """
        Save reference statement set to YAML file

        Args:
            ref_set: ReferenceStatementSet to save
        """
        yaml_path = self.data_dir / f"{ref_set.id}.yaml"

        data = {
            "id": ref_set.id,
            "version": ref_set.version,
            "domain": ref_set.domain,
            "language": ref_set.language,
            "description": ref_set.description,
            "created_at": ref_set.created_at,
            "statements": [
                {"rating": stmt.rating, "text": stmt.text}
                for stmt in ref_set.statements
            ],
        }

        with open(yaml_path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    def load_from_yaml(self, yaml_path: Path) -> ReferenceStatementSet:
        """
        Load reference statement set from YAML file

        Args:
            yaml_path: Path to YAML file

        Returns:
            ReferenceStatementSet loaded from file
        """
        with open(yaml_path) as f:
            data = yaml.safe_load(f)

        statements = [
            ReferenceStatement(rating=stmt["rating"], text=stmt["text"])
            for stmt in data["statements"]
        ]

        return ReferenceStatementSet(
            id=data["id"],
            version=data["version"],
            domain=data["domain"],
            language=data["language"],
            statements=statements,
            description=data.get("description"),
            created_at=data.get("created_at"),
        )

    def validate_set(self, ref_set: ReferenceStatementSet) -> Dict[str, bool]:
        """
        Validate reference statement set

        Checks:
        - 5 statements present
        - Statements ordered by rating
        - All embeddings computed
        - Embedding dimensions correct

        Args:
            ref_set: ReferenceStatementSet to validate

        Returns:
            Dictionary of validation results
        """
        results = {
            "has_five_statements": len(ref_set.statements) == 5,
            "correctly_ordered": all(
                stmt.rating == i for i, stmt in enumerate(ref_set.statements, 1)
            ),
            "embeddings_computed": all(
                stmt.embedding_cached for stmt in ref_set.statements
            ),
        }

        if results["embeddings_computed"]:
            results["correct_embedding_dim"] = all(
                stmt.embedding.shape == (1536,) for stmt in ref_set.statements
            )
        else:
            results["correct_embedding_dim"] = False

        results["all_valid"] = all(results.values())

        return results

    def get_statistics(self) -> Dict[str, int]:
        """Get statistics about managed reference statement sets"""
        embedded_count = sum(
            1
            for ref_set in self._sets.values()
            if all(stmt.embedding_cached for stmt in ref_set.statements)
        )

        return {
            "total_sets": len(self._sets),
            "sets_with_embeddings": embedded_count,
            "total_statements": len(self._sets) * 5,
        }


# Example usage and testing
if __name__ == "__main__":
    # Initialize manager
    manager = ReferenceStatementManager()

    print("Reference Statement Manager initialized")
    print(f"Loaded {len(manager.get_all_sets())} reference statement sets\n")

    # Show paper's default sets
    paper_sets = manager.get_paper_default_sets()
    print("Paper's 6 default reference statement sets:")
    for i, ref_set in enumerate(paper_sets, 1):
        print(f"\nSet {i}: {ref_set.description}")
        print("Statements:")
        for stmt in ref_set.statements:
            print(f'  Rating {stmt.rating}: "{stmt.text}"')

    # Show statistics
    print("\n" + "=" * 60)
    stats = manager.get_statistics()
    print("Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Validate all sets
    print("\n" + "=" * 60)
    print("Validation (before embedding computation):")
    for ref_set in paper_sets:
        results = manager.validate_set(ref_set)
        print(f"\n{ref_set.id}:")
        for check, passed in results.items():
            status = "✓" if passed else "✗"
            print(f"  {status} {check}")

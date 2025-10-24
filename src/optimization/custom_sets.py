"""
ABOUTME: Custom reference statement set creation and validation framework
ABOUTME: Enables domain-specific reference sets beyond the paper's 6 baseline sets

This module provides tools for creating custom reference statement sets tailored
to specific product categories, industries, or use cases. While the paper provides
6 general-purpose reference sets, certain domains may benefit from specialized
statements that capture domain-specific sentiment and purchase intent.

Use Cases:
- Healthcare products (regulatory language, safety concerns)
- Financial services (trust, security, compliance)
- Luxury goods (prestige, exclusivity, quality)
- B2B software (ROI, integration, support)
- Food and beverage (taste, health, convenience)

The validation framework ensures new sets meet minimum quality standards before
integration with the existing 6 baseline sets.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path
import yaml
import json
import numpy as np
from enum import Enum


class DomainCategory(Enum):
    """Product domain categories for specialized reference sets"""

    GENERAL = "general"  # Paper's baseline (consumer products)
    HEALTHCARE = "healthcare"  # Medical devices, pharmaceuticals
    FINANCIAL = "financial"  # Banking, insurance, investments
    LUXURY = "luxury"  # High-end consumer goods
    B2B_SOFTWARE = "b2b_software"  # Enterprise software
    FOOD_BEVERAGE = "food_beverage"  # Food and drink products
    AUTOMOTIVE = "automotive"  # Vehicles and automotive products
    REAL_ESTATE = "real_estate"  # Property and housing
    EDUCATION = "education"  # Courses, training, learning
    ENTERTAINMENT = "entertainment"  # Media, games, streaming
    CUSTOM = "custom"  # User-defined domain


@dataclass
class ReferenceStatement:
    """
    Individual reference statement

    Attributes:
        text: The statement text
        rating: Associated Likert rating (1-5)
        category: Type of statement (intent, appeal, consideration, etc.)
        domain: Product domain this statement is optimized for
        metadata: Additional metadata (author, date, validation metrics)
    """

    text: str
    rating: int
    category: str = "intent"
    domain: DomainCategory = DomainCategory.GENERAL
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate statement"""
        if not 1 <= self.rating <= 5:
            raise ValueError(f"Rating must be 1-5, got {self.rating}")
        if len(self.text.strip()) < 10:
            raise ValueError("Statement text must be at least 10 characters")


@dataclass
class CustomReferenceSet:
    """
    Custom reference statement set

    Attributes:
        set_id: Unique identifier for this set
        name: Human-readable name
        description: Description of set purpose and domain
        domain: Product domain category
        statements: List of 5 reference statements (one per rating)
        validation_metrics: Optional quality metrics from validation
        version: Version number for tracking changes
        author: Creator of this set
        created_date: Creation timestamp
    """

    set_id: str
    name: str
    description: str
    domain: DomainCategory
    statements: List[ReferenceStatement]
    validation_metrics: Optional[Dict[str, float]] = None
    version: str = "1.0.0"
    author: str = "unknown"
    created_date: Optional[str] = None

    def __post_init__(self):
        """Validate set"""
        if len(self.statements) != 5:
            raise ValueError(
                f"Must have exactly 5 statements, got {len(self.statements)}"
            )

        # Check that ratings cover 1-5
        ratings = {stmt.rating for stmt in self.statements}
        if ratings != {1, 2, 3, 4, 5}:
            raise ValueError(f"Statements must cover ratings 1-5, got {ratings}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "set_id": self.set_id,
            "name": self.name,
            "description": self.description,
            "domain": self.domain.value,
            "version": self.version,
            "author": self.author,
            "created_date": self.created_date,
            "validation_metrics": self.validation_metrics,
            "statements": [
                {
                    "text": stmt.text,
                    "rating": stmt.rating,
                    "category": stmt.category,
                    "domain": stmt.domain.value,
                    "metadata": stmt.metadata,
                }
                for stmt in self.statements
            ],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CustomReferenceSet":
        """Create from dictionary"""
        statements = [
            ReferenceStatement(
                text=stmt["text"],
                rating=stmt["rating"],
                category=stmt.get("category", "intent"),
                domain=DomainCategory(stmt.get("domain", "general")),
                metadata=stmt.get("metadata", {}),
            )
            for stmt in data["statements"]
        ]

        return cls(
            set_id=data["set_id"],
            name=data["name"],
            description=data["description"],
            domain=DomainCategory(data.get("domain", "general")),
            statements=statements,
            validation_metrics=data.get("validation_metrics"),
            version=data.get("version", "1.0.0"),
            author=data.get("author", "unknown"),
            created_date=data.get("created_date"),
        )


@dataclass
class ValidationResult:
    """
    Result of reference set validation

    Attributes:
        is_valid: Whether set passes validation
        quality_score: Overall quality score (0.0-1.0)
        issues: List of validation issues found
        warnings: List of non-critical warnings
        recommendations: List of improvement recommendations
        detailed_metrics: Detailed quality metrics
    """

    is_valid: bool
    quality_score: float
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    detailed_metrics: Dict[str, float] = field(default_factory=dict)

    def get_summary(self) -> str:
        """Generate validation summary"""
        lines = [
            "Validation Result",
            "=" * 60,
            f"Status: {'✅ VALID' if self.is_valid else '❌ INVALID'}",
            f"Quality Score: {self.quality_score:.2f}/1.00",
        ]

        if self.issues:
            lines.append("\n❌ Critical Issues:")
            for issue in self.issues:
                lines.append(f"  • {issue}")

        if self.warnings:
            lines.append("\n⚠️  Warnings:")
            for warning in self.warnings:
                lines.append(f"  • {warning}")

        if self.recommendations:
            lines.append("\n💡 Recommendations:")
            for rec in self.recommendations:
                lines.append(f"  • {rec}")

        return "\n".join(lines)


class CustomSetBuilder:
    """
    Builder for creating custom reference statement sets

    Features:
    - Template-based generation
    - Domain-specific optimization
    - Quality validation
    - Export/import functionality
    """

    def __init__(self):
        """Initialize custom set builder"""
        self.templates = self._load_templates()

    def create_from_template(
        self,
        set_id: str,
        name: str,
        domain: DomainCategory,
        product_context: Optional[Dict[str, str]] = None,
    ) -> CustomReferenceSet:
        """
        Create reference set from domain template

        Args:
            set_id: Unique identifier
            name: Human-readable name
            domain: Product domain
            product_context: Optional context for customization

        Returns:
            CustomReferenceSet with template-based statements
        """
        # Get template for domain
        template = self.templates.get(domain.value, self.templates["general"])

        # Create statements from template
        statements = []
        for rating, statement_template in template.items():
            # Optionally customize with product context
            if product_context:
                text = statement_template.format(**product_context)
            else:
                text = statement_template

            statements.append(
                ReferenceStatement(
                    text=text,
                    rating=rating,
                    category="intent",
                    domain=domain,
                )
            )

        return CustomReferenceSet(
            set_id=set_id,
            name=name,
            description=f"Custom reference set for {domain.value} domain",
            domain=domain,
            statements=sorted(statements, key=lambda s: s.rating),
        )

    def create_from_statements(
        self,
        set_id: str,
        name: str,
        description: str,
        domain: DomainCategory,
        statement_texts: List[str],
        ratings: Optional[List[int]] = None,
    ) -> CustomReferenceSet:
        """
        Create reference set from explicit statement texts

        Args:
            set_id: Unique identifier
            name: Human-readable name
            description: Set description
            domain: Product domain
            statement_texts: List of 5 statement texts
            ratings: Optional ratings (defaults to [1, 2, 3, 4, 5])

        Returns:
            CustomReferenceSet
        """
        if len(statement_texts) != 5:
            raise ValueError(
                f"Must provide exactly 5 statements, got {len(statement_texts)}"
            )

        if ratings is None:
            ratings = [1, 2, 3, 4, 5]

        if len(ratings) != 5:
            raise ValueError(f"Must provide 5 ratings, got {len(ratings)}")

        statements = [
            ReferenceStatement(
                text=text, rating=rating, category="intent", domain=domain
            )
            for text, rating in zip(statement_texts, ratings)
        ]

        return CustomReferenceSet(
            set_id=set_id,
            name=name,
            description=description,
            domain=domain,
            statements=statements,
        )

    def validate_set(
        self,
        reference_set: CustomReferenceSet,
        min_quality_score: float = 0.6,
        strict: bool = False,
    ) -> ValidationResult:
        """
        Validate reference set quality

        Args:
            reference_set: Set to validate
            min_quality_score: Minimum acceptable quality (default: 0.6)
            strict: Whether to apply strict validation rules

        Returns:
            ValidationResult with validation outcome
        """
        issues = []
        warnings = []
        recommendations = []
        detailed_metrics = {}

        # Check 1: Statement count and rating coverage
        if len(reference_set.statements) != 5:
            issues.append(
                f"Must have 5 statements, found {len(reference_set.statements)}"
            )

        ratings = {stmt.rating for stmt in reference_set.statements}
        if ratings != {1, 2, 3, 4, 5}:
            issues.append(f"Must cover ratings 1-5, found {ratings}")

        # Check 2: Statement length and quality
        for stmt in reference_set.statements:
            word_count = len(stmt.text.split())

            # Too short
            if word_count < 3:
                issues.append(
                    f"Rating {stmt.rating} statement too short ({word_count} words)"
                )
            elif word_count < 5:
                warnings.append(
                    f"Rating {stmt.rating} statement is short ({word_count} words)"
                )

            # Too long
            if word_count > 30:
                warnings.append(
                    f"Rating {stmt.rating} statement is long ({word_count} words)"
                )

            # Check for proper sentiment alignment
            sentiment_score = self._estimate_sentiment(stmt.text)
            expected_sentiment = (stmt.rating - 3) / 2  # Map 1-5 to -1 to +1

            sentiment_diff = abs(sentiment_score - expected_sentiment)
            detailed_metrics[f"sentiment_alignment_{stmt.rating}"] = (
                1.0 - sentiment_diff
            )

            if sentiment_diff > 0.5:
                warnings.append(
                    f"Rating {stmt.rating} statement sentiment ({sentiment_score:.2f}) "
                    f"misaligned with rating ({expected_sentiment:.2f})"
                )

        # Check 3: Statement diversity (avoid very similar statements)
        similarity_matrix = self._calculate_statement_similarities(
            [stmt.text for stmt in reference_set.statements]
        )

        max_similarity = np.max(similarity_matrix - np.eye(5))  # Exclude diagonal
        detailed_metrics["max_inter_statement_similarity"] = max_similarity

        if max_similarity > 0.95:
            issues.append(
                f"Statements too similar (max similarity: {max_similarity:.2f})"
            )
        elif max_similarity > 0.85:
            warnings.append(
                f"High statement similarity detected ({max_similarity:.2f})"
            )

        # Check 4: Domain appropriateness (if not general)
        if reference_set.domain != DomainCategory.GENERAL:
            has_domain_terms = any(
                self._check_domain_terms(stmt.text, reference_set.domain)
                for stmt in reference_set.statements
            )
            if not has_domain_terms and strict:
                warnings.append(
                    f"No domain-specific terms found for {reference_set.domain.value}"
                )

        # Calculate overall quality score
        quality_components = []

        # Component 1: Structure validity (0.3 weight)
        structure_valid = len(issues) == 0
        quality_components.append(1.0 if structure_valid else 0.0)

        # Component 2: Sentiment alignment (0.4 weight)
        sentiment_scores = [
            detailed_metrics.get(f"sentiment_alignment_{r}", 0.5) for r in range(1, 6)
        ]
        avg_sentiment_alignment = np.mean(sentiment_scores)
        quality_components.append(avg_sentiment_alignment)

        # Component 3: Statement diversity (0.3 weight)
        diversity_score = 1.0 - min(max_similarity, 1.0)
        quality_components.append(diversity_score)

        weights = [0.3, 0.4, 0.3]
        quality_score = np.average(quality_components, weights=weights)

        detailed_metrics["overall_quality"] = quality_score
        detailed_metrics["structure_validity"] = quality_components[0]
        detailed_metrics["sentiment_alignment"] = quality_components[1]
        detailed_metrics["statement_diversity"] = quality_components[2]

        # Determine validity
        is_valid = len(issues) == 0 and quality_score >= min_quality_score

        # Generate recommendations
        if quality_score < 0.7:
            recommendations.append(
                "Consider revising statements for better sentiment alignment"
            )
        if max_similarity > 0.85:
            recommendations.append(
                "Add more diversity to statement phrasing and perspectives"
            )
        if reference_set.domain != DomainCategory.GENERAL and not has_domain_terms:
            recommendations.append(
                f"Include domain-specific terminology for {reference_set.domain.value}"
            )

        return ValidationResult(
            is_valid=is_valid,
            quality_score=quality_score,
            issues=issues,
            warnings=warnings,
            recommendations=recommendations,
            detailed_metrics=detailed_metrics,
        )

    def export_set(
        self, reference_set: CustomReferenceSet, file_path: Path, format: str = "yaml"
    ) -> None:
        """
        Export reference set to file

        Args:
            reference_set: Set to export
            file_path: Path to save file
            format: File format ('yaml' or 'json')
        """
        data = reference_set.to_dict()

        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        if format == "yaml":
            with open(file_path, "w", encoding="utf-8") as f:
                yaml.dump(data, f, default_flow_style=False, sort_keys=False)
        elif format == "json":
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        else:
            raise ValueError(f"Unknown format: {format}")

    def import_set(self, file_path: Path) -> CustomReferenceSet:
        """
        Import reference set from file

        Args:
            file_path: Path to file (YAML or JSON)

        Returns:
            CustomReferenceSet
        """
        file_path = Path(file_path)

        if file_path.suffix in [".yaml", ".yml"]:
            with open(file_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
        elif file_path.suffix == ".json":
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            raise ValueError(f"Unknown file format: {file_path.suffix}")

        return CustomReferenceSet.from_dict(data)

    def _load_templates(self) -> Dict[str, Dict[int, str]]:
        """Load statement templates for each domain"""
        return {
            "general": {
                1: "I would never consider purchasing this product",
                2: "I have little interest in this product",
                3: "I am somewhat interested in this product",
                4: "I am very interested in this product",
                5: "I would definitely purchase this product",
            },
            "healthcare": {
                1: "This product does not meet my health and safety requirements",
                2: "I have concerns about this product's effectiveness and safety",
                3: "This product seems acceptable but I need more medical information",
                4: "This product appears safe and effective for my needs",
                5: "This product fully meets my health needs and has my doctor's approval",
            },
            "financial": {
                1: "This service poses unacceptable financial risk to me",
                2: "I am not confident in this service's security and reliability",
                3: "This service seems reasonable but I need to verify credentials",
                4: "This service appears trustworthy and meets my financial needs",
                5: "This service has excellent reputation and fully meets my requirements",
            },
            "luxury": {
                1: "This product does not meet my standards for quality and prestige",
                2: "This product lacks the exclusivity and craftsmanship I expect",
                3: "This product shows promise but needs more refinement",
                4: "This product demonstrates excellent quality and prestige",
                5: "This product represents the pinnacle of luxury and craftsmanship",
            },
            "b2b_software": {
                1: "This software would not integrate with our existing systems",
                2: "I have concerns about this software's scalability and support",
                3: "This software seems viable but requires further technical evaluation",
                4: "This software meets our requirements and offers good ROI",
                5: "This software perfectly addresses our needs with excellent support",
            },
        }

    def _estimate_sentiment(self, text: str) -> float:
        """
        Estimate sentiment of statement text (-1 to +1)

        Simple lexicon-based approach for validation purposes.
        """
        positive_terms = {
            "definitely",
            "very",
            "excellent",
            "perfect",
            "great",
            "love",
            "best",
            "fully",
        }
        negative_terms = {
            "never",
            "not",
            "no",
            "little",
            "concerns",
            "unacceptable",
            "lacks",
        }

        text_lower = text.lower()
        words = text_lower.split()

        positive_count = sum(1 for word in words if word in positive_terms)
        negative_count = sum(1 for word in words if word in negative_terms)

        total_sentiment_words = positive_count + negative_count
        if total_sentiment_words == 0:
            return 0.0  # Neutral

        # Normalize to [-1, 1]
        sentiment = (positive_count - negative_count) / total_sentiment_words
        return sentiment

    def _calculate_statement_similarities(self, statements: List[str]) -> np.ndarray:
        """
        Calculate pairwise similarities between statements

        Simple word overlap for validation purposes.
        For production, would use embeddings.
        """
        n = len(statements)
        similarity_matrix = np.zeros((n, n))

        for i in range(n):
            words_i = set(statements[i].lower().split())
            for j in range(n):
                words_j = set(statements[j].lower().split())

                # Jaccard similarity
                intersection = len(words_i & words_j)
                union = len(words_i | words_j)

                similarity = intersection / union if union > 0 else 0.0
                similarity_matrix[i, j] = similarity

        return similarity_matrix

    def _check_domain_terms(self, text: str, domain: DomainCategory) -> bool:
        """Check if text contains domain-specific terminology"""
        domain_lexicons = {
            DomainCategory.HEALTHCARE: {
                "health",
                "safety",
                "medical",
                "doctor",
                "effective",
                "safe",
            },
            DomainCategory.FINANCIAL: {
                "financial",
                "risk",
                "security",
                "reliable",
                "trust",
                "service",
            },
            DomainCategory.LUXURY: {
                "quality",
                "prestige",
                "exclusive",
                "craftsmanship",
                "luxury",
            },
            DomainCategory.B2B_SOFTWARE: {
                "software",
                "integrate",
                "scalable",
                "support",
                "ROI",
                "technical",
            },
        }

        if domain not in domain_lexicons:
            return True  # No specific terms required

        lexicon = domain_lexicons[domain]
        text_lower = text.lower()

        return any(term in text_lower for term in lexicon)


# Example usage and testing
if __name__ == "__main__":
    print("Custom Reference Set Builder Testing")
    print("=" * 70)

    builder = CustomSetBuilder()

    # Test 1: Create from template
    print("\n1. Create Healthcare Domain Set from Template:")

    healthcare_set = builder.create_from_template(
        set_id="healthcare_001",
        name="Healthcare Products Reference Set",
        domain=DomainCategory.HEALTHCARE,
    )

    print(f"Set ID: {healthcare_set.set_id}")
    print(f"Domain: {healthcare_set.domain.value}")
    print(f"Statements: {len(healthcare_set.statements)}")
    for stmt in healthcare_set.statements:
        print(f"  Rating {stmt.rating}: {stmt.text[:60]}...")

    # Test 2: Validate set
    print("\n" + "=" * 70)
    print("2. Validate Healthcare Set:")

    validation = builder.validate_set(healthcare_set)
    print(validation.get_summary())

    # Test 3: Create custom set
    print("\n" + "=" * 70)
    print("3. Create Custom Luxury Goods Set:")

    luxury_statements = [
        "This product falls far below my expectations for luxury",
        "This product shows potential but lacks refinement",
        "This product meets basic luxury standards",
        "This product demonstrates exceptional quality and prestige",
        "This product represents the absolute pinnacle of luxury and exclusivity",
    ]

    luxury_set = builder.create_from_statements(
        set_id="luxury_001",
        name="Luxury Goods Reference Set",
        description="Custom set for high-end consumer products",
        domain=DomainCategory.LUXURY,
        statement_texts=luxury_statements,
    )

    validation = builder.validate_set(luxury_set, strict=True)
    print(validation.get_summary())

    # Test 4: Export and import
    print("\n" + "=" * 70)
    print("4. Export and Import Test:")

    export_path = Path("/tmp/test_luxury_set.yaml")
    builder.export_set(luxury_set, export_path, format="yaml")
    print(f"Exported to: {export_path}")

    imported_set = builder.import_set(export_path)
    print(f"Imported: {imported_set.name}")
    print(f"Statements preserved: {len(imported_set.statements) == 5}")

    print("\n" + "=" * 70)
    print("Custom set builder testing complete")
    print("\nKey Insights:")
    print("- Domain templates provide starting points for customization")
    print("- Validation ensures quality before integration")
    print("- Export/import enables sharing and version control")
    print("- Custom sets complement paper's 6 baseline sets")

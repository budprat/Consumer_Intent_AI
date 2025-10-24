# ABOUTME: Reference set-related Pydantic models for API
# ABOUTME: Defines schemas for reference statement sets and quality metrics

from pydantic import BaseModel, Field, field_validator
from typing import Optional, Dict, Any
from enum import Enum


class DomainCategory(str, Enum):
    """Domain categories for reference sets."""

    GENERAL = "general"
    HEALTHCARE = "healthcare"
    FINANCIAL = "financial"
    LUXURY = "luxury"
    B2B_SOFTWARE = "b2b_software"


class ReferenceSetRequest(BaseModel):
    """
    Request model for creating a custom reference statement set.

    Allows users to define domain-specific reference statements.
    """

    set_id: str = Field(
        ...,
        min_length=1,
        max_length=100,
        pattern="^[a-zA-Z0-9_-]+$",
        description="Unique identifier for the reference set",
        examples=["healthcare_set_1", "custom_luxury_v2"],
    )

    domain: DomainCategory = Field(
        ...,
        description="Domain category for the reference set",
    )

    statements: Dict[int, str] = Field(
        ...,
        description="Reference statements mapped to 1-5 scale",
    )

    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional metadata (author, version, notes)",
    )

    @field_validator("statements")
    @classmethod
    def validate_statements(cls, v: Dict[int, str]) -> Dict[int, str]:
        """
        Validate reference statements.

        Requirements:
        - Must have exactly 5 statements (one per scale point)
        - Keys must be 1, 2, 3, 4, 5
        - Each statement must be non-empty and <500 chars
        """
        if len(v) != 5:
            raise ValueError("Must provide exactly 5 reference statements")

        required_keys = {1, 2, 3, 4, 5}
        if set(v.keys()) != required_keys:
            raise ValueError(f"Keys must be {required_keys}, got {set(v.keys())}")

        for rating, statement in v.items():
            if not statement or not statement.strip():
                raise ValueError(f"Statement for rating {rating} cannot be empty")

            if len(statement) > 500:
                raise ValueError(
                    f"Statement for rating {rating} exceeds 500 characters"
                )

        return v

    class Config:
        json_schema_extra = {
            "example": {
                "set_id": "healthcare_set_1",
                "domain": "healthcare",
                "statements": {
                    1: "This product does not meet my health and safety requirements",
                    2: "This product meets some health requirements but has concerns",
                    3: "This product adequately meets my health needs",
                    4: "This product exceeds my health expectations",
                    5: "This product fully meets my health needs with doctor approval",
                },
                "metadata": {
                    "author": "Clinical Research Team",
                    "version": "1.0",
                    "validated": True,
                },
            }
        }


class ReferenceSetResponse(BaseModel):
    """Response model for reference set creation and retrieval."""

    set_id: str

    domain: DomainCategory

    statements: Dict[int, str]

    quality_metrics: Optional[Dict[str, float]] = Field(
        default=None,
        description="Quality metrics if analysis has been run",
    )

    metadata: Optional[Dict[str, Any]] = Field(default=None)

    created_at: Optional[str] = Field(default=None)

    class Config:
        json_schema_extra = {
            "example": {
                "set_id": "healthcare_set_1",
                "domain": "healthcare",
                "statements": {
                    1: "Does not meet health requirements",
                    2: "Meets some requirements with concerns",
                    3: "Adequately meets health needs",
                    4: "Exceeds health expectations",
                    5: "Fully meets needs with approval",
                },
                "quality_metrics": {
                    "discriminative_power": 0.85,
                    "consistency_score": 0.92,
                    "correlation_with_truth": 0.88,
                },
                "created_at": "2025-01-15T10:00:00Z",
            }
        }

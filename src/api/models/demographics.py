# ABOUTME: Demographic-related Pydantic models for API
# ABOUTME: Defines schemas for demographic profiles and cohort configuration

from pydantic import BaseModel, Field, field_validator
from typing import Optional, Dict, List
from enum import Enum


class Gender(str, Enum):
    """Gender categories (US Census 2020)."""

    MALE = "Male"
    FEMALE = "Female"
    NON_BINARY = "Non-binary"
    OTHER = "Other"
    PREFER_NOT_TO_SAY = "Prefer not to say"


class IncomeLevel(str, Enum):
    """Income level categories (US Census 2020)."""

    LESS_THAN_25K = "Less than $25,000"
    _25K_TO_50K = "$25,000 to $50,000"
    _50K_TO_75K = "$50,000 to $75,000"
    _75K_TO_100K = "$75,000 to $100,000"
    _100K_TO_150K = "$100,000 to $150,000"
    OVER_150K = "Over $150,000"


class Ethnicity(str, Enum):
    """Ethnicity categories (US Census 2020)."""

    WHITE = "White"
    BLACK = "Black or African American"
    ASIAN = "Asian"
    HISPANIC_LATINO = "Hispanic or Latino"
    NATIVE_AMERICAN = "Native American or Alaska Native"
    PACIFIC_ISLANDER = "Native Hawaiian or Pacific Islander"
    TWO_OR_MORE = "Two or more races"
    OTHER = "Other"


class DemographicProfileRequest(BaseModel):
    """
    Request model for a single demographic profile.

    Used for custom cohort specification.
    """

    age: int = Field(..., ge=18, le=120, description="Age in years (18+)")

    gender: Gender = Field(..., description="Gender identity")

    income_level: IncomeLevel = Field(..., description="Annual household income level")

    location_state: str = Field(
        ...,
        min_length=2,
        max_length=2,
        description="US state code (2-letter)",
        examples=["CA", "NY", "TX"],
    )

    location_region: str = Field(
        ...,
        description="US Census region",
        examples=["West", "Northeast", "South", "Midwest"],
    )

    ethnicity: Ethnicity = Field(..., description="Ethnicity/race")

    @field_validator("location_state")
    @classmethod
    def validate_state_code(cls, v: str) -> str:
        """Validate US state code."""
        v = v.upper()
        # List of valid US state codes
        valid_states = {
            "AL",
            "AK",
            "AZ",
            "AR",
            "CA",
            "CO",
            "CT",
            "DE",
            "FL",
            "GA",
            "HI",
            "ID",
            "IL",
            "IN",
            "IA",
            "KS",
            "KY",
            "LA",
            "ME",
            "MD",
            "MA",
            "MI",
            "MN",
            "MS",
            "MO",
            "MT",
            "NE",
            "NV",
            "NH",
            "NJ",
            "NM",
            "NY",
            "NC",
            "ND",
            "OH",
            "OK",
            "OR",
            "PA",
            "RI",
            "SC",
            "SD",
            "TN",
            "TX",
            "UT",
            "VT",
            "VA",
            "WA",
            "WV",
            "WI",
            "WY",
            "DC",
        }

        if v not in valid_states:
            raise ValueError(f"Invalid US state code: {v}")

        return v

    @field_validator("location_region")
    @classmethod
    def validate_region(cls, v: str) -> str:
        """Validate US Census region."""
        valid_regions = {"West", "Northeast", "South", "Midwest"}

        if v not in valid_regions:
            raise ValueError(f"Invalid region: {v}. Must be one of {valid_regions}")

        return v

    class Config:
        json_schema_extra = {
            "example": {
                "age": 35,
                "gender": "Female",
                "income_level": "$75,000 to $100,000",
                "location_state": "CA",
                "location_region": "West",
                "ethnicity": "Asian",
            }
        }


class DemographicCohortRequest(BaseModel):
    """
    Request model for custom demographic cohort specification.

    Allows specifying exact demographic distributions or individual profiles.
    """

    distribution: Optional[Dict[str, Dict[str, float]]] = Field(
        default=None,
        description="Custom demographic distributions (alternative to individual profiles)",
    )

    individual_profiles: Optional[List[DemographicProfileRequest]] = Field(
        default=None,
        description="List of individual demographic profiles (alternative to distribution)",
    )

    @field_validator("distribution")
    @classmethod
    def validate_distribution(cls, v: Optional[Dict]) -> Optional[Dict]:
        """Validate that all distributions sum to 1.0."""
        if v is None:
            return v

        for attribute, dist in v.items():
            total = sum(dist.values())
            if not (0.99 <= total <= 1.01):  # Allow small floating point errors
                raise ValueError(
                    f"Distribution for '{attribute}' must sum to 1.0 (got {total})"
                )

        return v

    @field_validator("individual_profiles")
    @classmethod
    def validate_profiles(cls, v: Optional[List]) -> Optional[List]:
        """Validate that profiles list is not empty if provided."""
        if v is not None and len(v) == 0:
            raise ValueError("individual_profiles cannot be empty list")

        return v

    class Config:
        json_schema_extra = {
            "example": {
                "distribution": {
                    "age_groups": {
                        "18-24": 0.15,
                        "25-34": 0.20,
                        "35-44": 0.25,
                        "45-54": 0.20,
                        "55-64": 0.15,
                        "65+": 0.05,
                    },
                    "gender": {"Male": 0.50, "Female": 0.48, "Non-binary": 0.02},
                }
            }
        }

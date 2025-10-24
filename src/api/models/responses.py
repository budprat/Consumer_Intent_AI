# ABOUTME: Standard error response models for API
# ABOUTME: Defines consistent error response schemas for validation and runtime errors

from pydantic import BaseModel, Field
from typing import Optional, List


class ErrorResponse(BaseModel):
    """
    Standard error response model.

    Used for general errors (4xx, 5xx).
    """

    detail: str = Field(..., description="Human-readable error message")

    error_code: Optional[str] = Field(
        default=None,
        description="Machine-readable error code for client handling",
        examples=["INVALID_API_KEY", "RATE_LIMIT_EXCEEDED", "SURVEY_NOT_FOUND"],
    )

    error_type: Optional[str] = Field(
        default=None,
        description="Error type/category",
        examples=["ValidationError", "AuthenticationError", "ResourceNotFound"],
    )

    timestamp: Optional[float] = Field(
        default=None,
        description="Error timestamp (Unix time)",
    )

    request_id: Optional[str] = Field(
        default=None,
        description="Request ID for tracking and support",
    )

    hint: Optional[str] = Field(
        default=None,
        description="Suggested resolution or next steps",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "detail": "Survey not found",
                "error_code": "SURVEY_NOT_FOUND",
                "error_type": "ResourceNotFound",
                "timestamp": 1705315200.0,
                "request_id": "req_789ghi",
                "hint": "Check the survey_id and ensure it exists",
            }
        }


class ValidationError(BaseModel):
    """Individual validation error detail."""

    loc: List[str] = Field(
        ...,
        description="Location of the error in the request (e.g., ['body', 'cohort_size'])",
    )

    msg: str = Field(
        ...,
        description="Error message",
    )

    type: str = Field(
        ...,
        description="Error type",
        examples=["value_error", "type_error", "missing"],
    )

    class Config:
        json_schema_extra = {
            "example": {
                "loc": ["body", "cohort_size"],
                "msg": "ensure this value is greater than or equal to 10",
                "type": "value_error.number.not_ge",
            }
        }


class ValidationErrorResponse(BaseModel):
    """
    Validation error response model.

    Used for request validation errors (422).
    """

    detail: str = Field(
        default="Request validation failed",
        description="General error message",
    )

    errors: List[ValidationError] = Field(
        ...,
        description="List of validation errors",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "detail": "Request validation failed",
                "errors": [
                    {
                        "loc": ["body", "cohort_size"],
                        "msg": "ensure this value is greater than or equal to 10",
                        "type": "value_error.number.not_ge",
                    },
                    {
                        "loc": ["body", "temperature"],
                        "msg": "ensure this value is less than or equal to 2.0",
                        "type": "value_error.number.not_le",
                    },
                ],
            }
        }

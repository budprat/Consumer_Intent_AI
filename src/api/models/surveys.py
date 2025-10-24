# ABOUTME: Survey-related Pydantic models for API requests and responses
# ABOUTME: Defines schemas for survey creation, execution, and results retrieval

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from enum import Enum
from datetime import datetime


class LLMModel(str, Enum):
    """Supported LLM models for survey execution."""

    GPT4O = "gpt-4o"
    GEMINI_2_FLASH = "gemini-2.0-flash"


class AveragingStrategy(str, Enum):
    """Multi-reference set averaging strategies."""

    UNIFORM = "uniform"
    WEIGHTED = "weighted"
    ADAPTIVE = "adaptive"
    PERFORMANCE_BASED = "performance"
    BEST_SUBSET = "best_subset"


class SamplingStrategy(str, Enum):
    """Demographic sampling strategies."""

    STRATIFIED = "stratified"
    QUOTA = "quota"
    CUSTOM = "custom"


class SurveyStatus(str, Enum):
    """Survey execution status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# ============================================================================
# Request Models
# ============================================================================


class SurveyCreateRequest(BaseModel):
    """
    Request model for creating a new survey configuration.

    This creates a survey definition that can be executed multiple times
    with different cohorts.
    """

    product_name: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Product or service name",
        examples=["Premium Wireless Headphones"],
    )

    product_description: str = Field(
        ...,
        min_length=10,
        max_length=5000,
        description="Detailed product description",
        examples=[
            "High-quality over-ear headphones with active noise cancellation, "
            "40-hour battery life, and premium leather cushions. Price: $299."
        ],
    )

    reference_set_ids: Optional[List[str]] = Field(
        default=None,
        description="Reference statement set IDs to use (defaults to all 6 sets from paper)",
        examples=[["set_1", "set_2", "set_3", "set_4", "set_5", "set_6"]],
    )

    averaging_strategy: AveragingStrategy = Field(
        default=AveragingStrategy.UNIFORM,
        description="Strategy for averaging across multiple reference sets",
    )

    temperature: float = Field(
        default=1.0,
        ge=0.1,
        le=2.0,
        description="Temperature parameter for distribution scaling (tested: 0.5, 1.0, 1.5)",
    )

    enable_demographics: bool = Field(
        default=True,
        description="Enable demographic conditioning (CRITICAL: improves correlation from 50% to 90%)",
    )

    enable_bias_detection: bool = Field(
        default=True,
        description="Enable demographic bias detection and reporting",
    )

    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional metadata for the survey",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "product_name": "Premium Wireless Headphones",
                "product_description": "High-quality over-ear headphones with active noise cancellation",
                "reference_set_ids": ["set_1", "set_2", "set_3"],
                "averaging_strategy": "adaptive",
                "temperature": 1.0,
                "enable_demographics": True,
                "enable_bias_detection": True,
            }
        }


class SurveyExecuteRequest(BaseModel):
    """
    Request model for executing a survey with a specific cohort.

    Generates synthetic consumer responses using SSR methodology.
    """

    llm_model: LLMModel = Field(
        default=LLMModel.GPT4O,
        description="LLM model to use for response generation",
    )

    cohort_size: int = Field(
        ...,
        ge=10,
        le=10000,
        description="Number of synthetic respondents (paper used 1000)",
        examples=[1000],
    )

    sampling_strategy: SamplingStrategy = Field(
        default=SamplingStrategy.STRATIFIED,
        description="Demographic sampling strategy",
    )

    custom_demographics: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Custom demographic distributions (for CUSTOM sampling)",
    )

    enable_test_retest: bool = Field(
        default=False,
        description="Enable test-retest reliability simulation (generates 2 independent samples)",
    )

    async_execution: bool = Field(
        default=True,
        description="Execute asynchronously (returns task_id for status polling)",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "llm_model": "gpt-4o",
                "cohort_size": 1000,
                "sampling_strategy": "stratified",
                "enable_test_retest": True,
                "async_execution": True,
            }
        }


# ============================================================================
# Response Models
# ============================================================================


class SurveyResponse(BaseModel):
    """Response model for survey creation."""

    survey_id: str = Field(..., description="Unique survey identifier")

    product_name: str

    created_at: datetime

    status: SurveyStatus = Field(default=SurveyStatus.PENDING)

    configuration: Dict[str, Any] = Field(..., description="Survey configuration")

    class Config:
        json_schema_extra = {
            "example": {
                "survey_id": "survey_123abc",
                "product_name": "Premium Wireless Headphones",
                "created_at": "2025-01-15T10:30:00Z",
                "status": "pending",
                "configuration": {
                    "averaging_strategy": "adaptive",
                    "temperature": 1.0,
                    "enable_demographics": True,
                },
            }
        }


class SurveyExecuteResponse(BaseModel):
    """Response model for survey execution."""

    task_id: str = Field(
        ..., description="Background task identifier for async execution"
    )

    survey_id: str

    status: SurveyStatus

    estimated_completion_time: Optional[int] = Field(
        default=None,
        description="Estimated seconds until completion",
    )

    status_url: str = Field(
        ...,
        description="URL to poll for task status",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "task_id": "task_456def",
                "survey_id": "survey_123abc",
                "status": "running",
                "estimated_completion_time": 300,
                "status_url": "/api/v1/surveys/survey_123abc/tasks/task_456def/status",
            }
        }


class SurveyStatusResponse(BaseModel):
    """Response model for task status polling."""

    task_id: str

    survey_id: str

    status: SurveyStatus

    progress: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Progress percentage (0.0-1.0)",
    )

    responses_generated: Optional[int] = Field(
        default=None,
        description="Number of synthetic responses generated so far",
    )

    error: Optional[str] = Field(
        default=None,
        description="Error message if status is FAILED",
    )

    results_url: Optional[str] = Field(
        default=None,
        description="URL to retrieve results (available when status is COMPLETED)",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "task_id": "task_456def",
                "survey_id": "survey_123abc",
                "status": "completed",
                "progress": 1.0,
                "responses_generated": 1000,
                "results_url": "/api/v1/surveys/survey_123abc/results",
            }
        }


class SurveyResultsResponse(BaseModel):
    """Response model for survey results and metrics."""

    survey_id: str

    task_id: str

    completed_at: datetime

    # Cohort information
    cohort_size: int

    demographic_distribution: Dict[str, Any] = Field(
        ...,
        description="Actual demographic distribution of generated cohort",
    )

    # SSR Distribution
    distribution: List[float] = Field(
        ...,
        description="Purchase intent distribution (5-point scale probabilities)",
    )

    # Metrics
    metrics: Dict[str, Any] = Field(
        ...,
        description="Evaluation metrics (if ground truth provided)",
    )

    # Quality Indicators
    quality: Dict[str, Any] = Field(
        ...,
        description="Quality indicators and bias detection results",
    )

    # Raw data (optional, controlled by query param)
    raw_responses: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Individual synthetic responses (optional)",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "survey_id": "survey_123abc",
                "task_id": "task_456def",
                "completed_at": "2025-01-15T10:35:00Z",
                "cohort_size": 1000,
                "demographic_distribution": {
                    "age_groups": {"18-24": 0.12, "25-34": 0.18},
                    "gender": {"Male": 0.49, "Female": 0.50},
                },
                "distribution": [0.15, 0.20, 0.30, 0.25, 0.10],
                "metrics": {
                    "ks_similarity": 0.88,
                    "pearson_correlation": 0.902,
                    "mae": 0.15,
                },
                "quality": {
                    "bias_score": 0.05,
                    "reliability_estimate": 0.90,
                },
            }
        }

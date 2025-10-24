# ABOUTME: Survey execution endpoints with async task processing
# ABOUTME: Handles survey creation, execution, status polling, and results retrieval

from fastapi import APIRouter, HTTPException, status, BackgroundTasks, Query
from typing import Dict, Optional
import uuid
import logging
from datetime import datetime

from ..models.surveys import (
    SurveyCreateRequest,
    SurveyResponse,
    SurveyExecuteRequest,
    SurveyExecuteResponse,
    SurveyStatusResponse,
    SurveyResultsResponse,
    SurveyStatus,
)
from ..models.responses import ErrorResponse

# Import SSR Executor Service

logger = logging.getLogger(__name__)

router = APIRouter()

# In-memory storage for demo (production should use Redis/database)
_surveys: Dict[str, dict] = {}
_tasks: Dict[str, dict] = {}
_results: Dict[str, dict] = {}


# ============================================================================
# Survey Listing
# ============================================================================


@router.get(
    "/surveys",
    response_model=list[SurveyResponse],
    responses={
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
    summary="List all surveys",
    description="""
    Retrieve a list of all surveys created.

    Returns basic information about each survey including status.
    """,
)
async def list_surveys() -> list[SurveyResponse]:
    """
    Get a list of all surveys.

    Returns:
        List of survey responses ordered by creation date (newest first)
    """
    surveys_list = []

    for survey_id, survey in _surveys.items():
        surveys_list.append(
            SurveyResponse(
                survey_id=survey_id,
                product_name=survey["product_name"],
                created_at=survey["created_at"],
                status=SurveyStatus(survey["status"]),
                configuration={
                    "reference_set_ids": survey["reference_set_ids"],
                    "averaging_strategy": survey["averaging_strategy"],
                    "temperature": survey["temperature"],
                    "enable_demographics": survey["enable_demographics"],
                    "enable_bias_detection": survey["enable_bias_detection"],
                },
            )
        )

    # Sort by creation date (newest first)
    surveys_list.sort(key=lambda s: s.created_at, reverse=True)

    logger.info(f"Listed {len(surveys_list)} surveys")

    return surveys_list


# ============================================================================
# Survey Detail
# ============================================================================


@router.get(
    "/surveys/{survey_id}",
    response_model=SurveyResponse,
    responses={
        404: {"model": ErrorResponse, "description": "Survey not found"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
    summary="Get survey details",
    description="""
    Retrieve detailed information about a specific survey.

    Returns survey configuration and current status.
    """,
)
async def get_survey(survey_id: str) -> SurveyResponse:
    """
    Get details for a specific survey.

    Args:
        survey_id: Unique survey identifier

    Returns:
        Survey response with full configuration

    Raises:
        HTTPException: If survey not found
    """
    if survey_id not in _surveys:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Survey {survey_id} not found",
        )

    survey = _surveys[survey_id]

    logger.info(f"Retrieved survey: {survey_id}")

    return SurveyResponse(
        survey_id=survey_id,
        product_name=survey["product_name"],
        created_at=survey["created_at"],
        status=SurveyStatus(survey["status"]),
        configuration={
            "reference_set_ids": survey["reference_set_ids"],
            "averaging_strategy": survey["averaging_strategy"],
            "temperature": survey["temperature"],
            "enable_demographics": survey["enable_demographics"],
            "enable_bias_detection": survey["enable_bias_detection"],
        },
    )


# ============================================================================
# Survey Creation
# ============================================================================


@router.post(
    "/surveys",
    response_model=SurveyResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        422: {"model": ErrorResponse, "description": "Validation error"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
    summary="Create a new survey",
    description="""
    Create a new survey configuration for a product or service.

    The survey can be executed multiple times with different cohorts.
    """,
)
async def create_survey(request: SurveyCreateRequest) -> SurveyResponse:
    """
    Create a new survey configuration.

    Args:
        request: Survey creation request with product details and configuration

    Returns:
        Survey response with generated survey_id
    """
    # Generate unique survey ID
    survey_id = f"survey_{uuid.uuid4().hex[:12]}"

    # Create survey record
    survey = {
        "survey_id": survey_id,
        "product_name": request.product_name,
        "product_description": request.product_description,
        "reference_set_ids": request.reference_set_ids or [
            f"set_{i}" for i in range(1, 7)
        ],
        "averaging_strategy": request.averaging_strategy.value,
        "temperature": request.temperature,
        "enable_demographics": request.enable_demographics,
        "enable_bias_detection": request.enable_bias_detection,
        "metadata": request.metadata or {},
        "created_at": datetime.utcnow(),
        "status": SurveyStatus.PENDING.value,
    }

    # Store survey
    _surveys[survey_id] = survey

    logger.info(
        f"Survey created: {survey_id}",
        extra={
            "survey_id": survey_id,
            "product_name": request.product_name,
            "averaging_strategy": request.averaging_strategy.value,
        },
    )

    return SurveyResponse(
        survey_id=survey_id,
        product_name=request.product_name,
        created_at=survey["created_at"],
        status=SurveyStatus.PENDING,
        configuration={
            "reference_set_ids": survey["reference_set_ids"],
            "averaging_strategy": survey["averaging_strategy"],
            "temperature": survey["temperature"],
            "enable_demographics": survey["enable_demographics"],
            "enable_bias_detection": survey["enable_bias_detection"],
        },
    )


# ============================================================================
# Survey Execution
# ============================================================================


@router.post(
    "/surveys/{survey_id}/execute",
    response_model=SurveyExecuteResponse,
    status_code=status.HTTP_202_ACCEPTED,
    responses={
        404: {"model": ErrorResponse, "description": "Survey not found"},
        422: {"model": ErrorResponse, "description": "Validation error"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
    summary="Execute a survey",
    description="""
    Execute a survey to generate synthetic consumer responses.

    Returns a task_id for polling execution status.
    """,
)
async def execute_survey(
    survey_id: str,
    request: SurveyExecuteRequest,
    background_tasks: BackgroundTasks,
) -> SurveyExecuteResponse:
    """
    Execute a survey with specified cohort configuration.

    Args:
        survey_id: Survey identifier
        request: Execution configuration
        background_tasks: FastAPI background tasks manager

    Returns:
        Task information for polling status

    Raises:
        HTTPException: If survey not found
    """
    # Check if survey exists
    if survey_id not in _surveys:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Survey not found: {survey_id}",
        )

    survey = _surveys[survey_id]

    # Generate task ID
    task_id = f"task_{uuid.uuid4().hex[:12]}"

    # Create task record
    task = {
        "task_id": task_id,
        "survey_id": survey_id,
        "status": SurveyStatus.PENDING.value,
        "progress": 0.0,
        "responses_generated": 0,
        "request": {
            "llm_model": request.llm_model.value,
            "cohort_size": request.cohort_size,
            "sampling_strategy": request.sampling_strategy.value,
            "custom_demographics": request.custom_demographics,
            "enable_test_retest": request.enable_test_retest,
        },
        "created_at": datetime.utcnow(),
        "started_at": None,
        "completed_at": None,
        "error": None,
    }

    _tasks[task_id] = task

    # Add to background tasks if async execution requested
    if request.async_execution:
        background_tasks.add_task(_execute_survey_background, survey_id, task_id)
    else:
        # Synchronous execution (for small cohorts)
        await _execute_survey_sync(survey_id, task_id)

    # Estimate completion time (rough: 1 second per 10 responses)
    estimated_time = int(request.cohort_size / 10)

    logger.info(
        f"Survey execution started: {survey_id}",
        extra={
            "survey_id": survey_id,
            "task_id": task_id,
            "cohort_size": request.cohort_size,
            "async": request.async_execution,
        },
    )

    return SurveyExecuteResponse(
        task_id=task_id,
        survey_id=survey_id,
        status=SurveyStatus.PENDING,
        estimated_completion_time=estimated_time,
        status_url=f"/api/v1/surveys/{survey_id}/tasks/{task_id}/status",
    )


# ============================================================================
# Task Status Polling
# ============================================================================


@router.get(
    "/surveys/{survey_id}/tasks/{task_id}/status",
    response_model=SurveyStatusResponse,
    responses={
        404: {"model": ErrorResponse, "description": "Task not found"},
    },
    summary="Get task status",
    description="""
    Poll the status of a survey execution task.

    Use this endpoint to check progress and get results URL when completed.
    """,
)
async def get_task_status(survey_id: str, task_id: str) -> SurveyStatusResponse:
    """
    Get the current status of a survey execution task.

    Args:
        survey_id: Survey identifier
        task_id: Task identifier

    Returns:
        Task status and progress information

    Raises:
        HTTPException: If task not found
    """
    if task_id not in _tasks:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task not found: {task_id}",
        )

    task = _tasks[task_id]

    # Verify task belongs to survey
    if task["survey_id"] != survey_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task {task_id} does not belong to survey {survey_id}",
        )

    results_url = None
    if task["status"] == SurveyStatus.COMPLETED.value:
        results_url = f"/api/v1/surveys/{survey_id}/results?task_id={task_id}"

    return SurveyStatusResponse(
        task_id=task_id,
        survey_id=survey_id,
        status=SurveyStatus(task["status"]),
        progress=task["progress"],
        responses_generated=task["responses_generated"],
        error=task["error"],
        results_url=results_url,
    )


# ============================================================================
# Results Retrieval
# ============================================================================


@router.get(
    "/surveys/{survey_id}/results",
    response_model=SurveyResultsResponse,
    responses={
        404: {"model": ErrorResponse, "description": "Survey or results not found"},
        400: {"model": ErrorResponse, "description": "Task not completed"},
    },
    summary="Get survey results",
    description="""
    Retrieve survey results including distribution, metrics, and quality indicators.

    Use `include_raw=true` to include individual synthetic responses (large payload).
    """,
)
async def get_survey_results(
    survey_id: str,
    task_id: Optional[str] = Query(
        default=None,
        description="Specific task ID (returns latest if not specified)",
    ),
    include_raw: bool = Query(
        default=False,
        description="Include raw individual responses",
    ),
) -> SurveyResultsResponse:
    """
    Get survey results and metrics.

    Args:
        survey_id: Survey identifier
        task_id: Optional task ID (latest if not specified)
        include_raw: Whether to include raw responses

    Returns:
        Survey results with distribution, metrics, and quality indicators

    Raises:
        HTTPException: If survey/results not found or task not completed
    """
    if survey_id not in _surveys:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Survey not found: {survey_id}",
        )

    # If task_id not specified, find latest completed task
    if not task_id:
        completed_tasks = [
            t
            for t in _tasks.values()
            if t["survey_id"] == survey_id
            and t["status"] == SurveyStatus.COMPLETED.value
        ]

        if not completed_tasks:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No completed tasks found for survey {survey_id}",
            )

        # Get most recent
        task = max(completed_tasks, key=lambda t: t["completed_at"])
        task_id = task["task_id"]
    else:
        if task_id not in _tasks:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Task not found: {task_id}",
            )

        task = _tasks[task_id]

        if task["status"] != SurveyStatus.COMPLETED.value:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Task not completed. Current status: {task['status']}",
            )

    # Get results
    if task_id not in _results:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Results not found for task {task_id}",
        )

    results = _results[task_id]

    # Build response
    response = SurveyResultsResponse(
        survey_id=survey_id,
        task_id=task_id,
        completed_at=task["completed_at"],
        cohort_size=results["cohort_size"],
        demographic_distribution=results["demographic_distribution"],
        distribution=results["distribution"],
        metrics=results["metrics"],
        quality=results["quality"],
        raw_responses=results["raw_responses"] if include_raw else None,
    )

    logger.info(
        f"Results retrieved: {survey_id}",
        extra={
            "survey_id": survey_id,
            "task_id": task_id,
            "include_raw": include_raw,
        },
    )

    return response


# ============================================================================
# Background Task Execution
# ============================================================================


def _execute_survey_background(survey_id: str, task_id: str) -> None:
    """
    Background task for survey execution using SSR Executor.

    Args:
        survey_id: Survey identifier
        task_id: Task identifier
    """
    try:
        # Update task status
        _tasks[task_id]["status"] = SurveyStatus.RUNNING.value
        _tasks[task_id]["started_at"] = datetime.utcnow()

        survey = _surveys[survey_id]
        task = _tasks[task_id]
        request = task["request"]

        logger.info(f"Executing survey task: {task_id}")

        # Initialize SSR Executor
        from src.services.ssr_executor import SSRExecutor
        from src.api.config import settings
        executor = SSRExecutor(api_key=settings.OPENAI_API_KEY)

        # Extract demographic filters from survey metadata
        demographic_filters = None
        if survey.get("metadata") and survey["metadata"].get("demographic_filters"):
            demographic_filters = survey["metadata"]["demographic_filters"]
            logger.info(f"Extracted demographic filters from metadata: {demographic_filters}")
        else:
            logger.info("No demographic filters found in survey metadata")

        # Progress callback to update task status
        def progress_callback(progress_pct: int, status_msg: str):
            _tasks[task_id]["progress"] = progress_pct / 100.0
            _tasks[task_id]["responses_generated"] = int(
                (progress_pct / 100.0) * request["cohort_size"]
            )
            logger.info(f"Task {task_id} progress: {progress_pct}% - {status_msg}")

        # Execute survey with SSR
        logger.info(f"Executing survey with demographics_enabled={survey['enable_demographics']}, filters={demographic_filters}")
        result = executor.execute_survey(
            survey_id=survey_id,
            product_name=survey["product_name"],
            product_description=survey["product_description"],
            llm_model=request["llm_model"],
            temperature=survey["temperature"],
            enable_demographics=survey["enable_demographics"],
            demographic_filters=demographic_filters,
            consumer_count=min(request["cohort_size"], 10),  # Limit to 10 for demo
            progress_callback=progress_callback,
        )

        # Build demographic distribution from consumer results
        demographic_dist = {
            "age_groups": {},
            "gender": {},
            "income": {},
        }
        for consumer_result in result.consumer_results:
            demo = consumer_result.consumer_demographics

            # Age groups
            age = demo["age"]
            if age < 30:
                age_group = "18-29"
            elif age < 45:
                age_group = "30-44"
            elif age < 60:
                age_group = "45-59"
            else:
                age_group = "60+"
            demographic_dist["age_groups"][age_group] = (
                demographic_dist["age_groups"].get(age_group, 0) + 1
            )

            # Gender
            gender = demo["gender"]
            demographic_dist["gender"][gender] = (
                demographic_dist["gender"].get(gender, 0) + 1
            )

            # Income
            income = demo["income"]
            demographic_dist["income"][income] = (
                demographic_dist["income"].get(income, 0) + 1
            )

        # Normalize to proportions
        total_consumers = len(result.consumer_results)
        for category in demographic_dist:
            for key in demographic_dist[category]:
                demographic_dist[category][key] /= total_consumers

        # Build raw responses for detailed view
        raw_responses = [
            {
                "consumer_id": cr.consumer_id,
                "demographics": cr.consumer_demographics,
                "response_text": cr.response_text,
                "rating": cr.rating,
                "confidence": cr.confidence,
                "distribution": cr.distribution,
            }
            for cr in result.consumer_results
        ]

        # Store results
        _results[task_id] = {
            "cohort_size": total_consumers,
            "demographic_distribution": demographic_dist,
            "distribution": result.distribution,
            "metrics": {
                "mean_rating": result.mean_rating,
                "std_rating": result.std_rating,
                "confidence": result.confidence,
            },
            "quality": {
                "execution_time_seconds": result.execution_time_seconds,
                "llm_model_used": result.llm_model_used,
                "demographic_representation": "good" if survey["enable_demographics"] else "generic",
            },
            "raw_responses": raw_responses,
        }

        # Mark as completed
        _tasks[task_id]["status"] = SurveyStatus.COMPLETED.value
        _tasks[task_id]["completed_at"] = datetime.utcnow()
        _tasks[task_id]["progress"] = 1.0

        logger.info(
            f"Survey task completed: {task_id} - Mean rating: {result.mean_rating:.2f}"
        )

    except Exception as e:
        logger.error(f"Survey task failed: {task_id}", exc_info=True)
        _tasks[task_id]["status"] = SurveyStatus.FAILED.value
        _tasks[task_id]["error"] = str(e)
        _tasks[task_id]["completed_at"] = datetime.utcnow()


async def _execute_survey_sync(survey_id: str, task_id: str) -> None:
    """
    Synchronous survey execution (for small cohorts).

    Args:
        survey_id: Survey identifier
        task_id: Task identifier
    """
    # For sync execution, we can call the same background function
    # but in a non-blocking manner (still async but wait for completion)
    _execute_survey_background(survey_id, task_id)

# Force reload

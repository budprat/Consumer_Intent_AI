# API Reference

**Human Purchase Intent SSR - Complete REST API Documentation**

**Base URL**: `http://localhost:8000` (development) or `https://your-domain.com` (production)

**API Version**: `v1`

---

## Table of Contents

1. [Authentication](#1-authentication)
2. [Survey Endpoints](#2-survey-endpoints)
3. [SSR Endpoints](#3-ssr-endpoints)
4. [Evaluation Endpoints](#4-evaluation-endpoints)
5. [Task Management](#5-task-management)
6. [Health & Monitoring](#6-health--monitoring)
7. [Data Models](#7-data-models)
8. [Error Codes](#8-error-codes)
9. [Rate Limits](#9-rate-limits)
10. [Examples](#10-examples)

---

## API Versioning

### Current Version

**Version**: `v1.0.0` (January 2025)
**Status**: Beta - API is functional but may have breaking changes before stable release
**Stability**: Use for development and testing; production use at your own risk

### Version History

| Version | Release Date | Status | Notes |
|---------|-------------|--------|-------|
| **v1.0.0** | January 2025 | Beta | Initial release with core SSR endpoints |

### Versioning Policy

**Semantic Versioning**: API follows [SemVer](https://semver.org/) (MAJOR.MINOR.PATCH)

- **MAJOR**: Breaking changes (e.g., endpoint removal, parameter changes)
- **MINOR**: New features, backwards-compatible
- **PATCH**: Bug fixes, no API changes

**URL Versioning**: Version specified in URL path (`/api/v1/...`)

### Deprecation Policy

**Notice Period**: 6 months minimum for breaking changes

**Deprecation Process**:
1. **Announcement**: Breaking changes announced in release notes
2. **Warning Period**: 6 months with `Deprecation` header in responses
3. **Migration Guide**: Provided with alternative endpoints/parameters
4. **Sunset**: Old version removed after notice period

**Example Deprecation Header**:
```http
Deprecation: Sun, 01 Jul 2025 00:00:00 GMT
Sunset: Sun, 01 Jan 2026 00:00:00 GMT
Link: </api/v2/surveys/create>; rel="alternate"
```

### Compatibility Guarantees

**Within v1.x.x**:
- ✅ Existing endpoints remain functional
- ✅ Required parameters unchanged
- ✅ Response schemas backwards-compatible
- ⚠️ Optional parameters may be added
- ⚠️ New fields may be added to responses

**Beta Disclaimer**:
During beta (v1.0.0), breaking changes MAY occur with shorter notice (1 month minimum). Production users should pin to specific minor versions and test updates thoroughly.

### Migration Guide

When v2.0.0 is released, a detailed migration guide will be provided including:
- Breaking changes summary
- Endpoint mapping (v1 → v2)
- Code examples for migration
- Automated migration tools

### Staying Updated

**Release Notes**: Check `/api/v1/version` endpoint for current version
**Changelog**: See project CHANGELOG.md for detailed changes
**Notifications**: Subscribe to API updates at your-domain.com/api/updates

### Version Information Endpoint

**GET `/api/v1/version`**

Returns current API version and status:

```json
{
  "version": "1.0.0",
  "api_version": "v1",
  "status": "beta",
  "deprecation_notices": [],
  "sunset_date": null,
  "latest_version": "1.0.0"
}
```

---

## 1. Authentication

### API Keys

**Currently**: No authentication required for local development

**Production**: Set up API key authentication:

```bash
# Add to request headers
Authorization: Bearer YOUR_API_KEY
```

**Future implementation**:
- OAuth 2.0 support
- JWT token-based authentication
- Role-based access control (RBAC)

---

## 2. Survey Endpoints

### POST `/api/v1/surveys/create`

Create a new survey for SSR evaluation.

**Request Body**:

```json
{
  "product_name": "string (required, min: 3, max: 200)",
  "product_description": "string (required, min: 50, max: 2000)",
  "target_demographics": {
    "age_range": [25, 45],
    "income_range": [50000, 150000],
    "gender": "all | Male | Female | Non-binary",
    "regions": ["West", "Northeast"],
    "ethnicities": ["Asian", "White"]
  },
  "cohort_size": 200,
  "metadata": {
    "research_objectives": ["Purchase intent", "Price sensitivity"],
    "survey_date": "2024-Q4"
  }
}
```

**Response** (`201 Created`):

```json
{
  "survey_id": "550e8400-e29b-41d4-a716-446655440000",
  "product_name": "Premium Wireless Headphones",
  "product_description": "High-performance wireless headphones...",
  "target_demographics": {...},
  "cohort_size": 200,
  "status": "created",
  "created_at": "2025-01-15T10:30:00Z"
}
```

**Error Responses**:

```json
// 400 Bad Request
{
  "error": "Validation Error",
  "message": "Product description must be at least 50 characters",
  "field": "product_description"
}

// 500 Internal Server Error
{
  "error": "Server Error",
  "message": "Failed to create survey",
  "details": "Database connection error"
}
```

**Example**:

```bash
curl -X POST "http://localhost:8000/api/v1/surveys/create" \
  -H "Content-Type: application/json" \
  -d '{
    "product_name": "Eco Water Bottle",
    "product_description": "Insulated stainless steel water bottle, BPA-free, keeps drinks cold for 24h or hot for 12h. Available in 6 colors, dishwasher safe.",
    "cohort_size": 200,
    "target_demographics": {
      "age_range": [25, 45],
      "income_range": [50000, 120000]
    }
  }'
```

---

### GET `/api/v1/surveys/{survey_id}`

Retrieve survey details.

**Path Parameters**:
- `survey_id` (UUID, required): Survey identifier

**Response** (`200 OK`):

```json
{
  "survey_id": "550e8400-e29b-41d4-a716-446655440000",
  "product_name": "Eco Water Bottle",
  "product_description": "Insulated stainless steel...",
  "target_demographics": {...},
  "cohort_size": 200,
  "status": "created",
  "created_at": "2025-01-15T10:30:00Z",
  "updated_at": "2025-01-15T10:30:00Z"
}
```

**Error Responses**:

```json
// 404 Not Found
{
  "error": "Survey Not Found",
  "message": "Survey with ID 550e8400-... does not exist",
  "survey_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

---

### GET `/api/v1/surveys`

List all surveys with pagination.

**Query Parameters**:
- `limit` (integer, default: 20, max: 100): Number of surveys per page
- `offset` (integer, default: 0): Pagination offset
- `status` (string, optional): Filter by status (`created`, `processing`, `completed`)
- `sort_by` (string, default: `created_at`): Sort field
- `order` (string, default: `desc`): Sort order (`asc` or `desc`)

**Response** (`200 OK`):

```json
{
  "surveys": [
    {
      "survey_id": "550e8400-...",
      "product_name": "Product A",
      "status": "completed",
      "created_at": "2025-01-15T10:30:00Z"
    },
    {
      "survey_id": "660e8400-...",
      "product_name": "Product B",
      "status": "processing",
      "created_at": "2025-01-14T09:20:00Z"
    }
  ],
  "total": 57,
  "limit": 20,
  "offset": 0,
  "has_more": true
}
```

---

### DELETE `/api/v1/surveys/{survey_id}`

Delete a survey.

**Path Parameters**:
- `survey_id` (UUID, required): Survey identifier

**Response** (`204 No Content`): Empty body

**Error Responses**:

```json
// 404 Not Found
{
  "error": "Survey Not Found",
  "survey_id": "550e8400-..."
}

// 409 Conflict
{
  "error": "Survey In Use",
  "message": "Cannot delete survey with active SSR evaluations",
  "survey_id": "550e8400-..."
}
```

---

## 3. SSR Endpoints

### POST `/api/v1/ssr/run`

Run SSR evaluation for a survey (async processing).

**Request Body**:

```json
{
  "survey_id": "550e8400-e29b-41d4-a716-446655440000",
  "llm_model": "gpt-4o | gemini-2.0-flash",
  "temperature": 1.0,
  "enable_demographics": true,
  "averaging_strategy": "adaptive | uniform | weighted | performance_based | best_subset",
  "n_samples_per_consumer": 1,
  "cache_embeddings": true
}
```

**Field Descriptions**:

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `survey_id` | UUID | Yes | - | Survey to evaluate |
| `llm_model` | string | No | `"gpt-4o"` | LLM model (paper tested: gpt-4o, gemini-2.0-flash) |
| `temperature` | float | No | `1.0` | LLM temperature (paper tested: 0.5, 1.5) |
| `enable_demographics` | boolean | No | `true` | Enable demographic conditioning (+40% ρ improvement!) |
| `averaging_strategy` | string | No | `"adaptive"` | Multi-reference averaging method |
| `n_samples_per_consumer` | integer | No | `1` | Text samples per synthetic consumer |
| `cache_embeddings` | boolean | No | `true` | Cache embeddings for performance |

**Response** (`202 Accepted`):

```json
{
  "task_id": "770e8400-e29b-41d4-a716-446655440000",
  "survey_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "processing",
  "message": "SSR evaluation started",
  "started_at": "2025-01-15T10:35:00Z",
  "estimated_completion": "2025-01-15T10:40:00Z"
}
```

**Error Responses**:

```json
// 404 Not Found
{
  "error": "Survey Not Found",
  "survey_id": "550e8400-..."
}

// 400 Bad Request
{
  "error": "Invalid Parameters",
  "message": "temperature must be between 0.0 and 2.0",
  "field": "temperature",
  "value": 2.5
}

// 429 Too Many Requests
{
  "error": "Rate Limit Exceeded",
  "message": "Maximum concurrent SSR evaluations exceeded",
  "retry_after": 60
}
```

**Example**:

```bash
curl -X POST "http://localhost:8000/api/v1/ssr/run" \
  -H "Content-Type: application/json" \
  -d '{
    "survey_id": "550e8400-e29b-41d4-a716-446655440000",
    "llm_model": "gpt-4o",
    "temperature": 1.0,
    "enable_demographics": true,
    "averaging_strategy": "adaptive"
  }'
```

---

### POST `/api/v1/ssr/run-single`

Generate single SSR rating (synchronous).

**Request Body**:

```json
{
  "product_description": "string (required)",
  "demographic_profile": {
    "age": 32,
    "gender": "Female",
    "income": 85000,
    "location_state": "California",
    "location_region": "West",
    "ethnicity": "Asian"
  },
  "llm_model": "gpt-4o",
  "temperature": 1.0,
  "averaging_strategy": "adaptive"
}
```

**Response** (`200 OK`):

```json
{
  "rating": 4,
  "confidence": 0.87,
  "elicited_text": "As a 32-year-old professional living in California with a focus on health and wellness, I would be very interested in this product. The premium features align well with my lifestyle...",
  "embeddings_used": 6,
  "reference_set_ratings": [4, 4, 5, 4, 4, 4],
  "processing_time_ms": 1250
}
```

**Example**:

```bash
curl -X POST "http://localhost:8000/api/v1/ssr/run-single" \
  -H "Content-Type: application/json" \
  -d '{
    "product_description": "Premium organic protein bars with 20g protein, low sugar, gluten-free",
    "demographic_profile": {
      "age": 32,
      "gender": "Female",
      "income": 85000,
      "location_state": "California",
      "location_region": "West",
      "ethnicity": "Asian"
    }
  }'
```

---

## 4. Evaluation Endpoints

### POST `/api/v1/evaluation/metrics`

Compute evaluation metrics (KS similarity, correlation attainment).

**Request Body**:

```json
{
  "synthetic_distribution": [0.04, 0.15, 0.32, 0.36, 0.13],
  "human_distribution": [0.05, 0.14, 0.33, 0.35, 0.13],
  "human_retest_correlation": 0.85
}
```

**Field Descriptions**:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `synthetic_distribution` | float[] | Yes | SSR distribution P(rating=k), length 5, sums to 1.0 |
| `human_distribution` | float[] | Yes | Human response distribution, length 5, sums to 1.0 |
| `human_retest_correlation` | float | No | Human test-retest R^xx (for ρ calculation) |

**Response** (`200 OK`):

```json
{
  "ks_similarity": 0.87,
  "ks_distance": 0.13,
  "pearson_correlation": 0.92,
  "correlation_attainment": 0.91,
  "meets_paper_target_ks": true,
  "meets_paper_target_rho": true,
  "paper_targets": {
    "ks_similarity": 0.85,
    "correlation_attainment": 0.90
  },
  "interpretation": {
    "ks_similarity": "Distributions are very similar (target: ≥0.85)",
    "correlation_attainment": "Achieves 91% of human test-retest reliability (target: ≥90%)"
  }
}
```

**Error Responses**:

```json
// 400 Bad Request
{
  "error": "Validation Error",
  "message": "synthetic_distribution must sum to 1.0",
  "field": "synthetic_distribution",
  "sum": 0.99
}
```

**Example**:

```bash
curl -X POST "http://localhost:8000/api/v1/evaluation/metrics" \
  -H "Content-Type: application/json" \
  -d '{
    "synthetic_distribution": [0.04, 0.15, 0.32, 0.36, 0.13],
    "human_distribution": [0.05, 0.14, 0.33, 0.35, 0.13],
    "human_retest_correlation": 0.85
  }'
```

---

### POST `/api/v1/evaluation/test-retest`

Simulate test-retest reliability.

**Request Body**:

```json
{
  "cohort_ratings": [4, 3, 5, 4, 3, 4, 5, ...],
  "n_simulations": 2000,
  "split_ratio": 0.5
}
```

**Response** (`200 OK`):

```json
{
  "mean_correlation": 0.84,
  "std_correlation": 0.05,
  "median_correlation": 0.85,
  "correlations_distribution": {
    "min": 0.71,
    "q1": 0.81,
    "q2": 0.85,
    "q3": 0.88,
    "max": 0.94
  },
  "n_simulations": 2000,
  "split_ratio": 0.5,
  "paper_baseline": 0.85,
  "matches_paper_baseline": true
}
```

---

## 5. Task Management

### GET `/api/v1/tasks/{task_id}`

Get task status and results (for async operations).

**Path Parameters**:
- `task_id` (UUID, required): Task identifier from `/ssr/run`

**Response - Processing** (`200 OK`):

```json
{
  "task_id": "770e8400-e29b-41d4-a716-446655440000",
  "status": "processing",
  "progress": 0.45,
  "progress_details": {
    "completed_consumers": 90,
    "total_consumers": 200,
    "estimated_completion": "2025-01-15T10:38:00Z"
  },
  "started_at": "2025-01-15T10:35:00Z"
}
```

**Response - Completed** (`200 OK`):

```json
{
  "task_id": "770e8400-e29b-41d4-a716-446655440000",
  "status": "completed",
  "progress": 1.0,
  "results": {
    "survey_id": "550e8400-e29b-41d4-a716-446655440000",
    "distribution": [0.035, 0.120, 0.315, 0.380, 0.150],
    "mean_rating": 3.49,
    "std_rating": 1.08,
    "cohort_size": 200,
    "llm_model": "gpt-4o",
    "temperature": 1.0,
    "demographics_enabled": true,
    "averaging_strategy": "adaptive",
    "processing_time_seconds": 287.5
  },
  "started_at": "2025-01-15T10:35:00Z",
  "completed_at": "2025-01-15T10:39:47Z"
}
```

**Response - Failed** (`200 OK`):

```json
{
  "task_id": "770e8400-e29b-41d4-a716-446655440000",
  "status": "failed",
  "error": {
    "type": "RateLimitError",
    "message": "OpenAI API rate limit exceeded",
    "details": "Retry after 60 seconds",
    "retry_after": 60
  },
  "started_at": "2025-01-15T10:35:00Z",
  "failed_at": "2025-01-15T10:37:23Z"
}
```

**Error Responses**:

```json
// 404 Not Found
{
  "error": "Task Not Found",
  "task_id": "770e8400-..."
}
```

**Polling Example**:

```bash
# Start SSR evaluation
TASK_ID=$(curl -X POST "http://localhost:8000/api/v1/ssr/run" \
  -H "Content-Type: application/json" \
  -d '{"survey_id": "550e8400-...", ...}' \
  | jq -r '.task_id')

# Poll for completion
while true; do
  STATUS=$(curl "http://localhost:8000/api/v1/tasks/$TASK_ID" | jq -r '.status')
  if [ "$STATUS" = "completed" ] || [ "$STATUS" = "failed" ]; then
    break
  fi
  sleep 5
done

# Get results
curl "http://localhost:8000/api/v1/tasks/$TASK_ID"
```

---

### GET `/api/v1/tasks`

List all tasks with filtering.

**Query Parameters**:
- `status` (string, optional): Filter by status
- `survey_id` (UUID, optional): Filter by survey
- `limit` (integer, default: 20): Tasks per page
- `offset` (integer, default: 0): Pagination offset

**Response** (`200 OK`):

```json
{
  "tasks": [
    {
      "task_id": "770e8400-...",
      "survey_id": "550e8400-...",
      "status": "completed",
      "progress": 1.0,
      "started_at": "2025-01-15T10:35:00Z",
      "completed_at": "2025-01-15T10:39:47Z"
    },
    {
      "task_id": "880e8400-...",
      "survey_id": "660e8400-...",
      "status": "processing",
      "progress": 0.65,
      "started_at": "2025-01-15T10:40:00Z"
    }
  ],
  "total": 15,
  "limit": 20,
  "offset": 0
}
```

---

### DELETE `/api/v1/tasks/{task_id}`

Cancel a running task.

**Path Parameters**:
- `task_id` (UUID, required): Task identifier

**Response** (`204 No Content`): Empty body

**Error Responses**:

```json
// 404 Not Found
{
  "error": "Task Not Found",
  "task_id": "770e8400-..."
}

// 409 Conflict
{
  "error": "Task Already Completed",
  "message": "Cannot cancel completed task",
  "task_id": "770e8400-...",
  "status": "completed"
}
```

---

## 6. Health & Monitoring

### GET `/health`

Health check endpoint.

**Response** (`200 OK`):

```json
{
  "status": "healthy",
  "version": "1.0.0",
  "timestamp": "2025-01-15T10:45:00Z",
  "services": {
    "database": "healthy",
    "redis": "healthy",
    "openai_api": "healthy",
    "google_api": "healthy"
  }
}
```

**Unhealthy Response** (`503 Service Unavailable`):

```json
{
  "status": "unhealthy",
  "version": "1.0.0",
  "timestamp": "2025-01-15T10:45:00Z",
  "services": {
    "database": "healthy",
    "redis": "unhealthy",
    "openai_api": "healthy",
    "google_api": "healthy"
  },
  "errors": [
    {
      "service": "redis",
      "message": "Connection timeout",
      "since": "2025-01-15T10:44:30Z"
    }
  ]
}
```

---

### GET `/metrics`

Prometheus-compatible metrics endpoint.

**Response** (`200 OK`, `text/plain`):

```
# HELP ssr_requests_total Total number of SSR evaluation requests
# TYPE ssr_requests_total counter
ssr_requests_total{model="gpt-4o",status="success"} 1234
ssr_requests_total{model="gpt-4o",status="failed"} 5
ssr_requests_total{model="gemini-2.0-flash",status="success"} 789

# HELP ssr_processing_time_seconds SSR processing time in seconds
# TYPE ssr_processing_time_seconds histogram
ssr_processing_time_seconds_bucket{model="gpt-4o",le="10"} 45
ssr_processing_time_seconds_bucket{model="gpt-4o",le="30"} 120
ssr_processing_time_seconds_bucket{model="gpt-4o",le="60"} 200
ssr_processing_time_seconds_sum{model="gpt-4o"} 5432.1
ssr_processing_time_seconds_count{model="gpt-4o"} 250

# HELP ssr_ks_similarity SSR KS similarity scores
# TYPE ssr_ks_similarity gauge
ssr_ks_similarity{model="gpt-4o"} 0.87

# HELP ssr_correlation_attainment SSR correlation attainment
# TYPE ssr_correlation_attainment gauge
ssr_correlation_attainment{model="gpt-4o"} 0.91
```

---

## 7. Data Models

### Survey

```typescript
interface Survey {
  survey_id: string;           // UUID
  product_name: string;        // 3-200 characters
  product_description: string; // 50-2000 characters
  target_demographics?: {
    age_range?: [number, number];
    income_range?: [number, number];
    gender?: "all" | "Male" | "Female" | "Non-binary";
    regions?: string[];
    ethnicities?: string[];
  };
  cohort_size: number;         // 50-1000
  status: "created" | "processing" | "completed" | "failed";
  metadata?: Record<string, any>;
  created_at: string;          // ISO 8601
  updated_at: string;          // ISO 8601
}
```

### DemographicProfile

```typescript
interface DemographicProfile {
  age: number;                 // 18-75
  gender: "Male" | "Female" | "Non-binary" | "Prefer not to say";
  income: number;              // Annual income in USD
  location_state: string;      // U.S. state
  location_region: string;     // Geographic region
  ethnicity: string;           // Ethnicity category
  education?: string;          // Optional
  household_size?: number;     // Optional
  urban_rural?: string;        // Optional
}
```

### SSRResult

```typescript
interface SSRResult {
  rating: number;              // 1-5
  confidence: number;          // 0.0-1.0
  elicited_text: string;       // LLM-generated text
  embeddings_used: number;     // Number of embeddings
  reference_set_ratings: number[];  // Ratings from each reference set
  processing_time_ms: number;
}
```

### Distribution

```typescript
interface Distribution {
  distribution: number[];      // [P(1), P(2), P(3), P(4), P(5)], sums to 1.0
  mean_rating: number;         // Mean of distribution
  std_rating: number;          // Standard deviation
  cohort_size: number;         // Number of synthetic consumers
  llm_model: string;           // LLM used
  temperature: number;         // Temperature setting
  demographics_enabled: boolean;
  averaging_strategy: string;
}
```

---

## 8. Error Codes

### HTTP Status Codes

| Code | Status | Description |
|------|--------|-------------|
| `200` | OK | Request successful |
| `201` | Created | Resource created successfully |
| `202` | Accepted | Async operation started |
| `204` | No Content | Successful deletion |
| `400` | Bad Request | Invalid request parameters |
| `404` | Not Found | Resource not found |
| `409` | Conflict | Resource conflict (e.g., duplicate) |
| `422` | Unprocessable Entity | Validation error |
| `429` | Too Many Requests | Rate limit exceeded |
| `500` | Internal Server Error | Server error |
| `503` | Service Unavailable | Service temporarily unavailable |

### Error Response Format

```json
{
  "error": "Error Type",
  "message": "Human-readable error message",
  "field": "problematic_field",  // Optional
  "value": "invalid_value",      // Optional
  "details": "Additional context" // Optional
}
```

### Common Error Types

| Error Type | Description | HTTP Code |
|------------|-------------|-----------|
| `ValidationError` | Invalid request parameters | 400 |
| `SurveyNotFound` | Survey ID not found | 404 |
| `TaskNotFound` | Task ID not found | 404 |
| `RateLimitExceeded` | Too many requests | 429 |
| `LLMAPIError` | LLM API error (OpenAI, Google) | 500 |
| `InsufficientCredits` | API credits exhausted | 402 |
| `ServerError` | Internal server error | 500 |

---

## 9. Rate Limits

### Default Limits

| Endpoint | Limit | Window |
|----------|-------|--------|
| `/api/v1/surveys/*` | 100 req/min | Per IP |
| `/api/v1/ssr/run` | 10 req/min | Per IP |
| `/api/v1/ssr/run-single` | 30 req/min | Per IP |
| `/api/v1/evaluation/*` | 60 req/min | Per IP |
| `/api/v1/tasks/*` | 120 req/min | Per IP |

### Rate Limit Headers

Responses include rate limit information:

```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 85
X-RateLimit-Reset: 1705318800
```

### Rate Limit Exceeded Response

```json
{
  "error": "Rate Limit Exceeded",
  "message": "Maximum of 10 requests per minute exceeded",
  "limit": 10,
  "window_seconds": 60,
  "retry_after": 45
}
```

---

## 10. Examples

### Complete Workflow Example

```python
import requests
import time

BASE_URL = "http://localhost:8000"

# 1. Create survey
survey_response = requests.post(
    f"{BASE_URL}/api/v1/surveys/create",
    json={
        "product_name": "Eco Water Bottle",
        "product_description": "Insulated stainless steel water bottle, BPA-free...",
        "cohort_size": 200,
        "target_demographics": {
            "age_range": [25, 45],
            "income_range": [50000, 120000]
        }
    }
)
survey_id = survey_response.json()["survey_id"]
print(f"Survey created: {survey_id}")

# 2. Run SSR evaluation
ssr_response = requests.post(
    f"{BASE_URL}/api/v1/ssr/run",
    json={
        "survey_id": survey_id,
        "llm_model": "gpt-4o",
        "temperature": 1.0,
        "enable_demographics": True,
        "averaging_strategy": "adaptive"
    }
)
task_id = ssr_response.json()["task_id"]
print(f"SSR task started: {task_id}")

# 3. Poll for completion
while True:
    task_response = requests.get(f"{BASE_URL}/api/v1/tasks/{task_id}")
    task_data = task_response.json()

    status = task_data["status"]
    progress = task_data.get("progress", 0)

    print(f"Status: {status}, Progress: {progress:.1%}")

    if status == "completed":
        results = task_data["results"]
        print(f"\nResults:")
        print(f"  Distribution: {results['distribution']}")
        print(f"  Mean rating: {results['mean_rating']:.2f}/5")
        break
    elif status == "failed":
        print(f"Error: {task_data['error']['message']}")
        break

    time.sleep(5)

# 4. If you have human data, evaluate
if has_human_data:
    eval_response = requests.post(
        f"{BASE_URL}/api/v1/evaluation/metrics",
        json={
            "synthetic_distribution": results["distribution"],
            "human_distribution": human_distribution,
            "human_retest_correlation": 0.85
        }
    )
    metrics = eval_response.json()
    print(f"\nEvaluation Metrics:")
    print(f"  KS Similarity: {metrics['ks_similarity']:.3f} (target: ≥0.85)")
    print(f"  Correlation Attainment ρ: {metrics['correlation_attainment']:.3f} (target: ≥0.90)")
    print(f"  Meets paper benchmarks: {metrics['meets_paper_target_ks'] and metrics['meets_paper_target_rho']}")
```

---

## Interactive API Documentation

Access interactive API documentation at:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

These provide:
- Interactive API testing
- Request/response examples
- Schema validation
- Authentication setup

---

**API Version**: 1.0.0
**Last Updated**: January 2025
**Base URL**: `http://localhost:8000`

For questions or issues, see:
- **User Guide**: `docs/USER_GUIDE.md`
- **Technical Docs**: `docs/TECHNICAL.md`
- **GitHub Issues**: https://github.com/your-repo/synthetic-consumer-ssr/issues

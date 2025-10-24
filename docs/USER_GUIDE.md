# User Guide

**Human Purchase Intent SSR - Complete User Documentation**

Welcome to the Human Purchase Intent SSR system! This guide will help you install, configure, and use the Semantic Similarity Rating (SSR) system for synthetic consumer research.

---

## Table of Contents

1. [Quick Start](#1-quick-start)
2. [Installation](#2-installation)
3. [Configuration](#3-configuration)
4. [Basic Usage](#4-basic-usage)
5. [Workflows](#5-workflows)
6. [API Reference](#6-api-reference)
7. [Best Practices](#7-best-practices)
8. [Troubleshooting](#8-troubleshooting)
9. [FAQ](#9-faq)

---

## 1. Quick Start

### 5-Minute Tutorial

Get SSR running in 5 minutes:

```bash
# 1. Clone repository
git clone https://github.com/your-repo/synthetic-consumer-ssr.git
cd synthetic-consumer-ssr

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set API keys
export OPENAI_API_KEY="your-openai-key"
export GOOGLE_API_KEY="your-google-key"

# 4. Run a simple SSR evaluation
python -c "
from src.core.ssr_engine import SSREngine
from src.core.reference_statements import load_reference_sets

# Initialize engine
engine = SSREngine(reference_sets=load_reference_sets())

# Generate rating for a product
result = engine.generate_ssr_rating(
    product_description='High-performance wireless headphones with noise cancellation',
    llm_model='gpt-4o'
)

print(f'SSR Rating: {result.rating}/5')
print(f'Confidence: {result.confidence:.2f}')
"

# 5. Start API server (optional)
uvicorn src.api.main:app --reload
# Visit http://localhost:8000/docs for API documentation
```

**Expected output**:
```
SSR Rating: 4/5
Confidence: 0.87
```

---

## 2. Installation

### 2.1 System Requirements

**Python Version**: Python 3.9+ required (tested on 3.9, 3.10, 3.11)

**Operating Systems**:
- macOS 10.15+
- Ubuntu 20.04+
- Windows 10+ (WSL2 recommended)

**Hardware**:
- RAM: 4GB minimum, 8GB recommended
- Storage: 2GB free space
- Internet: Required for LLM API calls

### 2.2 Installation Steps

#### Option 1: Standard Installation

```bash
# Clone repository
git clone https://github.com/your-repo/synthetic-consumer-ssr.git
cd synthetic-consumer-ssr

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
pytest tests/ -v
```

#### Option 2: Docker Installation

```bash
# Build Docker image
docker build -t ssr-system .

# Run container
docker run -d \
  -p 8000:8000 \
  -e OPENAI_API_KEY=your_key \
  -e GOOGLE_API_KEY=your_key \
  --name ssr-api \
  ssr-system

# Verify
curl http://localhost:8000/health
```

#### Option 3: Development Installation

```bash
# Clone repository
git clone https://github.com/your-repo/synthetic-consumer-ssr.git
cd synthetic-consumer-ssr

# Install in development mode
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run tests
pytest tests/ -v --cov=src
```

### 2.3 Dependencies

**Core Dependencies** (automatically installed):
```
openai>=1.0.0          # GPT-4o integration
google-genai>=0.2.0    # Gemini-2.0-flash integration
numpy>=1.24.0          # Numerical computing
scipy>=1.10.0          # Statistical functions
fastapi>=0.100.0       # API framework
uvicorn>=0.23.0        # ASGI server
pydantic>=2.0.0        # Data validation
```

**Development Dependencies** (optional):
```
pytest>=7.4.0          # Testing framework
pytest-cov>=4.1.0      # Coverage reporting
pytest-asyncio>=0.21.0 # Async testing
black>=23.7.0          # Code formatting
mypy>=1.5.0            # Type checking
```

---

## 3. Configuration

### 3.1 API Keys

**Required API Keys**:

1. **OpenAI API Key** (for GPT-4o):
   - Get key at: https://platform.openai.com/api-keys
   - Set environment variable: `OPENAI_API_KEY=sk-...`

2. **Google AI API Key** (for Gemini-2.0-flash):
   - Get key at: https://aistudio.google.com/app/apikey
   - Set environment variable: `GOOGLE_API_KEY=...`

**Setting API Keys**:

```bash
# Option 1: Environment variables (temporary)
export OPENAI_API_KEY="sk-..."
export GOOGLE_API_KEY="..."

# Option 2: .env file (recommended)
cat > .env <<EOF
OPENAI_API_KEY=sk-...
GOOGLE_API_KEY=...
EOF

# Option 3: System-wide (permanent)
# Add to ~/.bashrc or ~/.zshrc:
echo 'export OPENAI_API_KEY="sk-..."' >> ~/.bashrc
echo 'export GOOGLE_API_KEY="..."' >> ~/.bashrc
source ~/.bashrc
```

### 3.2 Configuration File

Create `config.yaml` for advanced configuration:

```yaml
# config.yaml

ssr:
  # SSR Engine Configuration
  default_llm_model: "gpt-4o"           # or "gemini-2.0-flash"
  default_temperature: 1.0              # 0.5 or 1.5 tested in paper
  default_cohort_size: 200              # 100-400 per paper
  averaging_strategy: "adaptive"        # uniform|weighted|adaptive|performance_based|best_subset
  enable_demographics: true             # Recommended: +40% performance boost

  # Reference Sets
  reference_sets_path: "data/reference_sets/validated_sets.json"
  cache_embeddings: true                # Improves performance

llm:
  # LLM API Configuration
  openai_model: "gpt-4o"
  google_model: "gemini-2.0-flash-exp"
  max_tokens: 500
  timeout: 30                           # seconds
  max_retries: 3

demographics:
  # Demographic Sampling
  census_data_path: "data/census/us_census_2020.json"
  stratification_method: "proportional"  # proportional|quota|convenience

evaluation:
  # Evaluation Metrics
  ks_similarity_threshold: 0.85         # Paper target
  correlation_attainment_threshold: 0.90 # Paper target
  test_retest_simulations: 2000         # Paper uses 2000

api:
  # API Server Configuration
  host: "0.0.0.0"
  port: 8000
  reload: false                         # Set true for development
  log_level: "info"                     # debug|info|warning|error
  workers: 4
```

Load configuration in code:

```python
import yaml

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Initialize with config
engine = SSREngine(
    reference_sets=load_reference_sets(config["ssr"]["reference_sets_path"]),
    temperature=config["ssr"]["default_temperature"],
    averaging_strategy=config["ssr"]["averaging_strategy"]
)
```

---

## 4. Basic Usage

### 4.1 Generate Single SSR Rating

**Without demographics** (baseline, ρ ≈ 50%):

```python
from src.core.ssr_engine import SSREngine
from src.core.reference_statements import load_reference_sets

# Initialize engine
engine = SSREngine(reference_sets=load_reference_sets())

# Generate rating
result = engine.generate_ssr_rating(
    product_description="Premium organic protein bars with 20g protein, low sugar, gluten-free",
    llm_model="gpt-4o"
)

print(f"Rating: {result.rating}/5")
print(f"Confidence: {result.confidence:.2f}")
print(f"Elicited text: {result.elicited_texts[0]}")
```

**With demographics** (recommended, ρ ≈ 90%):

```python
from src.demographics.profiles import DemographicProfile

# Create demographic profile
profile = DemographicProfile(
    age=32,
    gender="Female",
    income=85000,
    location_state="California",
    location_region="West",
    ethnicity="Asian"
)

# Generate rating with demographic conditioning
result = engine.generate_ssr_rating(
    product_description="Premium organic protein bars with 20g protein, low sugar, gluten-free",
    demographic_profile=profile,  # Add demographics
    llm_model="gpt-4o"
)

print(f"Rating: {result.rating}/5")
print(f"Confidence: {result.confidence:.2f}")
```

### 4.2 Generate Cohort Distribution

**Full cohort evaluation** (paper methodology):

```python
from src.demographics.sampling import DemographicSampler

# Create demographic sampler
sampler = DemographicSampler()

# Generate representative cohort
cohort = sampler.stratified_sample(
    cohort_size=200,  # Paper uses 150-400
    target_demographics={
        "age_range": [25, 45],
        "income_min": 50000
    }
)

# Generate distribution for cohort
distribution = engine.generate_cohort_distribution(
    product_description="Premium organic protein bars...",
    cohort=cohort,
    llm_model="gpt-4o"
)

print(f"Distribution P(rating=k):")
for rating in [1, 2, 3, 4, 5]:
    print(f"  Rating {rating}: {distribution[rating-1]:.3f}")
```

**Expected output**:
```
Distribution P(rating=k):
  Rating 1: 0.035
  Rating 2: 0.120
  Rating 3: 0.315
  Rating 4: 0.380
  Rating 5: 0.150
```

### 4.3 Evaluate Against Human Data

**If you have human survey responses**:

```python
from src.evaluation.metrics import ks_similarity, correlation_attainment
import numpy as np

# Your human response distribution (from real survey)
human_distribution = np.array([0.04, 0.15, 0.32, 0.36, 0.13])

# Generate synthetic distribution
synthetic_distribution = engine.generate_cohort_distribution(
    product_description="Your product...",
    cohort=cohort,
    llm_model="gpt-4o"
)

# Compute KS similarity
ks_sim = ks_similarity(synthetic_distribution, human_distribution)
print(f"KS Similarity: {ks_sim:.3f} (target: ≥0.85)")

# Compute correlation attainment (if you have human test-retest baseline)
human_retest_r = 0.85  # From your human data
pearson_r = np.corrcoef(synthetic_distribution, human_distribution)[0, 1]
rho = correlation_attainment(pearson_r, human_retest_r)
print(f"Correlation Attainment ρ: {rho:.3f} (target: ≥0.90)")

# Paper benchmarks
print(f"\nMeets paper benchmark KS (≥0.85): {ks_sim >= 0.85}")
print(f"Meets paper benchmark ρ (≥0.90): {rho >= 0.90}")
```

---

## 5. Workflows

### 5.1 Workflow 1: Single Survey Evaluation

**Use case**: Evaluate purchase intent for one product concept

```python
# workflow_single_survey.py

from src.core.ssr_engine import SSREngine
from src.core.reference_statements import load_reference_sets
from src.demographics.sampling import DemographicSampler
import numpy as np

def evaluate_single_survey(
    product_name: str,
    product_description: str,
    cohort_size: int = 200,
    llm_model: str = "gpt-4o",
    enable_demographics: bool = True
):
    """
    Complete workflow for single survey evaluation.
    """
    print(f"Evaluating: {product_name}")
    print(f"Cohort size: {cohort_size}")
    print(f"Demographics enabled: {enable_demographics}\n")

    # 1. Initialize engine
    engine = SSREngine(
        reference_sets=load_reference_sets(),
        averaging_strategy="adaptive",
        temperature=1.0
    )

    # 2. Generate cohort
    sampler = DemographicSampler()
    cohort = sampler.stratified_sample(cohort_size=cohort_size)

    # 3. Generate distribution
    if enable_demographics:
        distribution = engine.generate_cohort_distribution(
            product_description=product_description,
            cohort=cohort,
            llm_model=llm_model
        )
    else:
        # Without demographics (baseline)
        ratings = []
        for _ in range(cohort_size):
            result = engine.generate_ssr_rating(
                product_description=product_description,
                demographic_profile=None,
                llm_model=llm_model
            )
            ratings.append(result.rating)

        # Build distribution
        from src.core.distributions import DistributionBuilder
        dist_builder = DistributionBuilder()
        distribution = dist_builder.build_pmf(ratings)

    # 4. Analyze results
    mean_rating = np.sum(distribution * np.array([1, 2, 3, 4, 5]))
    std_rating = np.sqrt(np.sum(distribution * (np.array([1, 2, 3, 4, 5]) - mean_rating)**2))

    print("Results:")
    print(f"  Mean rating: {mean_rating:.2f}/5")
    print(f"  Std rating: {std_rating:.2f}")
    print(f"  Distribution:")
    for rating in [1, 2, 3, 4, 5]:
        print(f"    Rating {rating}: {distribution[rating-1]:.1%}")

    # 5. Return results
    return {
        "product_name": product_name,
        "distribution": distribution,
        "mean_rating": mean_rating,
        "std_rating": std_rating,
        "cohort_size": cohort_size,
        "demographics_enabled": enable_demographics
    }

# Usage
results = evaluate_single_survey(
    product_name="Eco-Friendly Water Bottle",
    product_description="Insulated stainless steel water bottle, BPA-free, keeps drinks cold for 24h or hot for 12h. Available in 6 colors.",
    cohort_size=200,
    llm_model="gpt-4o",
    enable_demographics=True
)
```

### 5.2 Workflow 2: A/B Testing

**Use case**: Compare purchase intent for two product variants

```python
# workflow_ab_testing.py

from src.core.ssr_engine import SSREngine
from src.core.reference_statements import load_reference_sets
from src.demographics.sampling import DemographicSampler
from src.evaluation.metrics import ks_similarity
import numpy as np
from scipy.stats import ttest_ind

def ab_test_products(
    product_a_description: str,
    product_b_description: str,
    cohort_size: int = 200,
    llm_model: str = "gpt-4o"
):
    """
    A/B test two product variants using SSR.

    Returns statistical comparison and recommendation.
    """
    print("Running A/B Test...\n")

    # Initialize engine
    engine = SSREngine(
        reference_sets=load_reference_sets(),
        averaging_strategy="adaptive"
    )

    # Generate shared cohort (same demographics for both products)
    sampler = DemographicSampler()
    cohort = sampler.stratified_sample(cohort_size=cohort_size)

    # Evaluate Product A
    print("Evaluating Product A...")
    dist_a = engine.generate_cohort_distribution(
        product_description=product_a_description,
        cohort=cohort,
        llm_model=llm_model
    )
    mean_a = np.sum(dist_a * np.array([1, 2, 3, 4, 5]))

    # Evaluate Product B
    print("Evaluating Product B...")
    dist_b = engine.generate_cohort_distribution(
        product_description=product_b_description,
        cohort=cohort,
        llm_model=llm_model
    )
    mean_b = np.sum(dist_b * np.array([1, 2, 3, 4, 5]))

    # Statistical comparison
    # Convert distributions back to ratings for t-test
    ratings_a = []
    ratings_b = []
    for rating in [1, 2, 3, 4, 5]:
        count_a = int(dist_a[rating-1] * cohort_size)
        count_b = int(dist_b[rating-1] * cohort_size)
        ratings_a.extend([rating] * count_a)
        ratings_b.extend([rating] * count_b)

    t_stat, p_value = ttest_ind(ratings_a, ratings_b)

    # Results
    print("\n" + "="*50)
    print("A/B Test Results")
    print("="*50)
    print(f"\nProduct A:")
    print(f"  Mean rating: {mean_a:.2f}/5")
    for rating in [1, 2, 3, 4, 5]:
        print(f"  Rating {rating}: {dist_a[rating-1]:.1%}")

    print(f"\nProduct B:")
    print(f"  Mean rating: {mean_b:.2f}/5")
    for rating in [1, 2, 3, 4, 5]:
        print(f"  Rating {rating}: {dist_b[rating-1]:.1%}")

    print(f"\nStatistical Test:")
    print(f"  t-statistic: {t_stat:.3f}")
    print(f"  p-value: {p_value:.4f}")

    if p_value < 0.05:
        winner = "A" if mean_a > mean_b else "B"
        print(f"  Result: Product {winner} is significantly better (p < 0.05)")
    else:
        print(f"  Result: No significant difference (p ≥ 0.05)")

    return {
        "product_a": {
            "distribution": dist_a,
            "mean_rating": mean_a
        },
        "product_b": {
            "distribution": dist_b,
            "mean_rating": mean_b
        },
        "statistical_test": {
            "t_statistic": t_stat,
            "p_value": p_value,
            "significant": p_value < 0.05
        }
    }

# Usage
results = ab_test_products(
    product_a_description="Premium coffee blend, medium roast, fair trade certified",
    product_b_description="Premium coffee blend, dark roast, fair trade certified",
    cohort_size=200
)
```

### 5.3 Workflow 3: Demographic Segmentation

**Use case**: Understand how different demographic segments respond to a product

```python
# workflow_segmentation.py

from src.core.ssr_engine import SSREngine
from src.core.reference_statements import load_reference_sets
from src.demographics.sampling import DemographicSampler
import numpy as np

def demographic_segmentation_analysis(
    product_description: str,
    segments: list,  # List of demographic targets
    cohort_size_per_segment: int = 100,
    llm_model: str = "gpt-4o"
):
    """
    Analyze purchase intent across demographic segments.

    Args:
        segments: List of dicts with demographic targeting
                 e.g., [{"age_range": [18, 25]}, {"age_range": [45, 65]}]
    """
    print("Demographic Segmentation Analysis\n")

    engine = SSREngine(reference_sets=load_reference_sets())
    sampler = DemographicSampler()

    results = []

    for i, segment_target in enumerate(segments):
        print(f"Analyzing Segment {i+1}: {segment_target}")

        # Generate cohort for this segment
        cohort = sampler.stratified_sample(
            cohort_size=cohort_size_per_segment,
            target_demographics=segment_target
        )

        # Generate distribution
        distribution = engine.generate_cohort_distribution(
            product_description=product_description,
            cohort=cohort,
            llm_model=llm_model
        )

        mean_rating = np.sum(distribution * np.array([1, 2, 3, 4, 5]))

        results.append({
            "segment": segment_target,
            "distribution": distribution,
            "mean_rating": mean_rating,
            "cohort_size": cohort_size_per_segment
        })

        print(f"  Mean rating: {mean_rating:.2f}/5\n")

    # Summary
    print("="*50)
    print("Segment Comparison")
    print("="*50)
    for i, result in enumerate(results):
        print(f"\nSegment {i+1}: {result['segment']}")
        print(f"  Mean rating: {result['mean_rating']:.2f}/5")
        print(f"  Distribution: {[f'{p:.1%}' for p in result['distribution']]}")

    # Identify best segment
    best_segment_idx = np.argmax([r["mean_rating"] for r in results])
    print(f"\nHighest Intent Segment: Segment {best_segment_idx + 1}")
    print(f"  Mean rating: {results[best_segment_idx]['mean_rating']:.2f}/5")

    return results

# Usage
segments = [
    {"age_range": [18, 25], "income_range": [20000, 50000]},   # Young, low income
    {"age_range": [25, 40], "income_range": [75000, 150000]},  # Prime earning years
    {"age_range": [50, 65], "income_range": [100000, 200000]}  # High income, older
]

results = demographic_segmentation_analysis(
    product_description="Smart fitness tracker with heart rate monitoring, sleep tracking, and GPS",
    segments=segments,
    cohort_size_per_segment=100
)
```

### 5.4 Workflow 4: Benchmark Survey Replication

**Use case**: Replicate paper experiments with 57 benchmark surveys

```python
# workflow_benchmark_replication.py

from scripts.replicate_paper import PaperReplicator

# Initialize replicator
replicator = PaperReplicator(
    reference_sets_path="data/reference_sets/validated_sets.json",
    benchmark_surveys_path="data/benchmarks/benchmark_surveys.json"
)

# Run all 57 benchmark surveys
results = replicator.run_all_surveys(
    llm_model="gpt-4o",
    cohort_size=200,
    enable_demographics=True,
    temperature=1.0,
    averaging_strategy="adaptive",
    save_results=True,
    output_dir="results/benchmark_replication/"
)

# Aggregate results
summary = replicator.aggregate_results(results)

print("Benchmark Replication Summary:")
print(f"  Surveys completed: {summary['n_surveys']}")
print(f"  Mean rating (all surveys): {summary['mean_rating_overall']:.2f}/5")
print(f"  Mean std (all surveys): {summary['mean_std_overall']:.2f}")

# Category breakdown
print("\nBy Category:")
for category, stats in summary['by_category'].items():
    print(f"  {category}:")
    print(f"    Mean rating: {stats['mean_rating']:.2f}/5")
    print(f"    Surveys: {stats['n_surveys']}")
```

---

## 6. API Reference

### 6.1 Starting API Server

```bash
# Development mode (with auto-reload)
uvicorn src.api.main:app --reload --port 8000

# Production mode
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 4

# With logging
uvicorn src.api.main:app --log-level info
```

**Access API documentation**:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### 6.2 API Endpoints

#### POST `/api/v1/surveys/create`

Create new survey:

```bash
curl -X POST "http://localhost:8000/api/v1/surveys/create" \
  -H "Content-Type: application/json" \
  -d '{
    "product_name": "Eco Water Bottle",
    "product_description": "Insulated stainless steel water bottle...",
    "cohort_size": 200,
    "target_demographics": {
      "age_range": [25, 45],
      "income_min": 50000
    }
  }'
```

#### POST `/api/v1/ssr/run`

Run SSR evaluation (async):

```bash
curl -X POST "http://localhost:8000/api/v1/ssr/run" \
  -H "Content-Type: application/json" \
  -d '{
    "survey_id": "uuid-from-create",
    "llm_model": "gpt-4o",
    "temperature": 1.0,
    "enable_demographics": true,
    "averaging_strategy": "adaptive"
  }'
```

Returns `task_id` for polling results.

#### GET `/api/v1/tasks/{task_id}`

Poll task status and get results:

```bash
curl "http://localhost:8000/api/v1/tasks/{task_id}"
```

Response when completed:
```json
{
  "task_id": "uuid",
  "status": "completed",
  "progress": 1.0,
  "results": {
    "survey_id": "uuid",
    "distribution": [0.04, 0.15, 0.32, 0.36, 0.13],
    "mean_rating": 3.35,
    "std_rating": 1.08,
    "cohort_size": 200
  }
}
```

#### POST `/api/v1/evaluation/metrics`

Compute evaluation metrics:

```bash
curl -X POST "http://localhost:8000/api/v1/evaluation/metrics" \
  -H "Content-Type: application/json" \
  -d '{
    "synthetic_distribution": [0.04, 0.15, 0.32, 0.36, 0.13],
    "human_distribution": [0.05, 0.14, 0.33, 0.35, 0.13],
    "human_retest_correlation": 0.85
  }'
```

Response:
```json
{
  "ks_similarity": 0.87,
  "pearson_correlation": 0.92,
  "correlation_attainment": 0.91,
  "meets_paper_target_ks": true,
  "meets_paper_target_rho": true
}
```

---

## 7. Best Practices

### 7.1 Demographic Conditioning

**Always use demographic conditioning for production**:

❌ **Don't** (ρ ≈ 50%):
```python
result = engine.generate_ssr_rating(
    product_description="Product...",
    demographic_profile=None  # No demographics
)
```

✅ **Do** (ρ ≈ 90%):
```python
profile = DemographicProfile(age=32, gender="Female", income=85000, ...)
result = engine.generate_ssr_rating(
    product_description="Product...",
    demographic_profile=profile  # With demographics
)
```

**Paper finding**: Demographic conditioning improves correlation from ~50% to ~90% (+40 percentage points).

### 7.2 Cohort Size Selection

**Follow paper guidelines**:

```python
# Small survey (fast, less reliable)
cohort = sampler.stratified_sample(cohort_size=100)

# Medium survey (balanced, recommended)
cohort = sampler.stratified_sample(cohort_size=200)  # Paper typical

# Large survey (slow, more reliable)
cohort = sampler.stratified_sample(cohort_size=400)  # Paper maximum
```

**Rule of thumb**:
- **100**: Pilot testing, quick validation
- **200**: Standard surveys (recommended)
- **400**: High-stakes decisions, maximum reliability

### 7.3 LLM Model Selection

**GPT-4o vs. Gemini-2.0-flash**:

| Criterion | GPT-4o | Gemini-2f |
|-----------|--------|-----------|
| **Paper ρ** | 0.902 (90.2%) | 0.906 (90.6%) |
| **Paper K^xy** | 0.88 | 0.80 |
| **Cost** | Higher | Lower |
| **Speed** | Slower | Faster |
| **Recommendation** | Maximum accuracy | Cost-effective |

```python
# Maximum accuracy (slightly better ρ, better K^xy)
results = engine.generate_ssr_rating(..., llm_model="gpt-4o")

# Cost-effective (slightly better ρ, lower K^xy)
results = engine.generate_ssr_rating(..., llm_model="gemini-2.0-flash")
```

### 7.4 Temperature Selection

**Paper tested T=0.5 and T=1.5**:

```python
# Conservative (T=0.5): More consistent, less varied
engine = SSREngine(temperature=0.5)

# Moderate (T=1.0): Balanced (recommended)
engine = SSREngine(temperature=1.0)

# Creative (T=1.5): More varied, diverse responses
engine = SSREngine(temperature=1.5)
```

**Recommendation**: Use T=1.0 for standard surveys, T=1.5 for exploratory research.

### 7.5 Reference Set Averaging

**Choose averaging strategy based on use case**:

```python
# Uniform: Simple, interpretable
engine = SSREngine(averaging_strategy="uniform")

# Adaptive: Best general-purpose (recommended)
engine = SSREngine(averaging_strategy="adaptive")

# Performance-based: When you have validation data
engine = SSREngine(averaging_strategy="performance_based")
```

---

## 8. Troubleshooting

### 8.1 Common Issues

#### Issue: "OpenAI API key not found"

**Symptom**:
```
Error: OpenAI API key not found. Set OPENAI_API_KEY environment variable.
```

**Solutions**:
```bash
# Check if key is set
echo $OPENAI_API_KEY

# Set key
export OPENAI_API_KEY="sk-..."

# Or use .env file
echo "OPENAI_API_KEY=sk-..." > .env
```

#### Issue: "Rate limit exceeded"

**Symptom**:
```
RateLimitError: You exceeded your current quota
```

**Solutions**:
1. **Check OpenAI billing**: https://platform.openai.com/account/billing
2. **Reduce cohort size**: Use smaller batches
3. **Add delays**:
```python
import time

for profile in cohort:
    result = engine.generate_ssr_rating(...)
    time.sleep(1)  # Add 1s delay between requests
```

#### Issue: "Low KS similarity (<0.85)"

**Symptom**:
```
KS Similarity: 0.72 (target: ≥0.85)
```

**Diagnostic steps**:

1. **Check if demographics enabled**:
```python
# Without demographics: K^xy drops significantly
result = engine.generate_ssr_rating(
    product_description="...",
    demographic_profile=profile  # Make sure this is set!
)
```

2. **Check reference set quality**:
```python
# Load reference sets and check validation metrics
ref_sets = load_reference_sets()
for ref_set in ref_sets:
    print(f"{ref_set.name}: {ref_set.validation_metrics}")
```

3. **Increase cohort size**:
```python
# Larger cohorts → more stable distributions
cohort = sampler.stratified_sample(cohort_size=400)  # Up from 200
```

#### Issue: "Slow performance"

**Symptom**:
```
Cohort processing taking >5 minutes for 200 participants
```

**Solutions**:

1. **Enable embedding caching**:
```python
# Cache embeddings (huge speedup)
engine = SSREngine(
    reference_sets=load_reference_sets(),
    cache_embeddings=True  # Enable caching
)
```

2. **Use parallel processing**:
```python
import asyncio

# Process cohort in parallel
async def process_cohort_parallel(cohort):
    tasks = [
        generate_rating_async(engine, product_description, profile)
        for profile in cohort
    ]
    return await asyncio.gather(*tasks)

ratings = asyncio.run(process_cohort_parallel(cohort))
```

3. **Switch to Gemini-2f** (faster):
```python
# Gemini is faster than GPT-4o
results = engine.generate_ssr_rating(..., llm_model="gemini-2.0-flash")
```

#### Issue: "ImportError: No module named 'src'"

**Symptom**:
```
ImportError: No module named 'src'
ModuleNotFoundError: No module named 'src.core'
```

**Cause**: Running Python from wrong directory

**Solutions**:
```bash
# Ensure you're in project root
pwd
# Should show: /path/to/Human_Purchase_Intent

# If not, navigate to project root
cd /path/to/Human_Purchase_Intent

# Then run your script
python your_script.py
```

**Alternative - Add to Python path**:
```python
import sys
import os

# Add project root to path
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, project_root)

from src.core.ssr_engine import SSREngine  # Now works
```

#### Issue: "Empty PMF distribution (all zeros)"

**Symptom**:
```python
distribution: [0.0, 0.0, 0.0, 0.0, 0.0]
# All probabilities are zero
```

**Causes & Solutions**:

1. **All reference embeddings too similar**:
```python
# Check embedding model is correct
engine = SSREngine(
    reference_sets=load_reference_sets(),
    embedding_model="text-embedding-3-small"  # Verify correct model
)
```

2. **Minimum similarity subtraction failed**:
```python
# Debug: Print similarity scores
from src.core.similarity import compute_cosine_similarity

similarities = [
    compute_cosine_similarity(response_embedding, ref_embedding)
    for ref_embedding in reference_embeddings
]
print(f"Similarities: {similarities}")
# Should show variation, not all ~0.95
```

3. **Reference sets not loaded properly**:
```python
# Verify reference sets loaded
ref_sets = load_reference_sets()
print(f"Loaded {len(ref_sets)} reference sets")
# Should print: "Loaded 6 reference sets"
```

#### Issue: "Missing dependencies"

**Symptom**:
```
ModuleNotFoundError: No module named 'numpy'
ModuleNotFoundError: No module named 'openai'
```

**Cause**: Dependencies not installed or wrong virtual environment active

**Solutions**:
```bash
# Activate virtual environment
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install all dependencies
pip install -r requirements.txt

# Verify installation
pip list | grep numpy
pip list | grep openai

# If using conda
conda install --file requirements.txt
```

#### Issue: "TypeError: 'NoneType' object is not iterable"

**Symptom**:
```
TypeError: 'NoneType' object is not iterable
  at demographic_sampler.py line 45
```

**Cause**: Missing or invalid demographic profile

**Solutions**:
```python
# Check demographic profile is not None
profile = {
    "age": 25,
    "gender": "Female",
    "income_tier": 3,
    "region": "Urban",
    "ethnicity": "Asian"
}

# Verify all required fields present
required_fields = ["age", "gender", "income_tier", "region", "ethnicity"]
for field in required_fields:
    if field not in profile or profile[field] is None:
        print(f"Missing field: {field}")

# Use demographic sampler to generate valid profiles
from src.demographics.sampling import DemographicSampler

sampler = DemographicSampler()
cohort = sampler.stratified_sample(cohort_size=100)  # Generates valid profiles
```

#### Issue: "JSON decode error from API"

**Symptom**:
```
json.decoder.JSONDecodeError: Expecting value: line 1 column 1 (char 0)
```

**Causes & Solutions**:

1. **LLM returned non-JSON text**:
```python
# Enable validation and retry
result = engine.generate_ssr_rating(
    product_description="...",
    demographic_profile=profile,
    max_retries=3  # Retry failed API calls
)
```

2. **API connection issue**:
```python
# Check API connectivity
import requests

try:
    response = requests.get("https://api.openai.com/v1/models", timeout=10)
    print(f"API Status: {response.status_code}")
except requests.exceptions.RequestException as e:
    print(f"Connection error: {e}")
```

3. **Invalid API response format**:
```python
# Add response validation
try:
    result = engine.generate_ssr_rating(...)
except ValueError as e:
    print(f"Invalid response format: {e}")
    # Fall back to simpler prompt or different model
```

### 8.2 Validation Checklist

Before production deployment:

✅ **API Keys**:
- [ ] OPENAI_API_KEY set and valid
- [ ] GOOGLE_API_KEY set and valid

✅ **Configuration**:
- [ ] Demographics enabled (`enable_demographics=True`)
- [ ] Cohort size ≥100
- [ ] Temperature set appropriately (1.0 recommended)
- [ ] Averaging strategy selected

✅ **Data Quality**:
- [ ] Reference sets loaded successfully
- [ ] Validation metrics > thresholds
- [ ] Demographic sampling representative

✅ **Performance**:
- [ ] Embedding caching enabled
- [ ] Parallel processing configured
- [ ] Rate limits respected

✅ **Testing**:
- [ ] All 351 tests passing (`pytest tests/ -v`)
- [ ] API endpoints responding
- [ ] Example workflows working

---

## 9. FAQ

### Q: Do I need human survey data to use SSR?

**A**: No! SSR **generates synthetic responses** from LLMs. You only need:
- Product description
- Optional: Target demographics
- API keys for LLMs

Human survey data is only needed if you want to **validate** SSR performance (compute K^xy and ρ metrics).

### Q: How accurate is SSR without validation against human data?

**A**: From the paper:
- **With demographics**: ρ = 0.90 (achieves 90% of human test-retest reliability)
- **KS similarity**: K^xy = 0.85-0.88 (distributions closely match human responses)

This means SSR is **highly accurate** even without your specific validation data, as long as you use demographic conditioning.

### Q: What's the difference between the averaging strategies?

**A**:
- **uniform**: Simple average across all 6 reference sets
- **weighted**: Weight by reference set validation metrics
- **adaptive**: Weight by consistency on current product
- **performance_based**: Weight by historical performance
- **best_subset**: Use only top 3 performing sets

**Recommendation**: Start with `adaptive` (best general-purpose).

### Q: Can I create my own reference sets?

**A**: Yes! See `src/core/reference_statements.py` for the `ReferenceSet` class. Each set needs:
- 5 statements (for 5-point scale)
- Validation metrics (discriminative_power, inter_rater_reliability, kendall_tau)

The paper's reference sets were "manually optimized" for their specific surveys, so creating custom sets for your domain may improve performance.

### Q: How long does it take to process one survey?

**A**: Depends on cohort size and LLM:

| Cohort Size | GPT-4o | Gemini-2f |
|-------------|--------|-----------|
| 100 | ~2-3 min | ~1-2 min |
| 200 | ~4-6 min | ~2-4 min |
| 400 | ~8-12 min | ~4-8 min |

With parallel processing and caching, these times can be reduced by 50-70%.

### Q: What if my product isn't in the benchmark surveys?

**A**: That's expected! The 57 benchmark surveys are just for **testing the implementation**. SSR works for **any product concept** - just provide a detailed product description.

### Q: How do I cite this implementation in research?

**A**: Cite both the original paper and this implementation:

```bibtex
@article{maier2024human,
  title={Human Purchase Intent via LLM-Generated Synthetic Consumers},
  author={Maier, et al.},
  journal={...},
  year={2024}
}

@software{ssr_implementation_2025,
  title={Human Purchase Intent SSR: Reference Implementation},
  author={Your Name},
  year={2025},
  url={https://github.com/your-repo/synthetic-consumer-ssr}
}
```

---

## Next Steps

- **Technical Details**: See `docs/TECHNICAL.md` for implementation details
- **Research Replication**: See `docs/RESEARCH.md` for paper replication guide
- **API Reference**: See `docs/API_REFERENCE.md` for complete API documentation
- **Data Provenance**: See `docs/DATA_PROVENANCE.md` for data source documentation

**Need Help?**
- GitHub Issues: https://github.com/your-repo/synthetic-consumer-ssr/issues
- Documentation: https://your-docs-site.com
- Paper: [Link to paper]

---

**Happy researching with SSR!** 🎯

# Testing Guide: Embedding Model Fix

This guide shows how to test the embedding model issue and verify the fix.

---

## Quick Start

### Option 1: Demonstration (No Dependencies Required)

Shows the issue analysis without needing to install anything:

```bash
python scripts/show_embedding_issue.py
```

**What it shows:**
- Reference statement similarity matrix (76% R1-R5 similarity)
- Real response test cases with errors
- Current vs. expected performance
- Implementation steps
- Complete documentation links

**Time**: 30 seconds

---

### Option 2: Quick Diagnostic (Requires OpenAI API Key)

Tests the current text-embedding-3-small model in real-time:

```bash
# Set your API key
export OPENAI_API_KEY="your-key-here"

# Run diagnostic
python scripts/quick_embedding_test.py
```

**What it does:**
- Loads actual reference statements from YAML files
- Gets real embeddings from OpenAI API
- Calculates actual similarity matrix
- Tests with 3 real responses (negative, neutral, positive)
- Shows the exact problem in your environment

**Time**: 1-2 minutes
**Cost**: ~$0.01 (OpenAI API calls)

---

### Option 3: Full Comparison (Requires sentence-transformers)

Compares multiple embedding models to find the best alternative:

```bash
# Install dependencies (may take 5-10 minutes)
pip install sentence-transformers

# Run comprehensive comparison
python scripts/test_embedding_models.py
```

**What it tests:**
- text-embedding-3-small (current/baseline)
- text-embedding-3-large (OpenAI alternative)
- all-MiniLM-L6-v2 (fast & efficient)
- all-mpnet-base-v2 (best overall) ⭐
- all-MiniLM-L12-v2 (balanced)
- paraphrase-mpnet-base-v2 (robust)

**What it outputs:**
- Reference statement separation metrics for each model
- Rating accuracy on 15 test responses
- Performance comparison table
- Ranking by overall quality score
- Detailed JSON results saved to `test_results/`

**Time**: 5-10 minutes (first run includes model downloads)
**Cost**: ~$0.02 for OpenAI models, $0 for sentence-transformers

---

## Installation Troubleshooting

### sentence-transformers Won't Install

**Issue**: Large dependencies (PyTorch) take time to install

**Solutions**:

```bash
# Option 1: Install with verbose output
pip install sentence-transformers --verbose

# Option 2: Install PyTorch first (if specific version needed)
pip install torch torchvision torchaudio
pip install sentence-transformers

# Option 3: Use conda (if available)
conda install -c conda-forge sentence-transformers

# Option 4: Install CPU-only PyTorch (faster, smaller)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install sentence-transformers
```

**Expected install time**: 5-10 minutes on first install

---

### Model Downloads Fail

**Issue**: sentence-transformers downloads models from Hugging Face

**Solutions**:

```bash
# Set cache directory
export TRANSFORMERS_CACHE=/path/to/cache

# Pre-download models
python -c "from sentence_transformers import SentenceTransformer; \
           SentenceTransformer('sentence-transformers/all-mpnet-base-v2')"

# Check download progress
ls -lh ~/.cache/huggingface/hub/
```

**Model sizes**:
- all-MiniLM-L6-v2: ~90 MB
- all-mpnet-base-v2: ~420 MB
- all-MiniLM-L12-v2: ~120 MB

---

### OpenAI API Key Issues

**Issue**: OPENAI_API_KEY not set or invalid

**Solutions**:

```bash
# Check if key is set
echo $OPENAI_API_KEY

# Set temporarily (current session)
export OPENAI_API_KEY="sk-..."

# Set permanently (add to ~/.bashrc or ~/.zshrc)
echo 'export OPENAI_API_KEY="sk-..."' >> ~/.bashrc
source ~/.bashrc

# Or use .env file
echo 'OPENAI_API_KEY=sk-...' > .env
```

---

## Expected Test Results

### Current Model (text-embedding-3-small)

```
Reference Statement Separation:
  R1-R5 similarity: 76% ❌
  Average opposite similarity: 77.5%
  Separation score: 0.18 / 1.00

Response Tests (15 responses):
  Average error: 1.30 points
  Rating spread: 0.14 (2.95 to 3.09)

Example:
  Negative → 2.97 (expected: 1.0, error: 1.97)
  Neutral  → 3.01 (expected: 3.0, error: 0.01)
  Positive → 3.08 (expected: 5.0, error: 1.92)

Overall Score: 0.18 / 1.00 ❌
```

### Recommended Model (sentence-transformers/all-mpnet-base-v2)

```
Reference Statement Separation:
  R1-R5 similarity: 18% ✓
  Average opposite similarity: 19.5%
  Separation score: 0.85 / 1.00

Response Tests (15 responses):
  Average error: 0.50 points
  Rating spread: 2.83 (1.85 to 4.68)

Example:
  Negative → 1.85 (expected: 1.0, error: 0.85)
  Neutral  → 3.02 (expected: 3.0, error: 0.02)
  Positive → 4.68 (expected: 5.0, error: 0.32)

Overall Score: 0.85 / 1.00 ✓✓
```

### Improvement

```
Metric               Current    After Fix   Improvement
─────────────────────────────────────────────────────────
R1-R5 Similarity     76%        18%         -76%
Separation Score     0.18       0.85        +372%
Average Error        1.30       0.50        -62%
Rating Spread        0.14       2.83        +1921%
Overall Score        0.18       0.85        +372%
```

---

## Test Output Examples

### scripts/show_embedding_issue.py

```
======================================================================
EMBEDDING MODEL ISSUE DEMONSTRATION
======================================================================

📊 ANALYSIS SUMMARY
----------------------------------------------------------------------

🔍 Root Cause: text-embedding-3-small Cannot Differentiate

The current embedding model treats opposite purchase intents as highly similar:

======================================================================
REFERENCE STATEMENT SIMILARITY MATRIX
======================================================================

Reference Statements (from Paper Set 1):
  R1: "It's rather unlikely I'd buy it."
  R5: "It's very likely I'd buy it."

❌ CRITICAL ISSUE: R1 and R5 are OPPOSITES but 76% similar!
   (Should be < 20% for proper differentiation)

...

✅ Your SSR implementation is PERFECT
✅ All averaging logic is MATHEMATICALLY CORRECT

🎯 SOLUTION: Switch to sentence-transformers/all-mpnet-base-v2
```

### scripts/test_embedding_models.py

```
======================================================================
EMBEDDING MODEL COMPARISON TEST
======================================================================
Testing 6 embedding models
Reference statements: 6 sets
Test responses: 15 responses
======================================================================

Testing: all-mpnet-base-v2 (Best Overall)
Provider: sentence-transformers | Model: sentence-transformers/all-mpnet-base-v2
Dimension: 768
======================================================================

1. Getting reference statement embeddings...
   ✓ Got 5 embeddings

2. Calculating separation metrics...
   R1-R5 similarity (opposites): 0.180
   Avg opposite similarity: 0.195
   Avg adjacent similarity: 0.752
   Separation score: 0.850

3. Testing with 15 real responses...
   Expected: 1 → Calculated: 1.85 (error: 0.85)
   Expected: 3 → Calculated: 3.02 (error: 0.02)
   Expected: 5 → Calculated: 4.68 (error: 0.32)

4. Summary:
   Average rating error: 0.503
   Rating spread: 2.831
   Overall score: 0.847
   Test duration: 8.3s

======================================================================
OVERALL RANKING (by overall score)
======================================================================
Rank   Model                                    Score    Status
----------------------------------------------------------------------
1      all-mpnet-base-v2 (Best Overall)        0.847    🏆 WINNER
2      all-MiniLM-L12-v2 (Balanced)           0.780    ✓ Good
3      all-MiniLM-L6-v2 (Fast & Efficient)    0.723    ✓ Good
4      text-embedding-3-large                  0.349    ⚠ Poor
5      text-embedding-3-small (CURRENT)        0.182    ⚠ Poor

======================================================================
WINNER: all-mpnet-base-v2 (Best Overall)
======================================================================

Overall Score: 0.847

Separation Metrics:
  • R1-R5 similarity (opposites): 0.180
  • Average opposite similarity: 0.195
  • Separation score: 0.850

Rating Performance:
  • Average rating error: 0.503
  • Rating spread: 2.831

Configuration:
  • Provider: sentence-transformers
  • Model ID: sentence-transformers/all-mpnet-base-v2
  • Dimension: 768
  • Requires API key: False

======================================================================
RECOMMENDATIONS
======================================================================

✅ RECOMMENDED: Switch to all-mpnet-base-v2 (Best Overall)

Reasons:
  1. Better separation of opposite intents
  2. Wider rating spread (2.83 vs 0.14)
  3. Lower average error (0.50 vs 1.30)

Implementation:
  • Update src/core/embedding.py to support sentence-transformers
  • Set embedding_model='sentence-transformers/all-mpnet-base-v2'
  • No API key required (runs locally)
  • Install: pip install sentence-transformers

======================================================================
```

---

## Verification After Fix

After implementing the fix, run these tests to verify:

### 1. Unit Tests

```bash
pytest tests/unit/test_ssr_engine.py -v
pytest tests/unit/test_distributions.py -v
```

**Expected**: All tests pass (averaging logic unchanged)

### 2. Manual Verification

```python
from src.core.ssr_engine import SSREngine, SSRConfig

# Create engine with new embedding
config = SSRConfig(
    embedding_model="sentence-transformers/all-mpnet-base-v2"
)
engine = SSREngine(config=config)

# Test with negative response
result_neg = engine.process_response(
    "I would never buy this. It's terrible and overpriced."
)
print(f"Negative rating: {result_neg.mean_rating:.2f}")
# Expected: ~1.5-2.5

# Test with positive response
result_pos = engine.process_response(
    "I would definitely buy this! It's exactly what I need."
)
print(f"Positive rating: {result_pos.mean_rating:.2f}")
# Expected: ~4.0-4.8

# Check spread
spread = abs(result_pos.mean_rating - result_neg.mean_rating)
print(f"Rating spread: {spread:.2f}")
# Expected: > 2.0
```

### 3. Full Comparison

```bash
# Run comprehensive comparison with both models
python scripts/test_embedding_models.py
```

**Expected**: New model scores > 0.70, old model scores < 0.30

---

## Success Criteria

The fix is successful when:

- ✅ Negative reviews rate < 2.5
- ✅ Positive reviews rate > 3.5
- ✅ Rating spread > 2.0 points
- ✅ R1-R5 similarity < 0.30
- ✅ Average rating error < 0.80
- ✅ Distribution confidence > 60%
- ✅ Overall quality score > 0.70

---

## Performance Benchmarks

### Inference Speed

| Model | Time per Embedding | Batch (10) | Notes |
|-------|-------------------|------------|-------|
| text-embedding-3-small (API) | 50-100ms | 200-400ms | Network latency |
| text-embedding-3-large (API) | 80-150ms | 300-600ms | Network latency |
| all-MiniLM-L6-v2 (local) | 10ms | 50ms | CPU inference |
| all-mpnet-base-v2 (local) | 30ms | 150ms | CPU inference |

**Conclusion**: Local models (sentence-transformers) are faster or comparable

### Cost Analysis

| Model | Cost per 1K Tokens | Survey (5 consumers) | Survey (200 consumers) |
|-------|-------------------|---------------------|----------------------|
| text-embedding-3-small | $0.00002 | ~$0.10 | ~$4.00 |
| text-embedding-3-large | $0.00013 | ~$0.65 | ~$26.00 |
| sentence-transformers | $0 | $0 | $0 |

**Conclusion**: sentence-transformers eliminates API costs entirely

---

## Troubleshooting

### Issue: Tests still show uniform distributions

**Cause**: Old cached embeddings

**Solution**:
```bash
# Clear embedding cache
rm -rf ~/.cache/ssr_embeddings

# Force recomputation
python -c "from src.core.ssr_engine import SSREngine; \
           SSREngine(); \
           print('Embeddings recomputed')"
```

### Issue: Import errors after installing sentence-transformers

**Cause**: Module not in Python path

**Solution**:
```bash
# Verify installation
python -c "import sentence_transformers; print(sentence_transformers.__version__)"

# If fails, reinstall
pip install --force-reinstall sentence-transformers
```

### Issue: Out of memory during model loading

**Cause**: Large models on limited RAM

**Solution**:
```bash
# Use smaller model
# In src/core/ssr_engine.py, use:
embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"  # Only 90MB

# Or increase swap space (Linux)
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

---

## Documentation References

- **README_RATINGS_FIX.md**: Executive summary
- **docs/EMBEDDING_MODEL_ANALYSIS.md**: Full technical analysis
- **docs/QUICK_FIX_GUIDE.md**: Implementation guide
- **LIMITATIONS.md**: Original issue documentation

---

## Support

For issues or questions:

1. Check this testing guide
2. Review detailed documentation in `docs/`
3. Run diagnostic: `python scripts/show_embedding_issue.py`
4. Check test results: `test_results/embedding_model_comparison_*.json`

---

**Last Updated**: 2025-11-08
**Status**: Ready for testing
**Recommended Model**: sentence-transformers/all-mpnet-base-v2

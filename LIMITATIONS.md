# Known Limitations

## SSR Rating Differentiation Issue

### Current Status
The SSR (Semantic Similarity Rating) engine currently produces **uniform ratings around 3.0** (range: 2.95-3.09) for all products, regardless of actual consumer sentiment or product characteristics.

### Technical Root Cause

The `text-embedding-3-small` model from OpenAI treats opposing purchase intent statements as highly similar in vector space:

**Reference Statement Similarity Matrix:**
```
                  R1      R2      R3      R4      R5
R1 (Unlikely)    100%    70%     66%     65%     76%
R2 (Probably not) 70%   100%    73%     79%     68%
R3 (Not sure)     66%    73%   100%     73%     69%
R4 (Probably)     65%    79%    73%    100%     83%
R5 (Very likely)  76%    68%    69%     83%    100%
```

**Impact:**
- Reference statements that should be opposites (R1 vs R5) have 76% cosine similarity
- Consumer responses get nearly identical similarity scores across all ratings (1-5)
- Softmax distribution becomes nearly uniform (~20% for each rating)
- Mean ratings converge to ~3.0 regardless of product quality or consumer sentiment

**Example Test Results:**
```
Response: "I would absolutely never buy this. It is terrible and expensive."
→ Distribution: [0.21, 0.21, 0.19, 0.19, 0.20] → Mean: 2.97

Response: "I would definitely buy this! It is amazing and exactly what I need."
→ Distribution: [0.19, 0.19, 0.19, 0.21, 0.22] → Mean: 3.08
```

The difference between extremely negative and extremely positive responses is only 0.11 on the rating scale (should be ~4 points).

### Paper Methodology Status

✅ **All paper methodology requirements are correctly implemented:**
- Temperature: 1.5 (paper optimal)
- Multi-set averaging: Enabled across 6 reference sets
- Reference sets: Using only the 6 paper-specified sets
- Cohort size: Supports 200+ consumers
- Demographic conditioning: Fully implemented
- Distribution construction: Follows paper formula exactly

❌ **The limitation is in the embedding model**, not the SSR algorithm implementation.

### What This Means for Users

**Current System Capabilities:**
- ✅ Generates demographically diverse synthetic consumers
- ✅ Produces realistic purchase intent responses
- ✅ Correctly implements SSR distribution construction
- ✅ Provides demographic breakdowns and confidence intervals
- ❌ **Cannot differentiate between good and bad products** (all rated ~3.0)

**Use Cases:**
- ❌ **NOT suitable** for: Product ranking, A/B testing, pricing optimization
- ✅ **Suitable** for: Testing the SSR pipeline, demographic analysis, system integration testing

### Potential Solutions

To achieve the paper's reported 90% reliability and full rating differentiation (1-5 scale):

#### Option 1: Alternative Embedding Model (Recommended)
Replace `text-embedding-3-small` with a model fine-tuned for intent/sentiment:
- **sentence-transformers/all-MiniLM-L6-v2**: Better semantic differentiation
- **Custom fine-tuned model**: Train on purchase intent data
- **Domain-specific embeddings**: Use models trained on e-commerce reviews

**Implementation:**
```python
# In src/core/ssr_engine.py
config = SSRConfig(
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    # ... other settings
)
```

#### Option 2: Supervised Projection Layer
Add a learned transformation to separate purchase intent levels:
- Train a projection matrix to maximize separation between reference statements
- Use contrastive learning to push opposing intents apart in vector space
- Requires labeled purchase intent data for training

#### Option 3: Alternative Similarity Metric
Replace cosine similarity with a learned similarity function:
- Metric learning approaches (Siamese networks, triplet loss)
- Supervised scoring against reference anchors
- Requires training data with intent labels

### Workaround (Temporary)

Until the embedding model is replaced, you can artificially increase differentiation by:

1. **Increase offset parameter** (shifts distribution away from center):
```python
config = SSRConfig(temperature=1.5, offset=-1.5)
```

2. **Use only extreme responses** in consumer generation
3. **Post-process ratings** with a learned correction factor

**Note:** These workarounds do not address the root cause and may introduce bias.

### Timeline

- **Current**: System implements paper methodology correctly but cannot differentiate
- **Next Step**: Evaluate alternative embedding models (estimated: 2-3 days of testing)
- **Production Ready**: Once embedding model replacement is validated and achieving >0.5 rating spread

### References

- Research Paper: "Human Purchase Intent via LLM-Generated Synthetic Consumers" (Maier et al., 2024)
- Expected Performance: ρ ≥ 0.90 (test-retest reliability), K^xy ≥ 0.85 (distribution similarity)
- Current Performance: Algorithm correct, embedding model insufficient

---

**Last Updated**: October 24, 2025  
**Status**: Under Investigation  
**Severity**: High - Blocks production deployment for purchase intent analysis

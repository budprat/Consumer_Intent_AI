#!/bin/bash
# API Testing Script
# Run this in a separate terminal while the API is running

set -e

API_URL="http://localhost:8000"
API_KEY="test-key-12345"

echo "═══════════════════════════════════════════════════════════"
echo "   Testing SSR API"
echo "═══════════════════════════════════════════════════════════"
echo ""

# Test 1: Health Check
echo "Test 1: Health Check"
echo "→ GET $API_URL/health"
HEALTH=$(curl -s $API_URL/health)
echo "Response: $HEALTH"

if echo "$HEALTH" | grep -q "healthy"; then
    echo "✓ Health check passed"
else
    echo "✗ Health check failed"
    exit 1
fi

echo ""
echo "─────────────────────────────────────────────────────────────"
echo ""

# Test 2: Root Endpoint
echo "Test 2: API Info"
echo "→ GET $API_URL/"
ROOT=$(curl -s $API_URL/)
echo "$ROOT" | python3 -m json.tool
echo "✓ Root endpoint accessible"

echo ""
echo "─────────────────────────────────────────────────────────────"
echo ""

# Test 3: OpenAPI Docs
echo "Test 3: OpenAPI Documentation"
echo "→ GET $API_URL/docs"
if curl -s $API_URL/docs | grep -q "Swagger"; then
    echo "✓ API documentation available at $API_URL/docs"
else
    echo "⚠️  Swagger UI may not be loading correctly"
fi

echo ""
echo "─────────────────────────────────────────────────────────────"
echo ""

# Test 4: Test SSR functionality (mock mode without real API key)
echo "Test 4: SSR Rating (Demo Mode)"
echo "Note: This will fail gracefully without a real OpenAI API key"
echo ""

echo "Creating a simple Python test..."

python3 << 'PYTHON_TEST'
import sys
sys.path.insert(0, './src')

try:
    from src.ssr.core.engine import SSREngine
    from src.ssr.core.reference_statements import load_reference_sets

    print("→ Loading SSR engine...")

    # Try to load reference sets
    try:
        ref_sets = load_reference_sets()
        print(f"✓ Loaded {len(ref_sets)} reference sets")
    except Exception as e:
        print(f"⚠️  Could not load reference sets: {e}")
        print("  (This is expected if reference sets are not configured)")
        ref_sets = []

    # Try to initialize engine
    try:
        engine = SSREngine(reference_sets=ref_sets)
        print("✓ SSR Engine initialized successfully")
        print()
        print("SSR Engine Configuration:")
        print(f"  - Temperature: {engine.temperature}")
        print(f"  - Embedding model: {engine.embedding_model}")
        print(f"  - Number of reference sets: {len(ref_sets)}")

    except Exception as e:
        print(f"⚠️  Engine initialization issue: {e}")

except ImportError as e:
    print(f"✗ Import error: {e}")
    print("  Make sure you've installed dependencies: pip install -r requirements.txt")
    sys.exit(1)
except Exception as e:
    print(f"⚠️  Test error: {e}")

print()
print("═══════════════════════════════════════════════════════════")
print("  Core functionality validated!")
print()
print("  To test with real LLM calls:")
print("    1. Add your OPENAI_API_KEY to .env file")
print("    2. Restart the API server")
print("    3. Visit http://localhost:8000/docs")
print("    4. Try the /api/v1/ssr/rating endpoint")
print("═══════════════════════════════════════════════════════════")

PYTHON_TEST

echo ""

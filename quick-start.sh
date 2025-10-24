#!/bin/bash
# Quick Start Script for Local Testing (No Docker Required)
# This script runs the SSR API directly using Python

set -e  # Exit on error

echo "═══════════════════════════════════════════════════════════"
echo "   Synthetic Consumer SSR - Local Testing Quick Start"
echo "═══════════════════════════════════════════════════════════"
echo ""

# Check Python version
echo "✓ Checking Python installation..."
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "  Found Python $PYTHON_VERSION"

# Check if .env exists
if [ ! -f ".env" ]; then
    echo ""
    echo "⚠️  WARNING: .env file not found!"
    echo "  Please create .env from .env.example and add your API keys:"
    echo "    cp .env.example .env"
    echo "    # Edit .env and add OPENAI_API_KEY"
    exit 1
fi

# Check if OPENAI_API_KEY is set
echo ""
echo "✓ Checking API keys..."
if grep -q "OPENAI_API_KEY=your-openai-key-here" .env; then
    echo ""
    echo "⚠️  WARNING: OpenAI API key not configured!"
    echo "  Edit .env and replace 'your-openai-key-here' with your actual key"
    echo "  The API will run but LLM calls will fail without a valid key"
    echo ""
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    echo "  API key configured ✓"
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo ""
    echo "→ Creating virtual environment..."
    python3 -m venv venv
    echo "  Virtual environment created ✓"
fi

# Activate virtual environment
echo ""
echo "→ Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo ""
echo "→ Installing dependencies (this may take a minute)..."
pip install -q --upgrade pip
pip install -q -r requirements.txt
echo "  Dependencies installed ✓"

# Create required directories
echo ""
echo "→ Setting up directories..."
mkdir -p data/{cache,reference_sets,results}
echo "  Directories ready ✓"

# Check if reference sets exist
if [ ! -f "data/reference_sets/validated_sets.json" ]; then
    echo ""
    echo "⚠️  Reference sets not found. Creating sample reference sets..."

    # Create a minimal reference set for testing
    cat > data/reference_sets/validated_sets.json << 'EOF'
{
  "reference_sets": [
    {
      "set_id": "test_set_1",
      "statements": {
        "1": "I would never purchase this product.",
        "2": "I am unlikely to purchase this product.",
        "3": "I might purchase this product.",
        "4": "I would likely purchase this product.",
        "5": "I would definitely purchase this product."
      }
    }
  ]
}
EOF
    echo "  Sample reference sets created ✓"
fi

# Start the API
echo ""
echo "═══════════════════════════════════════════════════════════"
echo "   Starting SSR API Server"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "  URL:     http://localhost:8000"
echo "  Docs:    http://localhost:8000/docs"
echo "  Health:  http://localhost:8000/health"
echo ""
echo "  Press Ctrl+C to stop the server"
echo ""

# Run the API
cd "$(dirname "$0")"
python3 -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

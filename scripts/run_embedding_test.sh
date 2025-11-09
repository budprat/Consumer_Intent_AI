#!/bin/bash
# Load environment variables from .env if it exists
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | grep -v '^$' | xargs)
fi

# Run the embedding test
python scripts/quick_embedding_test.py

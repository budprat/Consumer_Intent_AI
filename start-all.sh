#!/bin/bash

echo "Starting backend API on port 8000..."
cd /home/runner/workspace
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

# Wait for backend to start
sleep 3

echo "Starting frontend on port 5000..."
cd /home/runner/workspace/web-app
npm run dev &
FRONTEND_PID=$!

# Keep script running and forward signals
trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null" EXIT
wait

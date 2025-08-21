#!/bin/bash

set -e

cleanup() {
    echo "Shutting down llama.cpp server..."
    if [ -n "$LLAMA_PID" ]; then
        kill "$LLAMA_PID"
        wait "$LLAMA_PID" 2>/dev/null
    fi
    echo "Cleanup complete."
}

trap cleanup SIGINT SIGTERM

# -c context window size
# -ngl GPU layer count
echo "Starting llama.cpp server in the background..."
"$LLAMA_CPP_SERVER_PATH" \
    -m "$LLAMA_CPP_MODEL_PATH" \
    --host "$LLAMA_CPP_HOST" \
    --port "$LLAMA_CPP_PORT" \
    -c 4096 \
    -ngl 0 &

LLAMA_PID=$!
echo "Llama.cpp server started with PID: $LLAMA_PID"

echo "Waiting for llama.cpp server to be ready..."
sleep 5

echo "Starting promptmask-web application..."
promptmask-web
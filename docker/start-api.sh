#!/bin/sh
set -eu

# Qdrant local mode needs a writable lock/database directory. The source corpus
# remains read-only; this disposable runtime copy is never persisted.
rm -rf /tmp/qdrant-runtime
cp -a /app/data/qdrant /tmp/qdrant-runtime
export QDRANT_PATH=/tmp/qdrant-runtime

exec uvicorn api.main:app --host "${API_HOST:-0.0.0.0}" --port "${API_PORT:-8000}"

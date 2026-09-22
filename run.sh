#!/usr/bin/env bash
# Start the findme server, then open http://127.0.0.1:8000
set -euo pipefail
cd "$(dirname "$0")"
exec .venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 "$@"

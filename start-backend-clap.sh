#!/usr/bin/env bash
cd "$(dirname "$0")/backend"
SFX_USE_CLAP=1 python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000

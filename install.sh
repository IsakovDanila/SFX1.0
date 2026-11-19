#!/usr/bin/env bash
set -e
pip install -r backend/requirements.txt
python3 scripts/generate_demo_sounds.py
(cd frontend && npm install)
echo "Done! Run: ./start-backend.sh and ./start-frontend.sh"

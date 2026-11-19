#!/usr/bin/env bash
set -e
echo "[1/2] torch (CPU, >=2.6 required by transformers)..."
pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cpu
echo "[2/2] laion-clap..."
pip install -r backend/requirements-ml.txt
echo "Done! CLAP mode: ./start-backend-clap.sh"

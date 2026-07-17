#!/usr/bin/env bash
# exit on error
set -o errexit

echo "=== Installing Python dependencies ==="
pip install -r requirements.txt

echo "=== Installing Node dependencies & Building React frontend ==="
cd frontend
npm install
npm run build
cd ..

echo "=== Training ML models (baking into build) ==="
python scripts/train_models.py

echo "=== Build complete ==="

#!/bin/bash
# ============================================================
#  PROMETHEUS — Render Build Script
#  Forces pip to use prebuilt wheels — no compilation needed
#  Build time: ~2 min instead of ~15 min
# ============================================================

set -e

echo "🔧 Installing PROMETHEUS dependencies..."

pip install --upgrade pip

# Force prebuilt wheels for heavy packages (no compilation)
pip install \
  --only-binary=pandas \
  --only-binary=numpy \
  --only-binary=scikit-learn \
  --only-binary=xgboost \
  pandas==2.2.1 \
  numpy==1.26.4

# Install rest normally
pip install -r requirements-render.txt

echo "✅ Build complete"

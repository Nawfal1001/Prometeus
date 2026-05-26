#!/bin/bash
# ============================================================
#  PROMETHEUS — Render Build Script
#  Python 3.11 + prebuilt wheels only
# ============================================================
set -e

echo "🐍 Python version: $(python --version)"
echo "📦 Installing dependencies..."

pip install --upgrade pip

# Install everything from the Render-specific requirements
# All packages here have prebuilt wheels for Python 3.11
pip install -r requirements-render.txt

echo "✅ Build complete — $(pip list | wc -l) packages installed"

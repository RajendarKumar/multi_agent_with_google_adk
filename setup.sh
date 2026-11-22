#!/usr/bin/env bash
# ==============================================================================
# setup.sh — Initialize virtual environment, dependencies, and default .env
# ==============================================================================
set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

echo "=========================================================="
echo "📦 Setting up Responsive Agent Platform"
echo "=========================================================="

# 1. Discover Python 3.11+
if command -v python3.11 &>/dev/null; then
    PYTHON="python3.11"
elif command -v python3 &>/dev/null; then
    PYTHON="python3"
else
    echo "❌ Error: Python 3.11+ is required but not found in PATH."
    exit 1
fi

echo "1. Creating virtual environment with $($PYTHON --version)..."
$PYTHON -m venv .venv

echo "2. Upgrading pip, setuptools, and wheel..."
.venv/bin/pip install --upgrade pip setuptools wheel

echo "3. Installing project + dev dependencies..."
.venv/bin/pip install -e ".[dev]"

# 4. Initialize .env from .env.example if missing
if [ ! -f ".env" ] && [ -f ".env.example" ]; then
    echo "4. Creating initial .env from .env.example..."
    cp .env.example .env
fi

echo ""
echo "=========================================================="
echo "✅ Environment setup complete!"
echo "=========================================================="
echo "Next steps:"
echo "1. Configure your API keys in .env (GOOGLE_API_KEY, WEATHERAPI_KEY)"
echo "2. Launch the application with: ./scripts/restart.sh"
echo "=========================================================="

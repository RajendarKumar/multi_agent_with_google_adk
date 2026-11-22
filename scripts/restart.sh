#!/usr/bin/env bash
# ==============================================================================
# Responsive Agent — Clean Application Startup & Restart Script
# ==============================================================================
set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

echo "=========================================================="
echo "🔄 Starting / Restarting Responsive Agent Application"
echo "=========================================================="

# 1. Stop existing processes if running
echo "1. Checking and stopping existing server and worker processes..."
pkill -f "uvicorn api.main:app" 2>/dev/null || true
pkill -f "celery -A api.tasks.celery_app" 2>/dev/null || true
sleep 1

# 2. Check and initialize virtual environment if first-time run
if [ ! -f ".venv/bin/python" ] || [ ! -f ".venv/bin/uvicorn" ]; then
    echo "2. Virtual environment (.venv) not found. Initializing first-time setup..."
    PYTHON_BIN=""
    if command -v python3.11 &>/dev/null; then
        PYTHON_BIN="python3.11"
    elif command -v python3 &>/dev/null; then
        PYTHON_BIN="python3"
    else
        echo "❌ Error: Python 3.11+ is required but not found in PATH."
        exit 1
    fi

    echo "   • Creating virtual environment with $($PYTHON_BIN --version)..."
    $PYTHON_BIN -m venv .venv

    echo "   • Upgrading pip..."
    .venv/bin/pip install --upgrade pip

    echo "   • Installing project dependencies (this may take 1-2 minutes)..."
    .venv/bin/pip install -e ".[dev]"
    echo "   ✅ Virtual environment & dependencies installed successfully."
else
    echo "2. Virtual environment (.venv) verified."
fi

# 3. Check for .env file
if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        echo "   • Creating .env from .env.example..."
        cp .env.example .env
        echo "   ⚠️  Created default .env. Please configure your GOOGLE_API_KEY and WEATHERAPI_KEY."
    fi
fi

# 4. Check and start Docker infrastructure
echo "3. Ensuring Docker containers are running (Postgres, Qdrant, Redis)..."
if command -v docker &> /dev/null; then
    docker compose up postgres qdrant redis -d 2>/dev/null || docker-compose up postgres qdrant redis -d 2>/dev/null || echo "⚠️  Docker command failed or daemon not running."
fi

# 5. Database table initialization
echo "4. Initializing database tables..."
if [ -f ".venv/bin/python" ]; then
    .venv/bin/python -c "import asyncio; from api.database import init_db; asyncio.run(init_db())" 2>/dev/null || true
fi

# 6. Start Celery Ingestion Worker in background
echo "5. Starting background Celery ingestion worker..."
nohup .venv/bin/celery -A api.tasks.celery_app worker --loglevel=info --concurrency=2 > celery.log 2>&1 &
CELERY_PID=$!
echo "   → Celery worker running in background (PID: $CELERY_PID, logs: celery.log)"

# 7. Start FastAPI Server
echo "6. Starting FastAPI server on http://localhost:8000..."
echo "   → UI available at: http://localhost:8000/ui/"
echo "   → API documentation (Swagger) available at: http://localhost:8000/docs"
echo "   → Health check available at: http://localhost:8000/health"
echo "   → Press Ctrl+C to stop the server"
echo "=========================================================="
exec .venv/bin/uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

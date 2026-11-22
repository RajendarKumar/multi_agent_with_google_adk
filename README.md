# Responsive Agent Platform — Architecture & Operations Guide

> **Production-grade Multi-Agent RAG + Model Context Protocol (MCP) platform built on Google ADK (Agent Development Kit), Gemini 2.5 Flash / Lite, Qdrant Hybrid Search, and PostgreSQL.**

This document provides the complete, production-grade guide to set up, configure, run, and verify the platform end-to-end. Every command, path, and API payload reflects the **actual codebase implementation**.

---

## Table of Contents
1. [Platform Capabilities & Architecture Overview](#1-platform-capabilities--architecture-overview)
2. [Prerequisites & System Requirements](#2-prerequisites--system-requirements)
3. [Configuration & Environment Keys Guide (`.env`)](#3-configuration--environment-keys-guide-env)
4. [Step-by-Step Installation & Startup](#4-step-by-step-installation--startup)
5. [Interactive Web UI Walkthrough (5 Operational Tabs)](#5-interactive-web-ui-walkthrough-5-operational-tabs)
6. [Weather MCP Server Registration & Verification (Script, cURL & UI)](#6-weather-mcp-server-registration--verification-script-curl--ui)
7. [End-to-End Operational Verification Walkthroughs](#7-end-to-end-operational-verification-walkthroughs)
8. [Automated RAG Quality Gates Benchmark (`evals/eval_ragas.py`)](#8-automated-rag-quality-gates-benchmark-evalseval_ragaspy)
9. [Automated Test Suite (72 Tests, 100% Pass Rate)](#9-automated-test-suite-72-tests-100-pass-rate)
10. [Storage Architecture (Why 3 Databases?)](#10-storage-architecture-why-3-databases)
11. [System Verification Checklist](#11-system-verification-checklist)

---

## 1. Platform Capabilities & Architecture Overview

The platform implements an asynchronous multi-agent pipeline using Google ADK (`LlmAgent`, `SequentialAgent`, `ParallelAgent`, and `LoopAgent`):

```
                      ┌───────────────────────────────────────────────┐
                      │    FastAPI Application (SSE Streaming)        │
                      └──────────────────────┬────────────────────────┘
                                             │
                       ┌─────────────────────▼─────────────────────┐
                       │  ADK Root Orchestrator (gemini-3.5-flash) │
                       └─────────────────────┬─────────────────────┘
                                             │
      ┌──────────────────────────────────────┴──────────────────────────────────────┐
      │                                                                             │
┌─────▼────────────────────────┐                                      ┌─────────────▼───────────────┐
│     Summarizer Agent         │                                      │   Unified Search Pipeline   │
│  Structured Summarization    │                                      └─────────────┬───────────────┘
└──────────────────────────────┘                                                    │
                                                   ┌────────────────────────────────┴────────────────────────────────┐
                                                   │                                                                 │
                                     ┌─────────────▼───────────────┐                                   ┌─────────────▼───────────────┐
                                     │     Query Rewriter          │                                   │  Deterministic Reranker     │
                                     │ Standalone Query Generation │                                   │ Cross-Encoder (Math, <180ms)│
                                     └─────────────┬───────────────┘                                   └─────────────┬───────────────┘
                                                   │                                                                 │
                                     ┌─────────────▼───────────────┐                                   ┌─────────────▼───────────────┐
                                     │  Parallel Evidence Gatherer │                                   │ Grounded Synthesizer Loop   │
                                     │ ┌────────────┬────────────┐ │                                   │ LoopAgent (Max 2 Iterations)│
                                     │ │ Local RAG  │ Dynamic MCP│ │                                   │ Synthesizer + Verifier NLI  │
                                     │ │   Agent    │   Agent    │ │                                   └─────────────────────────────┘
                                     │ └─────┬──────┴─────┬──────┘ │
                                     └───────┼────────────┼────────┘
                                             │            │
                                             ▼            ▼
                                        Qdrant DB    Weather MCP
                                      (Dense+BM25)  (WeatherAPI.com)
```

### Core Implemented Capabilities:
1. **Google ADK Multi-Agent Hierarchy:** `orchestrator.py` wires a root coordinator with specialized sub-agents for query rewriting, parallel retrieval, cross-encoder reranking, grounded synthesis, and iterative hallucination verification (`LoopAgent`).
2. **Polymorphic Ingestion Engine:** Dynamic file-type router (`ingestion/parsers/router.py`) supporting **PDF, DOCX, TXT, and Markdown** with structured HTML table extraction and section-heading preservation.
3. **Automated Web Crawler with Freshness Schedules:** Scrapes web pages (`ingestion/parsers/web_parser.py`) and schedules automated background re-crawling by **minute, hour, day, or week** via APScheduler.
4. **Hybrid Dense + Sparse (BM25) Vector Retrieval:** Generates 3072-dimensional semantic dense vectors (`gemini-embedding-001`) and lexical BM25 sparse vectors, fused via Reciprocal Rank Fusion (RRF) in Qdrant.
5. **Deterministic Cross-Encoder Reranker:** Mathematical sequence-pair cross-attention ranking (`agents/reranker.py`) scoring query-chunk relevance in under 180ms without LLM token cost.
6. **Model Context Protocol (MCP) Server (`WeatherService`):** FastMCP JSON-RPC 2.0 server (`mcp_servers/weather/server.py`) with Stdio transport, externalized endpoints, and live queries to WeatherAPI.com.
7. **Dual-Tier Memory System:** Short-Term Memory (STM) conversation scratchpad in PostgreSQL (`DatabaseSessionService`) and Long-Term Memory (LTM) cross-session semantic storage.
8. **Dual-Tier Observability & Traces (PostgreSQL + Langfuse):** Telemetry architecture combining local audit persistence in PostgreSQL table `query_traces` with distributed LLM observability via **Langfuse** (tracking the root pipeline chain, query rewriter, parallel evidence gathering, deterministic reranker, LLM token consumption, and grounding verification).
9. **Automated RAG Benchmark Gates:** Curated 12-question benchmark (`evals/golden_dataset.json`) and automated scoring harness (`evals/eval_ragas.py`).

---

## 2. Prerequisites & System Requirements

Before running the application, ensure the host machine meets the following prerequisites:

### 1. Docker & Docker Compose (Required)
Docker is required to run the local backing services (PostgreSQL, Qdrant, and Redis).
- **Docker Engine:** `20.10+`
- **Docker Compose:** `v2.0+`
- **Verify Installation:**
  ```bash
  docker --version
  docker compose version
  ```
  *(Ensure the Docker daemon is active and running).*

### 2. Python Environment
- **Python:** `3.11.x`
- **Verify Installation:**
  ```bash
  python3 --version
  ```

### 3. API Keys & Credentials
- **`GOOGLE_API_KEY` (Required):** API key from [Google AI Studio](https://aistudio.google.com/) for Gemini models (`gemini-3.5-flash-lite`) and embeddings (`gemini-embedding-001`).
- **`WEATHERAPI_KEY` (Required for live Weather MCP):** Free or commercial API key from [weatherapi.com](https://www.weatherapi.com/).
- **`LANGFUSE_PUBLIC_KEY` & `LANGFUSE_SECRET_KEY` (Optional for Distributed Observability):** Public and secret API keys from [cloud.langfuse.com](https://cloud.langfuse.com) (or self-hosted) for distributed LLM tracing and multi-span latency/token visualization.

### 4. Standard CLI Utilities
- `curl`, `git`

---

## 3. Configuration & Environment Keys Guide (`.env`)

Generate `.env` from `.env.example`:
```bash
cp .env.example .env
```

Configuration variables fall into two distinct operational categories:
1. **External Cloud Provider Keys:** Hosted on external third-party infrastructure (Google Gemini, WeatherAPI, Langfuse). Because these require account-linked credentials, secret keys **cannot** be bundled in the repository and **must be supplied** by you.
2. **Local Infrastructure & Internal System Keys:** Backed by local Docker containers or internal deterministic algorithms. **Exact working values are provided out-of-the-box.**

---

### 3.1 External Cloud Provider Keys (Must Be User-Provided)

These keys authenticate with external cloud platforms. You must obtain active credentials from each respective provider:

| Key Name | Requirement | Example / Format | Where to Obtain | Purpose |
| :--- | :---: | :--- | :--- | :--- |
| **`GOOGLE_API_KEY`** | **Mandatory** | `AIzaSy...` (39 chars) | [Google AI Studio](https://aistudio.google.com/) | Powers Google ADK agents (`gemini-2.5-pro`, `gemini-2.5-flash`) and dense vector embeddings (`gemini-embedding-001`). |
| **`WEATHERAPI_KEY`** | **Mandatory for MCP** | `a1b2c3d4e5f6...` (32 chars) | [WeatherAPI.com](https://www.weatherapi.com/) | Powers the `WeatherService` MCP server for live real-time conditions, multi-day forecasts, and air quality tools. |
| **`LANGFUSE_PUBLIC_KEY`** | Optional | `pk-lf-...` | [Langfuse Cloud](https://cloud.langfuse.com/) | Project public API key for distributed multi-span trace visualization. |
| **`LANGFUSE_SECRET_KEY`** | Optional | `sk-lf-...` | [Langfuse Cloud](https://cloud.langfuse.com/) | Project secret API key for distributed multi-span trace visualization. |
| **`COHERE_API_KEY`** | Optional | `co-...` | [Cohere Platform](https://dashboard.cohere.com/) | Only required if switching from the internal deterministic cross-encoder to Cohere cloud reranking. |
| **`OPENAI_API_KEY`** | Optional | `sk-...` | [OpenAI Platform](https://platform.openai.com/) | Only required if switching vector embeddings to OpenAI (`text-embedding-3-large`). |

---

### 3.2 Local Infrastructure & System Keys (Pre-Configured Actual Values)

For all components running locally within Docker, on the host OS, or inside the application process, use these **exact actual values**:

| Component | Variable Name | Actual Value in Repo / Docker | Purpose / Description |
| :--- | :--- | :--- | :--- |
| **PostgreSQL (Async)** | `DATABASE_URL` | `postgresql+asyncpg://user:password@localhost:5432/responsive_agent` | Async SQLAlchemy connection to local PostgreSQL container. |
| **PostgreSQL (Sync)** | `SYNC_DATABASE_URL` | `postgresql://user:password@localhost:5432/responsive_agent` | Alembic migration connection to local PostgreSQL container. |
| **Qdrant Vector DB** | `QDRANT_URL` | `http://localhost:6333` | Local Qdrant hybrid vector store REST endpoint. |
| **Qdrant Collection** | `QDRANT_COLLECTION_NAME` | `responsive_agent_docs` | Target vector collection for dense (3072-dim) and BM25 sparse vectors. |
| **Redis Broker** | `REDIS_URL` | `redis://localhost:6379/0` | Local Redis container URL used by Celery async task workers. |
| **MCP AES Encryption Key** | `MCP_CREDENTIAL_ENCRYPTION_KEY` | `responsive-agent-mcp-aes256-key-32b!` | 32-byte AES key used by Fernet to encrypt registered MCP secrets at rest in PostgreSQL. |
| **MCP Scope Default** | `MCP_DEFAULT_READ_ONLY` | `true` | Restricts MCP tools to read-only execution by default for safe agent operation. |
| **MCP Server Limit** | `MCP_MAX_SERVERS_PER_QUERY` | `3` | Maximum number of dynamically selected MCP servers routed per user turn. |
| **Weather Base URL** | `WEATHERAPI_BASE_URL` | `https://api.weatherapi.com/v1` | WeatherAPI base service URL managed by `WeatherConfig`. |
| **Weather Current Endpoint** | `WEATHERAPI_CURRENT_ENDPOINT` | `/current.json` | Endpoint path for current weather conditions and air quality metrics. |
| **Weather Forecast Endpoint** | `WEATHERAPI_FORECAST_ENDPOINT` | `/forecast.json` | Endpoint path for multi-day weather forecasts. |
| **Weather Client Timeout** | `WEATHERAPI_TIMEOUT_SECONDS` | `10.0` | Timeout threshold in seconds for upstream WeatherAPI HTTP requests. |
| **Weather AQI Toggle** | `WEATHERAPI_AQI_ENABLED` | `true` | Appends `aqi=yes` to request air quality metrics with weather calls. |
| **Langfuse State** | `LANGFUSE_ENABLED` | `true` | Flag to enable/disable background trace dispatching to Langfuse. |
| **Langfuse Host** | `LANGFUSE_HOST` | `https://cloud.langfuse.com` | Langfuse ingestion host (or custom self-hosted endpoint). |
| **Embedding Provider** | `EMBEDDING_PROVIDER` | `google` | Active dense embedding backend (`google`, `openai`, `cohere`, `local`). |
| **Embedding Model** | `EMBEDDING_MODEL` | `gemini-embedding-001` | 3072-dimensional embedding model name. |
| **Embedding Dimension** | `EMBEDDING_DIMENSION` | `3072` | Vector length matching Qdrant collection vector configuration. |
| **Reranker Provider** | `RERANKER_PROVIDER` | `cohere` | Ranking backend (paired with internal mathematical Cross-Encoder fallback). |
| **Orchestrator Model** | `ORCHESTRATOR_MODEL` | `gemini-2.5-pro` | Primary LLM coordinating sub-agents and tool synthesis. |
| **Sub-Agent Model** | `FAST_MODEL` | `gemini-2.5-flash` | Lightweight LLM powering fast sub-agents (rewriter, summarizer). |
| **App Security Secret** | `APP_SECRET_KEY` | `responsive-agent-dev-secret-key-2026` | Secret salt for session cookie tokens and session ID signatures. |
| **Tenancy Mode** | `TENANCY_MODE` | `single` | Tenancy isolation mode (`single` or `multi`). |
| **Default Tenant** | `DEFAULT_TENANT_ID` | `default` | Default tenant identifier for multi-tenant data partitioning. |
| **UI Sources Toggle** | `SHOW_EVIDENCE_SOURCES` | `false` | Default UI state for displaying raw chunk evidence text blocks. |

---

### 3.3 Ready-to-Use `.env` File Blueprint

To configure the application, paste this exact block into `.env` and fill in your two external API keys (`GOOGLE_API_KEY` and `WEATHERAPI_KEY`):

```bash
# ==============================================================================
# 1. EXTERNAL CLOUD PROVIDER KEYS (Must be supplied by user)
# ==============================================================================
GOOGLE_API_KEY="your-google-api-key-here"
WEATHERAPI_KEY="your-weatherapi-key-here"

# Optional Cloud Observability (Langfuse)
LANGFUSE_ENABLED=true
LANGFUSE_PUBLIC_KEY="pk-lf-..."
LANGFUSE_SECRET_KEY="sk-lf-..."
LANGFUSE_HOST="https://cloud.langfuse.com"

# ==============================================================================
# 2. LOCAL INFRASTRUCTURE & BACKING SERVICES (Pre-configured actual values)
# ==============================================================================
# PostgreSQL (Docker port 5432)
DATABASE_URL="postgresql+asyncpg://user:password@localhost:5432/responsive_agent"
SYNC_DATABASE_URL="postgresql://user:password@localhost:5432/responsive_agent"

# Qdrant Vector Store (Docker port 6333)
QDRANT_URL="http://localhost:6333"
QDRANT_COLLECTION_NAME="responsive_agent_docs"

# Redis Task Broker (Docker port 6379)
REDIS_URL="redis://localhost:6379/0"

# ==============================================================================
# 3. MCP SERVER CONFIGURATION & SECURITY (Pre-configured actual values)
# ==============================================================================
MCP_CREDENTIAL_ENCRYPTION_KEY="responsive-agent-mcp-aes256-key-32b!"
MCP_DEFAULT_READ_ONLY=true
MCP_MAX_SERVERS_PER_QUERY=3

# Weather MCP Server Properties (WeatherAPI.com)
WEATHERAPI_BASE_URL="https://api.weatherapi.com/v1"
WEATHERAPI_CURRENT_ENDPOINT="/current.json"
WEATHERAPI_FORECAST_ENDPOINT="/forecast.json"
WEATHERAPI_TIMEOUT_SECONDS=10.0
WEATHERAPI_AQI_ENABLED=true

# ==============================================================================
# 4. AGENT MODELS & PIPELINE TUNING (Pre-configured actual values)
# ==============================================================================
APP_ENV="development"
APP_SECRET_KEY="responsive-agent-dev-secret-key-2026"
LOG_LEVEL="INFO"
TENANCY_MODE="single"
DEFAULT_TENANT_ID="default"
SHOW_EVIDENCE_SOURCES=false

# Google Gemini Agent Configurations
ORCHESTRATOR_MODEL="gemini-2.5-pro"
FAST_MODEL="gemini-2.5-flash"
SUMMARIZER_MODEL="gemini-2.5-flash"

# Embeddings & Reranking
EMBEDDING_PROVIDER="google"
EMBEDDING_MODEL="gemini-embedding-001"
EMBEDDING_DIMENSION=3072
RERANKER_PROVIDER="cohere"
RERANKER_TOP_N=8
RERANKER_MIN_SCORE=0.1
```

---

## 4. Step-by-Step Installation & Startup

The entire platform can be set up and launched in **two simple script executions**:

### Quick Start (2-Script Automated Setup)

```bash
# 1. Initialize Virtual Environment & Install Dependencies (Run once)
chmod +x ./setup.sh ./scripts/restart.sh
./setup.sh

# 2. Configure API Keys in .env
# Open .env and add your GOOGLE_API_KEY and WEATHERAPI_KEY

# 3. Launch Platform Services
./scripts/restart.sh

# 4. Register the Weather MCP Server (In a new terminal window)
.venv/bin/python scripts/register_weather_mcp.py
```

- **`./setup.sh`**: Detects Python 3.11+, creates `.venv`, upgrades pip/setuptools, installs all project dependencies (`pip install -e ".[dev]"`), and initializes `.env` from `.env.example`.
- **`./scripts/restart.sh`**: Ensures Docker containers are active (PostgreSQL, Qdrant, Redis), initializes database tables, starts the background Celery worker, and launches the FastAPI application on `http://localhost:8000`.
- **`scripts/register_weather_mcp.py`**: Invokes the REST registration endpoint (`POST /mcp/servers`), dynamically discovers the 3 tools (`get_air_quality`, `get_current_weather`, `get_weather_forecast`), and registers `WeatherService` into PostgreSQL.

---

### Step-by-Step Manual Installation & Startup (Alternative)

If you prefer executing each step manually rather than using the automated helper scripts:

#### Step 1: Initialize Virtual Environment & Install Dependencies
```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev]"
```

#### Step 2: Configure Environment Variables
```bash
cp .env.example .env
# Edit .env with your GOOGLE_API_KEY and WEATHERAPI_KEY
```

#### Step 3: Start Docker Infrastructure Containers
Launch PostgreSQL (port 5432), Qdrant (port 6333), and Redis (port 6379):
```bash
docker compose up postgres qdrant redis -d
```
*Verify all 3 containers are healthy:*
```bash
docker compose ps
```

#### Step 4: Start the Celery Ingestion Worker
In a terminal window (with `.venv` activated), start the background worker:
```bash
.venv/bin/celery -A api.tasks.celery_app worker --loglevel=info --concurrency=2
```

#### Step 5: Start the FastAPI Application Server
In another terminal window (with `.venv` activated), start the API server:
```bash
.venv/bin/uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

> [!NOTE]
> **Automatic Table Initialization:** You do not need to run manual database migrations. On startup, FastAPI's application lifespan automatically initializes all required PostgreSQL tables (`tenants`, `document_sources`, `mcp_servers`, `query_traces`) via `Base.metadata.create_all`.

#### Step 6: Verify Service Health
Run a cURL command against the health endpoint:
```bash
curl http://localhost:8000/health
```
*Expected Response:*
```json
{
  "status": "healthy",
  "active_refresh_jobs": 0,
  "tenant_id": "default"
}
```

#### Step 7: Register the Weather MCP Server
Register the Weather MCP Server through the running REST API:
```bash
.venv/bin/python scripts/register_weather_mcp.py
```
*Expected Output:*
```
📡 Registering WeatherService MCP Server via API endpoint: http://localhost:8000/mcp/servers
✅ API Response received (HTTP 200):
   • Server ID:         e7b87ef4-1afa-41da-b4fe-8435646e8bb1
   • Server Name:       WeatherService
   • Action Scope:      read_only
   • Tools Discovered:  3
   • Discovered Tools:  ['get_air_quality', 'get_current_weather', 'get_weather_forecast']
   • Message:           MCP server registered successfully
```

---

## 5. Interactive Web UI Walkthrough (5 Operational Tabs)

Open your browser and navigate to: **`http://localhost:8000/ui/`**

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│  ADK 2.0  Responsive Agent Platform                             Tenant ID: [default]       [API Online 🟢]             │
├──────────────────────────┬──────────────────────────┬──────────────────────────┬──────────────────────┬────────────────┤
│ 💬 Chat & Grounded Res...│ 📁 Knowledge Ingestion   │ 🛠️ MCP Toolset Registry │ ⚙️ Agent Config      │ 📊 Observability│
└──────────────────────────┴──────────────────────────┴──────────────────────────┴──────────────────────┴────────────────┘
```

The Web UI implements **5 distinct operational tabs** directly in the navigation bar:

1. **`💬 Chat & Grounded Research` (`#tab-chat`):**
   - **Header Controls:** Tenant selector with live connectivity badge (`API Online`), `📚 Show Sources` toggle switch (default: OFF), `➕ New Session` button, and `🗑️ Clear Chat` button.
   - **Real-Time Streaming:** Server-Sent Events (SSE) streaming with character-by-character token rendering from Gemini.
   - **Dynamic Live Breadcrumbs:** Animated status indicator (`#liveBreadcrumbs`) showing active sub-agent actions (`Analysing your query...`, `Calling WeatherService.get_current_weather`, `Reranking chunks`).
   - **Execution Context Sidebar:** Live metadata display showing active Session ID, Reranker algorithm (Deterministic Cross-Encoder), Memory mode (STM scratchpad + LTM fact bank), Verification loop, and inline `Show Sources in Chat` switch.
   - **Interactive Citation Inspector:** Click any inline citation badge in an assistant answer to inspect its exact retrieved text payload, source filename, page number, and rerank score.

2. **`📁 Knowledge Ingestion` (`#tab-ingest`):**
   - **Card 1 (`📄 Polymorphic Document Upload`):** Drag-and-drop file upload zone supporting `.pdf`, `.docx`, `.txt`, and `.md` files (single or batch upload).
   - **Card 2 (`🌐 Web URL Registration & Refresh Scheduler`):** Input target web documentation URL with configurable refresh intervals (`refreshValue`) and units: `Hour(s)`, `Minute(s)`, `Day(s)`, `Week(s)`.
   - **Card 3 (`📚 Ingested Knowledge Sources`):** Real-time source monitoring table with columns: `Source Name`, `Type`, `Status` (`queued`, `processing`, `indexed`, `failed`), `Chunks Indexed`, `Refresh Interval`, `Created At`, and `Actions` (delete button that removes database source record and cleans all associated vector points from Qdrant). Has a manual `🔄 Refresh List` button.

3. **`🛠️ MCP Toolset Registry` (`#tab-mcp`):**
   - **Header:** Title, description, and `➕ Register MCP Server` button that opens the registration modal.
   - **Registered Servers Data Table:** Columns: `Server Name`, `Transport` (`stdio` or `sse`), `Endpoint / Command`, `Action Scope` (`read_only`), `Status` (green ready badge), `Tools Discovered`, and `Actions` (`Test Connectivity` button and delete button).
   - **MCP Registration Modal (`#mcpModal`):** Complete registration form with fields: Server Name, Description, Transport Type (`streamable_http`, `sse`, `stdio`), Endpoint URL or Stdio Command, Action Scope Policy (`read_only`, `read,create`, `read,update`, `full`), Auth Type (`none`, `bearer`, `api_key`, `basic`), and Auth Secret (encrypted with AES Fernet at rest).

4. **`⚙️ Agent Configuration` (`#tab-config`):**
   - **Header:** `Zero-Redeploy Agent Configuration` with `💾 Save All Changes` button.
   - **Agent Config Grid (`#agentConfigContainer`):** Dynamically rendered agent cards for each agent configured in `config/agents.yaml` (`platform_orchestrator`, `local_rag_agent`, `dynamic_mcp_agent`, `query_rewriter_agent`, `grounded_synthesizer_agent`, `verifier_agent`, `content_summarizer_agent`).
   - **Interactive Property Controls:** Adjust Model Name, Temperature slider (0.0 to 1.0), Max Output Tokens, Instruction prompt textarea, and Enabled/Disabled toggle without restarting the server.

5. **`📊 Observability & Traces` (`#tab-traces`):**
   - **Header:** `Execution Traces & Observability Audit` with `🔄 Refresh Traces` button.
   - **Execution Traces Table:** Columns: `Trace ID`, `User Query`, `Rewritten Query (AI Model)`, `Total Latency` (ms), `Tokens (In/Out)`, `Grounding Status` (Pass/Fail badge), `Timestamp`, and `Details` button.
   - **Trace Drilldown Modal (`#traceModal`):** Detailed execution inspector displaying the rewritten query, latency breakdown, exact evidence chunks retrieved from Qdrant and MCP tools, grounding verifier verification results, and full synthesis payload.
   - **Distributed Observability:** Synchronized with **Langfuse** when `LANGFUSE_ENABLED=true` to provide end-to-end multi-span flame graphs in the cloud dashboard alongside local records.

---

## 6. Weather MCP Server Registration & Verification (Script, cURL & UI)

The Weather MCP server is implemented in `mcp_servers/weather/server.py` and powered by `mcp_servers/weather/client.py`. It integrates strictly with **WeatherAPI.com** with zero third-party fallback.

### Tools Exposed by WeatherService:
1. `get_current_weather(city: str)`: Returns temperature (°C), feels-like, wind speed (kph), direction, humidity, precipitation, UV index, and air quality (US EPA Index & PM2.5/PM10).
2. `get_weather_forecast(city: str, days: int = 3)`: Returns daily max/min temperatures, precipitation probabilities, and wind speeds up to 14 days.
3. `get_air_quality(city: str)`: Returns US EPA Index and PM2.5 / PM10 particulate levels.

---

### Method A: Automated Registration via Python Script (Recommended)
Execute the registration script while the platform is running:
```bash
.venv/bin/python scripts/register_weather_mcp.py
```
*Expected Output:*
```
📡 Registering WeatherService MCP Server via API endpoint: http://localhost:8000/mcp/servers
   Command: /path/to/.venv/bin/python /path/to/mcp_servers/weather/server.py --stdio

✅ API Response received (HTTP 200):
   • Server ID:         e7b87ef4-1afa-41da-b4fe-8435646e8bb1
   • Server Name:       WeatherService
   • Action Scope:      read_only
   • Tools Discovered:  3
   • Discovered Tools:  ['get_air_quality', 'get_current_weather', 'get_weather_forecast']
   • Message:           MCP server registered successfully

🎉 WeatherService successfully registered and validated end-to-end through the API!
```

---

### Method B: Registration via cURL REST API
Execute this command to register directly against the REST API:
```bash
curl -X POST http://localhost:8000/mcp/servers \
  -H "Content-Type: application/json" \
  -d '{
    "name": "WeatherService",
    "description": "Real-time meteorological intelligence server providing live conditions, multi-day forecasts, wind, and air quality indices via WeatherAPI.com.",
    "transport_type": "stdio",
    "stdio_command": ".venv/bin/python mcp_servers/weather/server.py --stdio",
    "action_scope": "read_only",
    "tenant_id": "default"
  }'
```
*Expected Response (HTTP 200):*
```json
{
  "id": "e7b87ef4-1afa-41da-b4fe-8435646e8bb1",
  "name": "WeatherService",
  "tools_discovered": 3,
  "tool_names": ["get_air_quality", "get_current_weather", "get_weather_forecast"],
  "action_scope": "read_only",
  "message": "MCP server registered successfully"
}
```

---

### Step 6.2: List Registered MCP Servers via cURL
```bash
curl http://localhost:8000/mcp/servers
```
*Expected Response:*
```json
[
  {
    "id": "e7b87ef4-1afa-41da-b4fe-8435646e8bb1",
    "name": "WeatherService",
    "description": "Real-time meteorological intelligence server providing live conditions, multi-day forecasts, wind, and air quality indices via WeatherAPI.com.",
    "transport_type": "stdio",
    "endpoint_url": null,
    "auth_type": "none",
    "action_scope": "read_only",
    "enabled": true,
    "tool_count": 3,
    "tools": ["get_air_quality", "get_current_weather", "get_weather_forecast"]
  }
]
```

---

### Step 6.3: Test / Re-Probe WeatherService Connectivity via cURL
Use the `id` returned from Step 6.1 / 6.2:
```bash
curl -X POST http://localhost:8000/api/admin/mcp/servers/e7b87ef4-1afa-41da-b4fe-8435646e8bb1/test
```
*Expected Response:*
```json
{
  "status": "connected",
  "tools_discovered": 3,
  "tools": ["get_air_quality", "get_current_weather", "get_weather_forecast"]
}
```

---

## 7. End-to-End Operational Verification Walkthroughs

### Walkthrough A: Polymorphic File Ingestion (Using Existing Docx)

#### Using cURL:
Ingest the existing documentation file from the repository (`docs/responsive_agent.docx`):
```bash
curl -X POST http://localhost:8000/api/ingest/file \
  -F "file=@docs/responsive_agent.docx"
```
*Expected Response:*
```json
{
  "source_id": "01ac8355-f64b-4b9d-bc21-24f11770d6b0",
  "filename": "responsive_agent.docx",
  "status": "queued",
  "message": "File accepted for ingestion"
}
```

#### Check Ingestion Status:
```bash
curl http://localhost:8000/api/ingest/sources
```
Wait a few seconds for Celery to finish parsing, chunking, and indexing. The status will transition from `queued` to `indexed` with chunk count populated.

#### Query the Ingested Document via cURL:
```bash
curl -N -X POST http://localhost:8000/api/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"message": "What is the primary role of the Dynamic MCP Agent described in the document?", "session_id": "verify-session-1"}'
```
*Expected Output:*
The model streams an SSE response citing `[1] responsive_agent.docx` explaining the dynamic tool discovery mechanism.

---

### Walkthrough B: Web URL Ingestion with Freshness Scheduling

#### Using cURL:
Ingest a web documentation page and set it to automatically re-crawl every 12 hours:
```bash
curl -X POST http://localhost:8000/api/ingest/url \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://docs.qdrant.tech/concepts/search/",
    "refresh_enabled": true,
    "refresh_value": 12,
    "refresh_unit": "hour"
  }'
```
*Expected Response:*
```json
{
  "source_id": "28fe654b-0d4a-468d-885b-68e030199e19",
  "url": "https://docs.qdrant.tech/concepts/search/",
  "status": "queued",
  "refresh_enabled": true,
  "refresh_value": 12,
  "refresh_unit": "hour",
  "message": "Web URL registered and queued for ingestion"
}
```

---

### Walkthrough C: Live Weather MCP Query (No Fallback, Pure WeatherAPI)

#### Using cURL:
Send a query asking for live real-time conditions:
```bash
curl -N -X POST http://localhost:8000/api/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"message": "What is the current temperature, wind speed, and air quality in Wellington, New Zealand?", "session_id": "verify-session-2"}'
```
*Expected Output:*
- SSE status events stream: `Query Rewriter` → `Dynamic MCP Agent: Calling WeatherService.get_current_weather` → `Reranking` → `Synthesizer`.
- Final answer reports real-time temperature (e.g. 15°C), southerly wind speed, and EPA air quality index, citing `[WeatherService]`.

---

### Walkthrough D: Dual RAG + MCP Orchestration Query
Ask a question that requires **both** static document knowledge from Qdrant and live dynamic tools from WeatherService:
```bash
curl -N -X POST http://localhost:8000/api/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"message": "Summarize how the Responsive Agent platform connects to external tools from the document, and check the current weather in London to confirm the tool is working.", "session_id": "verify-session-3"}'
```
*Expected Output:*
- `LocalRagAgent` retrieves chunk passages from `responsive_agent.docx`.
- `DynamicMcpAgent` invokes `WeatherService.get_current_weather(city="London")`.
- `DeterministicRerankerAgent` scores both evidence items.
- The Synthesizer outputs a single unified response citing both `[responsive_agent.docx]` and `[WeatherService]`.

---

### Walkthrough E: Inspect Query Traces & Langfuse Observability

The platform provides dual-tier observability: local audit records stored in PostgreSQL and live distributed tracing published to **Langfuse**.

#### 1. Local Database & REST Traces (PostgreSQL)
Retrieve execution traces captured in PostgreSQL via cURL or inspect them in the **📊 Observability & Traces** Web UI tab:
```bash
curl http://localhost:8000/api/admin/traces
```
*Expected Response:*
```json
[
  {
    "id": "trace-uuid-1234",
    "session_id": "verify-session-2",
    "user_query": "What is the current temperature...",
    "rewritten_query": "current temperature wind speed air quality Wellington New Zealand",
    "final_answer": "The current temperature in Wellington is 15.0°C...",
    "grounding_passed": true,
    "total_latency_ms": 1420,
    "input_tokens": 1250,
    "output_tokens": 180
  }
]
```

#### 2. Distributed Multi-Span Tracing (Langfuse)
When `LANGFUSE_ENABLED=true` and credentials (`LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`) are set in `.env`, `api/tracing.py` dispatches structured multi-span trace graphs to Langfuse asynchronously without adding latency to the user response:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ Root Span: agentic_chat_pipeline (type: chain)                              │
│ Tags: [tenant:default, routing:search] | Attributes: session_id, user_id    │
├─────────────────────────────────────────────────────────────────────────────┤
│ ├── Span 1: query_rewriter (type: span)                                     │
│ │   Input: raw query ──► Output: standalone contextualized query            │
│ │                                                                           │
│ ├── Span 2: evidence_gathering (type: retriever)                            │
│ │   Metadata: { rag_count: 5, mcp_count: 1 }                                │
│ │                                                                           │
│ ├── Span 3: deterministic_reranker (type: span)                             │
│ │   Metadata: { reranked_count: 5, cross_encoder_scores: {...} }            │
│ │                                                                           │
│ ├── Span 4: grounded_synthesizer (type: generation)                         │
│ │   Model: gemini-2.5-pro | Usage: { input_tokens, output_tokens, total }   │
│ │                                                                           │
│ └── Span 5: verifier_agent (type: verification check)                       │
│     Metadata: { grounding_passed: true }                                    │
└─────────────────────────────────────────────────────────────────────────────┘
```

- **Langfuse Cloud Dashboard:** Log into [cloud.langfuse.com](https://cloud.langfuse.com) to view end-to-end latency waterfalls, per-agent execution times, token cost accounting, and input/output payloads.
- **Fail-Safe Execution:** Langfuse dispatches run within guarded exception handlers; network or authentication errors with Langfuse will log a warning and never block chat generation.

---

## 8. Automated RAG Quality Gates Benchmark (`evals/eval_ragas.py`)

The platform includes a curated Golden Dataset with 12 reference questions across 9 functional categories (`evals/golden_dataset.json`).

Run the automated benchmark suite:
```bash
.venv/bin/python evals/eval_ragas.py
```

*Expected Output:*
```json
{
  "total_samples": 12,
  "citation_coverage_score": 1.0,
  "unanswerable_safety_score": 1.0,
  "categories_covered": [
    "pdf_rag",
    "web_rag",
    "mcp_query",
    "memory",
    "reranking",
    "verification",
    "unanswerable",
    "streaming",
    "observability"
  ]
}
```

- **`citation_coverage_score: 1.0`**: 100% of benchmark questions have verified expected sources mapped.
- **`unanswerable_safety_score: 1.0`**: Includes negative control questions (e.g. `eval_010` asking for private CEO data) to verify hallucination refusal safety.

---

## 9. Automated Test Suite (72 Tests, 100% Pass Rate)

Run the entire automated pytest suite:
```bash
.venv/bin/pytest -v tests/
```

*Expected Output:*
```
======================= 72 passed, 13 warnings in 25.25s =======================
```

### Test Suite Directory Breakdown:
- `tests/test_weather_mcp.py` (14 tests): WeatherConfig properties, WeatherAPI response normalization, missing key rejection, Swagger error handling (400, 401, 403, 404, 429, 500), tool schemas, tool execution, and REST API registration via `POST /mcp/servers`.
- `tests/test_mcp_integration.py` & `tests/test_mcp_policy.py` (9 tests): Fernet AES credential encryption, ADK McpToolset builder across Stdio/SSE transports, semantic tool routing, and read-only action scope enforcement.
- `tests/test_chat_orchestration.py` (4 tests): ADK agent graph construction, YAML configuration loading, RRF fusion blending, and cross-encoder scoring.
- `tests/test_parsers.py`, `tests/test_chunker.py`, `tests/test_docx_and_polymorphic.py` (17 tests): PDF Docling parser, web parser, Markdown table conversion, section heading preservation, and chunk overlap.
- `tests/test_api_chat.py`, `tests/test_api_ingest.py`, `tests/test_database_sqlite.py`, `tests/test_ui_and_admin_config.py` (28 tests): REST API endpoints, sync/SSE chat streaming, query traces, and SQLite CRUD.

---

## 10. Storage Architecture (Why 3 Databases?)

Each database handles a dedicated responsibility:

| Database | System Role | Port | What It Stores | Why It Cannot Be Replaced |
| :--- | :--- | :---: | :--- | :--- |
| **PostgreSQL** | **Relational ACID Store** | `5432` | Source metadata, MCP server configs, encrypted secrets, refresh schedules, query traces, and STM session state. | Requires strict relational integrity, ACID transactions, and foreign key cascades. |
| **Qdrant** | **Vector Search Engine** | `6333` | Dense semantic embeddings (3072-dim) + BM25 sparse vectors + chunk text payloads. | Dedicated approximate nearest neighbors (ANN) index supporting hybrid dense+sparse search and RRF fusion. |
| **Redis** | **In-Memory Message Broker** | `6379` | Celery asynchronous ingestion job queues and task coordination. | Microsecond FIFO task queue preventing heavy PDF parsing and embedding from blocking HTTP threads. |

---

## 11. System Verification Checklist

Use this checklist to verify complete platform functionality:
- [ ] **Docker Containers:** PostgreSQL (5432), Qdrant (6333), Redis (6379) running.
- [ ] **Weather MCP Server:** Registered via `POST /mcp/servers` or script; 3 tools discovered.
- [ ] **Document Ingestion:** Ingested `docs/responsive_agent.docx`; verified status `indexed` in sources list.
- [ ] **Chat Console:** Tested SSE streaming query at `http://localhost:8000/ui/`.
- [ ] **Dual Retrieval:** Cites both document and live WeatherAPI in a single response.
- [ ] **Observability:** Telemetry trace captured with latency and tokens in `http://localhost:8000/api/admin/traces`.
- [ ] **Automated Evals:** `.venv/bin/python evals/eval_ragas.py` reports 1.0 citation coverage and 1.0 safety score.
- [ ] **Automated Tests:** `.venv/bin/pytest -v tests/` completes with **72 passed, 0 failed**.

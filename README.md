# Leadership Intelligence Agent 🤖

A high-performance RAG (Retrieval-Augmented Generation) agent built with **FastAPI**, **LangGraph**, and **LangCache**. It helps leadership teams extract insights from both unstructured documents (strategy/risk) and structured datasets (numeric KPIs).

---

## 🚀 Quick Start

### 1. Prerequisites
Ensure you have `uv` installed ([Install uv](https://docs.astral.sh/uv/getting-started/installation/)).

### 2. Setup Dependencies
```bash
uv sync
```

### 3. Environment Configuration
Create a `.env` file from the example:
```bash
cp .env.example .env
```

Fill in the following API keys and endpoints:

| Service | Environment Variable | Console Link |
| :--- | :--- | :--- |
| **OpenAI** | `OPENAI_API_KEY` | [platform.openai.com](https://platform.openai.com) |
| **Qdrant** | `QDRANT_API_KEY`, `QDRANT_CLUSTER_ENDPOINT` | [cloud.qdrant.io](https://cloud.qdrant.io) |
| **LangCache** | `LANGCACHE_API_KEY`, `LANGCACHE_SERVER_URL` | [langcache.redis.io](https://langcache.redis.io) |
| **Cohere** | `CO_API_KEY` | [dashboard.cohere.com](https://dashboard.cohere.com) |
| **Anthropic** | `ANTHROPIC_API_KEY` | [console.anthropic.com](https://console.anthropic.com) |

### 4. Run Development Server
```bash
npm run dev
```
*The API will be available at `http://localhost:8000`*

---

## 🛠 Tech Stack

- **Framework**: [FastAPI](https://fastapi.tiapi.com/)
- **Agent Orchestration**: [LangGraph](https://langchain-ai.github.io/langgraph/)
- **Semantic Caching**: [LangCache](https://redis.io/langcache/) (Ultra-fast Redis-based query caching)
- **Vector Database**: [Qdrant](https://qdrant.tech/) (Hybrid search: Dense + Sparse)
- **Metadata & Data Storage**: **SQLite** (Stores ingested file registry and structured queryable tables)
- **Reranking**: **Cohere Rerank v3.5** (Top-tier relevance sorting)

## 📂 Project Structure

- `app/agent/`: Core agent logic, prompts, and tools.
- `app/api/`: FastAPI routes for ingestion and chat.
- `app/db/`: Database connection handlers (SQLite, Qdrant, Redis).
- `app/ingest/`: Processors for unstructured (PDF/Docx) and structured (CSV/XLSX) data.
- `main.py`: Application entry point and lifespan management.

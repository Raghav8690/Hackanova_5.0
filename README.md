# Research Literature Agent — Crawler & Citation Agents

Part of the **Code4Change Autonomous Research Literature Agent** system.

Two FastAPI-based agents that discover citations and crawl academic paper graphs using **LangChain + Gemini API**, storing structured data as JSON nodes in **SQLite**.

## Architecture

```
Paper Discovery Agent → [Crawling Agent] → [Citation Agent] → Semantic Scholar / CrossRef
                              ↓                    ↑
                        DFS Traversal ─────────────┘
                              ↓
                        SQLite (JSON nodes)
                              ↓
                     Overall Updation Agent
```

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and set your GEMINI_API_KEY
```

### 3. Run the server

```bash
uvicorn app.main:app --reload --port 8000
```

### 4. Open API docs

Visit [http://localhost:8000/docs](http://localhost:8000/docs) to see all endpoints.

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/citation/find` | Find top K citations for a paper |
| `POST` | `/crawler/crawl` | Start DFS crawl from paper links |
| `GET` | `/crawler/papers` | List all stored paper nodes |
| `GET` | `/crawler/papers/{id}` | Get a specific paper node |
| `GET` | `/crawler/tree/{id}` | Get citation tree from a root paper |

## Example Usage

### Find Citations

```bash
curl -X POST http://localhost:8000/citation/find \
  -H "Content-Type: application/json" \
  -d '{"paper_url": "https://arxiv.org/abs/2301.00001", "top_k": 10}'
```

### Start a Crawl

```bash
curl -X POST http://localhost:8000/crawler/crawl \
  -H "Content-Type: application/json" \
  -d '{"paper_links": ["https://arxiv.org/abs/2301.00001"], "max_depth": 2}'
```

## Project Structure

```
app/
├── main.py              # FastAPI entrypoint
├── config.py            # Settings from .env
├── database.py          # SQLite CRUD operations
├── models.py            # Pydantic schemas
├── utils.py             # ID generation, URL parsing
├── agents/
│   ├── citation_agent.py   # LangChain agent for citation discovery
│   └── crawling_agent.py   # DFS crawler engine
└── routers/
    ├── citation.py      # POST /citation/find
    └── crawler.py       # Crawl & paper retrieval endpoints
```

## Tech Stack

- **FastAPI** — API framework
- **LangChain + Gemini** — AI agent for citation ranking
- **Semantic Scholar API** — Academic paper search
- **CrossRef API** — Fallback paper search
- **SQLite** — JSON node storage

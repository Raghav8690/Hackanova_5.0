"""
Autonomous Research Literature Agent — Crawling & Citation Agents

FastAPI application entrypoint.
"""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.database import init_db
from app.routers import citation, crawler
from app.routers import citation, crawler, synthesizer
from app.routers import citation, crawler, synthesizer, discovery

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize resources on startup and cleanup on shutdown."""
    logger.info("Initializing database...")
    init_db()
    logger.info("Application started — Crawling & Citation Agents ready.")
    yield
    logger.info("Application shutting down.")


app = FastAPI(
    title="Research Literature Agent — Crawler & Citation",
    description=(
        "Autonomous agents for crawling academic papers, discovering citations, "
        "and storing structured paper data as JSON nodes in SQLite. "
        "Part of the Code4Change Autonomous Research Literature Agent system."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware (allows other agents to call these endpoints)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount routers
app.include_router(citation.router)
app.include_router(crawler.router)
app.include_router(synthesizer.router)
app.include_router(discovery.router)

@app.get("/", tags=["Health"])
async def root():
    """Health check endpoint."""
    return {
        "service": "Research Literature Agent — Crawler & Citation",
        "status": "running",
        "endpoints": {
            "citation": "/citation/find",
            "crawl": "/crawler/crawl",
            "papers": "/crawler/papers",
            "paper_detail": "/crawler/papers/{unique_id}",
            "citation_tree": "/crawler/tree/{unique_id}",
            "synthesize_update": "/synthesizer/update", 
            "synthesize_state": "/synthesizer/state",
            "docs": "/docs",
        },
    }

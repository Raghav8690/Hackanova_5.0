"""
Configuration settings loaded from environment variables.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Let's try loading from the app/ directory directly if it exists
env_path = Path(__file__).resolve().parent / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
else:
    load_dotenv()


class Settings:
    """Application settings loaded from .env file."""

    GEMINI_API_KEY_CITATION: str = os.getenv("GEMINI_API_KEY_CITATION", "AIzaSy_fake_key_fallback")
    GEMINI_API_KEY_SYNTHESIS: str = os.getenv("GEMINI_API_KEY_SYNTHESIS", "AIzaSy_fake_key_fallback")
    GEMINI_API_KEY_QUERY: str = os.getenv("GEMINI_API_KEY_QUERY", "AIzaSy_fake_key_fallback")
    DATABASE_PATH: str = os.getenv("DATABASE_PATH", "research_papers.db")
    MAX_DEPTH: int = int(os.getenv("MAX_DEPTH", "5"))
    TOP_K_CITATIONS: int = int(os.getenv("TOP_K_CITATIONS", "12"))

    # Other agent endpoints (external services)
    PAPER_ANALYZER_URL: str = os.getenv(
        "PAPER_ANALYZER_URL", "http://localhost:8001/analyze"
    )
    UPDATION_AGENT_URL: str = os.getenv(
        "UPDATION_AGENT_URL", "http://localhost:8002/update"
    )


settings = Settings()

# ──────────────────────────────────────────────────────────────────────────────
# NEW: Constants for Paper Discovery & Query Understanding Agents
# ──────────────────────────────────────────────────────────────────────────────

# Use Synthesis key for generic Gemini tasks if a general one isn't provided
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", settings.GEMINI_API_KEY_SYNTHESIS)
SEMANTIC_SCHOLAR_API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")

RATE_LIMIT_DELAY = 1.5          
MAX_RETRIES      = 3            
RETRY_BACKOFF    = 2.0          
TOP_K_PAPERS     = 3            
MAX_RESULTS_PER_QUERY = 20      

ARXIV_API_URL           = "http://export.arxiv.org/api/query"
PUBMED_ESEARCH_URL      = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
PUBMED_EFETCH_URL       = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
SEMANTIC_SCHOLAR_API_URL = "https://api.semanticscholar.org/graph/v1"
GOOGLE_SCHOLAR_URL      = "https://scholar.google.com/scholar"

DOMAIN_SYNONYMS = {
    "reinforcement learning": ["deep reinforcement learning", "Q-learning", "policy gradient", "actor-critic", "temporal difference learning", "model-based RL"],
    "smart grid": ["smart grids", "intelligent grid", "power grid", "microgrid", "electrical grid", "power systems"],
    "energy efficiency": ["energy optimization", "power optimization", "energy management", "energy conservation", "demand response"],
    "machine learning": ["deep learning", "neural networks", "supervised learning", "unsupervised learning", "transfer learning"],
    "natural language processing": ["NLP", "text mining", "language model", "transformer", "text classification", "sentiment analysis"],
    "computer vision": ["image recognition", "object detection", "image segmentation", "convolutional neural network", "CNN"],
    "optimization": ["mathematical optimization", "convex optimization", "metaheuristic", "genetic algorithm", "swarm intelligence"],
    "healthcare": ["clinical", "biomedical", "medical informatics", "health informatics", "electronic health records"],
    "robotics": ["autonomous systems", "robot learning", "motion planning", "robot control", "manipulation"],
    "internet of things": ["IoT", "sensor networks", "edge computing", "embedded systems", "cyber-physical systems"],
}
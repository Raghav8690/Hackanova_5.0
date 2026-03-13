"""
Configuration settings loaded from environment variables.
"""

import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    """Application settings loaded from .env file."""

    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
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

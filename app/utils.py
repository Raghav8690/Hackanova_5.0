"""
Utility functions for ID generation, URL parsing, and text sanitization.
"""

import re
import unicodedata
from urllib.parse import urlparse, unquote

import time
import logging
import functools
from typing import Callable, Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from app.config import RATE_LIMIT_DELAY, MAX_RETRIES, RETRY_BACKOFF

logging.basicConfig(level=logging.INFO, format="%(asctime)s │ %(name)-28s │ %(levelname)-7s │ %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("utils")

def generate_unique_id(year: int | str | None, author: str, title: str) -> str:
    """
    Generate a unique paper ID in the format: year_authorname_title
    
    Examples:
        generate_unique_id(2023, "John Smith", "Deep Learning for NLP")
        → "2023_johnsmith_deep_learning_for_nlp"
    """
    # Handle year
    year_str = str(year) if year else "unknown"

    # Slugify author name
    author_slug = _slugify(author.split(",")[0].strip() if "," in author else author)

    # Slugify title (truncate to keep ID manageable)
    title_slug = _slugify(title)
    # Keep only first 8 words of title for the ID
    title_words = title_slug.split("_")[:8]
    title_slug = "_".join(title_words)

    unique_id = f"{year_str}_{author_slug}_{title_slug}"
    return unique_id


def _slugify(text: str) -> str:
    """Convert text to a URL-friendly slug."""
    # Normalize unicode characters
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    # Lowercase
    text = text.lower()
    # Replace non-alphanumeric with underscores
    text = re.sub(r"[^a-z0-9]+", "_", text)
    # Remove leading/trailing underscores
    text = text.strip("_")
    return text


def extract_doi_from_url(url: str) -> str | None:
    """
    Try to extract a DOI from a URL.
    
    Supports formats like:
        - https://doi.org/10.1234/example
        - https://dx.doi.org/10.1234/example
        - https://arxiv.org/abs/2301.00001
    """
    if not url:
        return None

    # Direct DOI URL
    doi_match = re.search(r"(10\.\d{4,}/[^\s]+)", url)
    if doi_match:
        return doi_match.group(1).rstrip(".")

    return None


def extract_arxiv_id(url: str) -> str | None:
    """Extract arXiv paper ID from a URL."""
    match = re.search(r"arxiv\.org/(?:abs|pdf)/(\d+\.\d+(?:v\d+)?)", url)
    if match:
        return match.group(1)
    return None


def extract_pmcid(url: str) -> str | None:
    """Extract PubMed Central ID (PMCID) from a URL."""
    match = re.search(r"PMC(\d+)", url, re.I)
    if match:
        return f"PMC{match.group(1)}"
    return None


def sanitize_url(url: str) -> str:
    """Clean up and normalize a URL."""
    url = url.strip()
    url = unquote(url)
    # Remove trailing slashes
    url = url.rstrip("/")
    # Ensure scheme
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    return url


def extract_paper_info_from_url(url: str) -> dict:
    """
    Extract basic info from a paper URL.
    Returns dict with keys: source, paper_id, doi
    """
    info = {"source": "unknown", "paper_id": None, "doi": None}

    parsed = urlparse(url)
    domain = parsed.netloc.lower()

    if "arxiv.org" in domain:
        info["source"] = "arxiv"
        info["paper_id"] = extract_arxiv_id(url)
    elif "doi.org" in domain:
        info["source"] = "doi"
        info["doi"] = extract_doi_from_url(url)
    elif "pmc.ncbi.nlm.nih.gov" in domain or "pubmed.ncbi.nlm.nih.gov" in domain:
        info["source"] = "pubmed"
        info["paper_id"] = extract_pmcid(url)
    elif "semanticscholar.org" in domain:
        info["source"] = "semantic_scholar"
        # Extract paper ID from S2 URL
        parts = parsed.path.strip("/").split("/")
        if parts:
            info["paper_id"] = parts[-1]
    elif "scholar.google" in domain:
        info["source"] = "google_scholar"
    else:
        info["doi"] = extract_doi_from_url(url)

    return info


def truncate_text(text: str, max_length: int = 500) -> str:
    """Truncate text to a maximum length, adding ellipsis if needed."""
    if len(text) <= max_length:
        return text
    return text[: max_length - 3] + "..."


class RateLimiter:
    def __init__(self, delay: float = RATE_LIMIT_DELAY):
        self._delay = delay
        self._last_call: float = 0.0

    def wait(self) -> None:
        elapsed = time.time() - self._last_call
        if elapsed < self._delay:
            time.sleep(self._delay - elapsed)
        self._last_call = time.time()

_rate_limiters: dict[str, RateLimiter] = {}

def get_rate_limiter(source: str) -> RateLimiter:
    if source not in _rate_limiters:
        _rate_limiters[source] = RateLimiter()
    return _rate_limiters[source]

def create_session() -> requests.Session:
    session = requests.Session()
    retry_strategy = Retry(total=MAX_RETRIES, backoff_factor=RETRY_BACKOFF, status_forcelist=[429, 500, 502, 503, 504], allowed_methods=["GET", "POST"])
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({"User-Agent": "AutonomousResearchLitSystem/1.0 (mailto:research-system@example.com)"})
    return session

def retry_on_failure(max_retries: int = MAX_RETRIES, backoff: float = RETRY_BACKOFF):
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exc: Exception | None = None
            for attempt in range(1, max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as exc:
                    last_exc = exc
                    wait = backoff ** attempt
                    time.sleep(wait)
            raise last_exc
        return wrapper
    return decorator

_session = create_session()

def fetch_json(url: str, params: dict | None = None, headers: dict | None = None, source: str = "default") -> dict | list | None:
    get_rate_limiter(source).wait()
    try:
        resp = _session.get(url, params=params, headers=headers, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException:
        return None

def fetch_xml(url: str, params: dict | None = None, source: str = "default") -> str | None:
    get_rate_limiter(source).wait()
    try:
        resp = _session.get(url, params=params, timeout=30)
        resp.raise_for_status()
        return resp.text
    except requests.RequestException:
        return None
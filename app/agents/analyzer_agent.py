"""
Analyzer Agent — Fetches paper content and extracts structured analysis.

This agent:
1. Takes a paper URL (ArXiv, DOI, or direct URL)
2. Fetches the paper content (XML for ArXiv, PDF/HTML for others)
3. Uses Gemini 2.0 Flash to extract:
   - Methodology
   - Datasets Used
   - Atomic Claims
   - Limitations
   - Key Findings
4. Returns structured JSON analysis
"""

import logging
import re
import httpx
from typing import Optional
import json
import time

from google import genai
from google.genai import types
from app.config import settings
from app.models import PaperNode
from app.utils import extract_arxiv_id, extract_doi_from_url, sanitize_url, generate_unique_id, RATE_LIMIT_DELAY

logger = logging.getLogger(__name__)

# Configure Gemini API
# genai configuration is handled per client instance in the new SDK
client = genai.Client(api_key=settings.GEMINI_API_KEY_SYNTHESIS)

# HTTP client for fetching paper content
_http_client = httpx.Client(timeout=30.0, follow_redirects=True)


# ──────────────────────────────────────────────
# Paper Content Fetching
# ──────────────────────────────────────────────


def _fetch_arxiv_content(arxiv_id: str) -> str:
    """Fetch abstract and metadata from ArXiv for a given ArXiv ID."""
    try:
        # ArXiv API endpoint
        url = f"http://export.arxiv.org/api/query?id_list={arxiv_id}"
        response = _http_client.get(url, timeout=15.0)
        
        if response.status_code == 200:
            # Parse XML response to get abstract
            import xml.etree.ElementTree as ET
            root = ET.fromstring(response.text)
            
            # Extract namespace
            ns = {'atom': 'http://www.w3.org/2005/Atom'}
            
            # Get summary (abstract)
            summary_elem = root.find('.//atom:summary', ns)
            abstract = summary_elem.text.strip() if summary_elem is not None else ""
            
            # Get title
            title_elem = root.find('.//atom:title', ns)
            title = title_elem.text.strip() if title_elem is not None else ""
            
            # Get authors
            authors = []
            for author in root.findall('.//atom:author', ns):
                name_elem = author.find('atom:name', ns)
                if name_elem is not None:
                    authors.append(name_elem.text)
            
            # Get published date
            published_elem = root.find('.//atom:published', ns)
            year = None
            if published_elem is not None:
                year = int(published_elem.text[:4])
            
            return json.dumps({
                "title": title,
                "authors": authors,
                "year": year,
                "abstract": abstract,
                "source": "arxiv"
            })
        else:
            logger.warning("ArXiv API returned status %d", response.status_code)
            return json.dumps({"error": f"ArXiv returned {response.status_code}"})
            
    except Exception as e:
        logger.error("Error fetching from ArXiv: %s", e)
        return json.dumps({"error": str(e)})


def _fetch_generic_paper_content(url: str) -> str:
    """
    Fetch paper content from a generic URL using BeautifulSoup.
    Attempts to extract title, authors, abstract from HTML.
    """
    try:
        from bs4 import BeautifulSoup
        
        response = _http_client.get(url, timeout=15.0)
        if response.status_code != 200:
            logger.warning("Failed to fetch URL %s: status %d", url, response.status_code)
            return json.dumps({"error": f"HTTP {response.status_code}"})
        
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Try to extract title
        title = ""
        title_elem = soup.find("h1", class_=re.compile("title|heading", re.I))
        if title_elem:
            title = title_elem.get_text(strip=True)
        else:
            title_elem = soup.find("title")
            if title_elem:
                title = title_elem.get_text(strip=True)
        
        # Try to extract abstract
        abstract = ""
        abstract_elem = soup.find("div", class_=re.compile("abstract|summary", re.I))
        if abstract_elem:
            abstract = abstract_elem.get_text(strip=True)[:1000]
        
        # Try to extract authors
        authors = []
        author_elems = soup.find_all("span", class_=re.compile("author|creator", re.I))
        for elem in author_elems[:10]:  # Limit to first 10
            author_text = elem.get_text(strip=True)
            if author_text:
                authors.append(author_text)
        
        return json.dumps({
            "title": title,
            "authors": authors,
            "abstract": abstract,
            "source": "generic_url"
        })
        
    except ImportError:
        logger.warning("BeautifulSoup not installed, cannot parse HTML")
        return json.dumps({"error": "BeautifulSoup not installed"})
    except Exception as e:
        logger.error("Error fetching generic URL: %s", e)
        return json.dumps({"error": str(e)})


def _fetch_paper_content(paper_url: str) -> str:
    """
    Fetch paper content from various sources.
    Handles ArXiv, DOI URLs, and generic URLs.
    """
    url = sanitize_url(paper_url)
    
    # Try ArXiv first
    arxiv_id = extract_arxiv_id(url)
    if arxiv_id:
        logger.info("Detected ArXiv paper: %s", arxiv_id)
        return _fetch_arxiv_content(arxiv_id)
    
    # For other URLs, use generic fetcher
    logger.info("Fetching content from generic URL: %s", url)
    return _fetch_generic_paper_content(url)


# ──────────────────────────────────────────────
# Paper Analysis using Gemini
# ──────────────────────────────────────────────


ANALYSIS_PROMPT_TEMPLATE = """You are a research paper analyzer. Analyze the following paper content and extract structured information.

PAPER CONTENT:
{paper_content}

Extract and provide the following in JSON format:
1. methodology: List of methodologies, techniques, or approaches used
2. datasets_used: List of datasets, benchmarks, or data sources mentioned
3. atomic_claims: Key factual claims or findings (as a list)
4. limitations: Stated or implied limitations of the work
5. key_findings: Main contributions or findings

Return ONLY valid JSON, no other text. Example format:
{{
  "methodology": ["Method A", "Method B"],
  "datasets_used": ["Dataset1", "Dataset2"],
  "atomic_claims": ["Claim 1", "Claim 2"],
  "limitations": ["Limitation 1"],
  "key_findings": ["Finding 1", "Finding 2"]
}}

Be exact and concise. Extract only what is explicitly mentioned or clearly implied."""


def _analyze_with_gemini(paper_content: str, paper_url: str) -> dict:
    """
    Use Gemini to analyze paper content and extract structured information.
    """
    try:
        # Rate limiting
        time.sleep(RATE_LIMIT_DELAY)
        
        logger.info("Calling Gemma to analyze paper: %s", paper_url)
        
        prompt = ANALYSIS_PROMPT_TEMPLATE.format(paper_content=paper_content[:15000]) # Increased context for Gemma
        prompt += "\nReturn valid JSON."

        response = client.models.generate_content(
            model='gemma-3-27b-it',
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.1,
            )
        )
        
        # Parse the response
        response_text = response.text.strip()
        
        # Try to extract JSON from the response
        json_match = re.search(r'\{[\s\S]*\}', response_text)
        if not json_match:
            logger.warning("No JSON found in Gemini response for %s", paper_url)
            return {
                "methodology": [],
                "datasets_used": [],
                "atomic_claims": [],
                "limitations": [],
                "key_findings": []
            }
        
        analysis = json.loads(json_match.group())
        
        # Ensure all required keys exist
        analysis.setdefault("methodology", [])
        analysis.setdefault("datasets_used", [])
        analysis.setdefault("atomic_claims", [])
        analysis.setdefault("limitations", [])
        analysis.setdefault("key_findings", [])
        
        return analysis
        
    except Exception as e:
        logger.error("Gemini analysis error for %s: %s", paper_url, e)
        return {
            "methodology": [],
            "datasets_used": [],
            "atomic_claims": [],
            "limitations": [],
            "key_findings": []
        }


# ──────────────────────────────────────────────
# Main Paper Analyzer Function
# ──────────────────────────────────────────────


async def analyze_paper(paper_url: str, title: str = "", authors: list = None) -> dict:
    """
    Analyze a research paper by fetching its content and extracting structured information.
    
    Args:
        paper_url: URL of the paper (ArXiv, DOI, or direct link)
        title: Optional paper title (used if extraction fails)
        authors: Optional list of authors (used if extraction fails)
        
    Returns:
        Dictionary with analysis results including:
        - title
        - authors
        - year
        - abstract
        - methodology
        - datasets_used
        - atomic_claims
        - limitations
        - key_findings
        - url
    """
    logger.info("Analyzing paper: %s", paper_url)
    
    try:
        # Fetch paper content
        content_json = _fetch_paper_content(paper_url)
        content_data = json.loads(content_json)
        
        if "error" in content_data:
            logger.warning("Failed to fetch paper content: %s", content_data.get("error"))
            # Use provided metadata if available
            content_data = {
                "title": title or "Unknown Title",
                "authors": authors or [],
                "abstract": "",
                "year": None,
            }
        
        # Extract paper details
        paper_title = content_data.get("title", title or "Unknown Title")
        paper_authors = content_data.get("authors", authors or [])
        paper_year = content_data.get("year")
        paper_abstract = content_data.get("abstract", "")
        
        # Prepare content for analysis (combine title and abstract)
        analysis_content = f"Title: {paper_title}\n\nAuthors: {', '.join(paper_authors)}\n\nAbstract: {paper_abstract}"
        
        # Analyze with Gemini
        analysis = _analyze_with_gemini(analysis_content, paper_url)
        
        # Combine all information
        result = {
            "title": paper_title,
            "authors": paper_authors,
            "year": paper_year,
            "abstract": paper_abstract,
            "url": paper_url,
            "methodology": analysis.get("methodology", []),
            "datasets_used": analysis.get("datasets_used", []),
            "atomic_claims": analysis.get("atomic_claims", []),
            "limitations": analysis.get("limitations", []),
            "key_findings": analysis.get("key_findings", []),
        }
        
        logger.info(
            "Successfully analyzed paper: %s (methodology: %d, datasets: %d, claims: %d)",
            paper_url, 
            len(result.get("methodology", [])),
            len(result.get("datasets_used", [])),
            len(result.get("atomic_claims", []))
        )
        
        return result
        
    except Exception as e:
        logger.error("Paper analysis failed for %s: %s", paper_url, e)
        return {
            "title": title or "Unknown Title",
            "authors": authors or [],
            "year": None,
            "abstract": "",
            "url": paper_url,
            "methodology": [],
            "datasets_used": [],
            "atomic_claims": [],
            "limitations": [],
            "key_findings": [],
            "error": str(e)
        }

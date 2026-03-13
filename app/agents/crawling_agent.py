"""
Crawling Agent — Performs DFS traversal of the citation graph.

This agent:
1. Takes a list of paper links and a max_depth
2. For each paper, calls the Citation Agent to get top citations
3. Performs DFS traversal: for each citation, recursively finds more citations
4. Stores each paper as a JSON node in SQLite (via Paper Analyzer or mock data)
5. Notifies the Overall Updation Agent with each stored node's unique_id
6. Backtracks to parent and visits next sibling (standard DFS)
"""

import json
import logging
from typing import Optional

import httpx

from app.config import settings
from app.database import store_paper_node, paper_exists
from app.models import CitationInfo
from app.utils import generate_unique_id, sanitize_url
from app.agents.citation_agent import find_citations

logger = logging.getLogger(__name__)

# HTTP client for calling other agents
_http_client = httpx.AsyncClient(timeout=30.0, follow_redirects=True)


# ──────────────────────────────────────────────
# Helper Functions
# ──────────────────────────────────────────────


async def _call_paper_analyzer(paper_url: str) -> dict:
    """
    Call the Paper Analyzer Agent to get structured JSON data for a paper.
    MOCKED: This currently returns dummy data since the agent is not ready.
    """
    logger.info("Mocking Paper Analyzer response for %s", paper_url)
    # Generate some random/dummy analysis data
    return {
        "abstract": "This is a mock abstract for the paper found at " + paper_url,
        "methodology": ["Mocked Method A", "Mocked Method B"],
        "datasets_used": ["Dummy Dataset 1"],
        "key_claims": [
            "This paper demonstrates X.",
            "Mocked claim: The methodology outperforms baseline."
        ],
        "limitations": "MOCKED: Time and resource constraints.",
        "analysis_confidence": 0.85
    }


async def _notify_updation_agent(unique_id: str) -> bool:
    """
    Notify the Knowledge Synthesizer Agent with the unique_id of a stored paper.
    """
    try:
        # The user added the synthesizer endpoint locally
        synthesizer_url = "http://localhost:8000/synthesizer/update"
        response = await _http_client.post(
            synthesizer_url,
            json={"unique_id": unique_id},
            timeout=15.0,
        )
        if response.status_code == 200:
            logger.info("Notified Synthesizer Agent about %s", unique_id)
            return True
        else:
            logger.warning(
                "Synthesizer Agent returned %d for %s: %s",
                response.status_code, unique_id, response.text
            )
            return False
    except Exception as e:
        logger.warning(
            "Could not notify Synthesizer Agent for %s: %s", unique_id, e
        )
        return False


def _build_node_data(
    citation: CitationInfo,
    paper_url: str,
    analyzer_data: dict,
) -> dict:
    """
    Build the JSON data for a paper node.
    Merges citation metadata with Paper Analyzer output.
    """
    base_data = {
        "title": citation.title,
        "authors": citation.authors,
        "year": citation.year,
        "url": paper_url,
        "doi": citation.doi,
        "relevance_score": citation.relevance_score,
    }

    # Merge with analyzer data (analyzer takes precedence for overlapping keys)
    if analyzer_data:
        merged = {**base_data, **analyzer_data}
        # But keep our URL and citation info
        merged["source_url"] = paper_url
        merged["relevance_score"] = citation.relevance_score
        return merged

    return base_data


def _build_root_node_data(paper_url: str, analyzer_data: dict) -> dict:
    """Build node data for a root paper (no citation metadata yet)."""
    base_data = {
        "url": paper_url,
        "title": analyzer_data.get("title", paper_url),
        "authors": analyzer_data.get("authors", []),
        "year": analyzer_data.get("year"),
        "is_root": True,
    }
    if analyzer_data:
        merged = {**base_data, **analyzer_data}
        merged["source_url"] = paper_url
        return merged
    return base_data


# ──────────────────────────────────────────────
# DFS Crawling Engine
# ──────────────────────────────────────────────


async def crawl_papers(
    paper_links: list[str],
    max_depth: int = 5,
    top_k_citations: int = 12,
) -> dict:
    """
    Main crawling function. Performs recursive DFS traversal of the citation graph.
    Matches the user requirement: traverses to max_depth/leaf, then on backtrack
    calls the paper analyzer, saves to SQLite, and notifies the summary agent.
    """
    all_paper_ids: list[str] = []
    total_processed = 0
    visited: set[str] = set()

    async def run_dfs(
        current_url: str, 
        current_id: str, 
        parent_id: Optional[str], 
        current_depth: int,
        citation_metadata: dict = None
    ):
        nonlocal total_processed

        if current_id in visited:
            logger.debug("Already visited: %s", current_id)
            return
        
        # Mark as visited immediately to avoid cycles in recursion
        visited.add(current_id)
        logger.info("DFS Visiting: %s at depth %d", current_id, current_depth)

        # 1. Expand children recursively if we are not at max depth
        if current_depth < max_depth:
            try:
                citations = await find_citations(current_url, top_k_citations)
                if citations:
                    logger.info("Found %d citations for %s at depth %d", len(citations), current_id, current_depth)
                    for citation in citations:
                        child_url = sanitize_url(citation.url) if citation.url else ""
                        if not child_url:
                            continue
                            
                        # Crawler designs the node with unique id
                        child_author = citation.authors[0] if citation.authors else "unknown"
                        child_id = generate_unique_id(citation.year, child_author, citation.title)
                        
                        # Recursive DFS call to the child
                        await run_dfs(
                            current_url=child_url,
                            current_id=child_id,
                            parent_id=current_id,
                            current_depth=current_depth + 1,
                            citation_metadata=citation.model_dump()
                        )
                else:
                    logger.info("No citations found for %s", current_id)
            except Exception as e:
                logger.error("Citation Agent failed for %s: %s. Continuing...", current_url, e)

        # 2. Backtracking phase / Leaf condition
        # "as you reach the max_dept or the condition where the paper does not have citations you will send that paper url to the paper analyzer agent"
        logger.info("Processing data for node: %s (depth %d)", current_id, current_depth)
        
        analyzer_data = await _call_paper_analyzer(current_url)
        
        # Build node data
        if citation_metadata:
            # We already have title/author from the citation
            node_data = {**citation_metadata, **analyzer_data}
            node_data["source_url"] = current_url
        else:
            # Root node fallback
            node_data = _build_root_node_data(current_url, analyzer_data)

        # Check if already in DB to avoid double-processing
        if paper_exists(current_id):
            logger.debug("Already in DB: %s", current_id)
            all_paper_ids.append(current_id)
            return

        # "into the sqllite database with the unique id which we made intially"
        stored = store_paper_node(
            unique_id=current_id,
            url=current_url,
            data=node_data,
            parent_id=parent_id,
            depth=current_depth,
        )

        if stored:
            all_paper_ids.append(current_id)
            total_processed += 1
            logger.info("Stored node to SQLite: %s", current_id)

            # "after saving the data imideately send the unique id of the node to the agent name overall summary agent"
            await _notify_updation_agent(current_id)


    for paper_url in paper_links:
        paper_url = sanitize_url(paper_url)
        logger.info("=" * 60)
        logger.info("Starting recursive DFS crawl for root: %s", paper_url)
        logger.info("Max depth: %d | Top K: %d", max_depth, top_k_citations)
        logger.info("=" * 60)

        # Pre-fetch basic info for the root node to generate its ID properly
        analyzer_data = await _call_paper_analyzer(paper_url)
        root_title = analyzer_data.get("title", paper_url)
        root_author = analyzer_data.get("authors", ["unknown"])[0] if analyzer_data.get("authors") else "unknown"
        root_year = analyzer_data.get("year", "unknown")
        
        root_id = generate_unique_id(root_year, root_author, root_title)
        
        await run_dfs(paper_url, root_id, None, 0)

    summary = (
        f"DFS recursive crawl complete. Processed {total_processed} papers "
        f"from {len(paper_links)} root paper(s) with max_depth={max_depth}."
    )
    logger.info(summary)

    return {
        "status": "completed",
        "total_papers_processed": total_processed,
        "paper_ids": all_paper_ids,
        "message": summary,
    }

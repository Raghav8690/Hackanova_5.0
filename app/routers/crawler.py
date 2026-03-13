"""
FastAPI router for the Crawling Agent endpoint.
"""

import logging
from fastapi import APIRouter, HTTPException, Query

from app.models import (
    CrawlRequest,
    CrawlResponse,
    PaperNodeResponse,
    PaperListResponse,
)
from app.database import (
    get_paper_by_id,
    get_all_papers,
    get_paper_count,
    get_citation_tree,
)
from app.agents.crawling_agent import crawl_papers

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/crawler", tags=["Crawling Agent"])


@router.post("/crawl", response_model=CrawlResponse)
async def start_crawl(request: CrawlRequest):
    """
    Start a DFS crawl of the citation graph starting from the given paper links.
    
    - **paper_links**: List of paper URLs to start crawling from
    - **max_depth**: Maximum DFS depth (default: 5, max: 10)
    - **top_k_citations**: Citations to find per paper (default: 12)
    """
    logger.info(
        "Crawl request — papers: %d, max_depth: %d, top_k: %d",
        len(request.paper_links), request.max_depth, request.top_k_citations,
    )

    try:
        result = await crawl_papers(
            paper_links=request.paper_links,
            max_depth=request.max_depth,
            top_k_citations=request.top_k_citations,
        )

        return CrawlResponse(
            status=result["status"],
            total_papers_processed=result["total_papers_processed"],
            paper_ids=result["paper_ids"],
            message=result["message"],
        )

    except Exception as e:
        logger.error("Crawl endpoint error: %s", e)
        raise HTTPException(
            status_code=500,
            detail=f"Crawl failed: {str(e)}",
        )


@router.get("/papers", response_model=PaperListResponse)
async def list_papers(
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
):
    """
    List all stored paper nodes with pagination.
    """
    papers = get_all_papers(limit=limit, offset=offset)
    total = get_paper_count()
    return PaperListResponse(papers=papers, total=total)


@router.get("/papers/{unique_id}", response_model=PaperNodeResponse)
async def get_paper(unique_id: str):
    """
    Get a specific paper node by its unique ID.
    """
    paper = get_paper_by_id(unique_id)
    if not paper:
        raise HTTPException(status_code=404, detail=f"Paper not found: {unique_id}")
    return PaperNodeResponse(paper=paper)


@router.get("/tree/{unique_id}")
async def get_tree(unique_id: str):
    """
    Get the citation tree rooted at a given paper.
    Returns a nested JSON structure showing parent-child relationships.
    """
    tree = get_citation_tree(unique_id)
    if not tree:
        raise HTTPException(
            status_code=404,
            detail=f"Paper not found or no tree exists for: {unique_id}",
        )
    return tree

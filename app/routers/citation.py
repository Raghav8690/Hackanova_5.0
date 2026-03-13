"""
FastAPI router for the Citation Agent endpoint.
"""

import logging
from fastapi import APIRouter, HTTPException

from app.models import CitationRequest, CitationResponse
from app.agents.citation_agent import find_citations

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/citation", tags=["Citation Agent"])


@router.post("/find", response_model=CitationResponse)
async def find_paper_citations(request: CitationRequest):
    """
    Find the top K most relevant citations for a given research paper.
    
    - **paper_url**: URL of the paper (ArXiv, DOI, Semantic Scholar, etc.)
    - **top_k**: Number of top citations to return (default: 12)
    """
    logger.info(
        "Citation request — paper_url: %s, top_k: %d",
        request.paper_url, request.top_k,
    )

    try:
        citations = await find_citations(
            paper_url=request.paper_url,
            top_k=request.top_k,
        )

        return CitationResponse(
            paper_url=request.paper_url,
            citations=citations,
            total_found=len(citations),
            message=f"Found {len(citations)} relevant citations.",
        )

    except Exception as e:
        logger.error("Citation endpoint error: %s", e)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to find citations: {str(e)}",
        )

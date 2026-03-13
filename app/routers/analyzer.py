"""
FastAPI router for the Paper Analyzer Agent endpoint.
"""

import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional

from app.agents.analyzer_agent import analyze_paper

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/analyzer", tags=["Paper Analyzer Agent"])


# ──────────────────────────────────────────────
# Request/Response Models
# ──────────────────────────────────────────────


class AnalyzerRequest(BaseModel):
    """Request body for the Paper Analyzer endpoint."""
    paper_url: str = Field(..., description="URL of the paper to analyze")
    title: Optional[str] = Field(None, description="Optional paper title (used if extraction fails)")
    authors: Optional[list[str]] = Field(None, description="Optional list of authors")


class AnalysisResult(BaseModel):
    """Analysis result for a paper."""
    title: str
    authors: list[str] = Field(default_factory=list)
    year: Optional[int] = None
    abstract: str = ""
    url: str
    methodology: list[str] = Field(default_factory=list)
    datasets_used: list[str] = Field(default_factory=list)
    atomic_claims: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    key_findings: list[str] = Field(default_factory=list)
    error: Optional[str] = None


class AnalyzerResponse(BaseModel):
    """Response from the Paper Analyzer endpoint."""
    analysis: AnalysisResult
    message: str = ""


# ──────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────


@router.post("/analyze", response_model=AnalyzerResponse)
async def analyze_paper_endpoint(request: AnalyzerRequest):
    """
    Analyze a research paper by extracting structured information.
    
    - **paper_url**: URL of the paper (ArXiv, DOI, or direct link)
    - **title**: Optional paper title (used if extraction fails)
    - **authors**: Optional list of authors
    
    Returns structured analysis with:
    - Methodology
    - Datasets used
    - Atomic claims
    - Limitations
    - Key findings
    """
    logger.info("Analyzer request — URL: %s", request.paper_url)

    try:
        analysis_result = await analyze_paper(
            paper_url=request.paper_url,
            title=request.title,
            authors=request.authors or [],
        )

        return AnalyzerResponse(
            analysis=AnalysisResult(**analysis_result),
            message=f"Successfully analyzed paper: {analysis_result.get('title', 'Unknown')}",
        )

    except Exception as e:
        logger.error("Analyzer endpoint error: %s", e)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to analyze paper: {str(e)}",
        )

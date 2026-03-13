"""
FastAPI router for the Q&A Agent endpoint.
"""

import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional

from app.agents.qa_agent import answer_query

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/qa", tags=["Q&A Agent"])


# ──────────────────────────────────────────────
# Request/Response Models
# ──────────────────────────────────────────────


class QARequest(BaseModel):
    """Request body for the Q&A Agent endpoint."""
    query: str = Field(..., description="Research question to answer")
    topic: Optional[str] = Field(None, description="Optional topic/domain for context")


class QAResponse(BaseModel):
    """Response from the Q&A Agent endpoint."""
    query: str
    answer: str
    evidence: list[str] = Field(default_factory=list)
    relevant_papers: list[str] = Field(default_factory=list)
    contradictions: list[str] = Field(default_factory=list)
    research_gaps: list[str] = Field(default_factory=list)
    confidence: float = 0.5
    error: Optional[str] = None
    message: str = ""


# ──────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────


@router.post("/ask", response_model=QAResponse)
async def ask_question(request: QARequest):
    """
    Ask a research question based on the synthesized knowledge graph.
    
    - **query**: Research question to answer
    - **topic**: Optional topic/domain for context
    
    Returns:
    - answer: The answer to the query
    - evidence: Supporting findings
    - relevant_papers: Papers cited
    - contradictions: Known contradictions in literature
    - research_gaps: Identified research gaps
    - confidence: Confidence score (0-1)
    """
    logger.info("Q&A endpoint — query: %s", request.query)

    try:
        result = await answer_query(
            query=request.query,
            topic=request.topic,
        )

        return QAResponse(
            query=request.query,
            answer=result.get("answer", ""),
            evidence=result.get("evidence", []),
            relevant_papers=result.get("relevant_papers", []),
            contradictions=result.get("contradictions", []),
            research_gaps=result.get("research_gaps", []),
            confidence=float(result.get("confidence", 0.5)),
            error=result.get("error"),
            message=f"Query processed with confidence {result.get('confidence', 0.5):.2f}",
        )

    except Exception as e:
        logger.error("Q&A endpoint error: %s", e)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process query: {str(e)}",
        )

"""
Pydantic models for request/response schemas.
"""

from pydantic import BaseModel, Field
from typing import Optional


# ──────────────────────────────────────────────
# Citation Agent Models
# ──────────────────────────────────────────────

class CitationInfo(BaseModel):
    """A single citation found by the Citation Agent."""
    title: str = Field(..., description="Title of the cited paper")
    authors: list[str] = Field(default_factory=list, description="List of author names")
    url: str = Field(default="", description="URL or link to the paper")
    year: Optional[int] = Field(None, description="Publication year")
    relevance_score: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Relevance score between 0 and 1"
    )
    doi: Optional[str] = Field(None, description="DOI if available")


class CitationRequest(BaseModel):
    """Request body for the Citation Agent endpoint."""
    paper_url: str = Field(..., description="URL of the paper to find citations for")
    top_k: int = Field(default=12, ge=1, le=50, description="Number of top citations to return")


class CitationResponse(BaseModel):
    """Response from the Citation Agent endpoint."""
    paper_url: str
    citations: list[CitationInfo] = Field(default_factory=list)
    total_found: int = 0
    message: str = ""


# ──────────────────────────────────────────────
# Crawling Agent Models
# ──────────────────────────────────────────────

class CrawlRequest(BaseModel):
    """Request body for the Crawling Agent endpoint."""
    paper_links: list[str] = Field(
        ..., min_length=1,
        description="List of paper URLs to start crawling from"
    )
    max_depth: int = Field(
        default=5, ge=1, le=10,
        description="Maximum DFS traversal depth"
    )
    top_k_citations: int = Field(
        default=12, ge=1, le=50,
        description="Number of citations to fetch per paper"
    )


class CrawlResponse(BaseModel):
    """Response from the Crawling Agent endpoint."""
    status: str = "completed"
    total_papers_processed: int = 0
    paper_ids: list[str] = Field(default_factory=list)
    message: str = ""


# ──────────────────────────────────────────────
# Paper Node (stored in SQLite)
# ──────────────────────────────────────────────

class PaperNode(BaseModel):
    """A paper node stored in the database."""
    unique_id: str = Field(..., description="Unique ID: year_authorname_title")
    url: str = Field(default="", description="URL of the paper")
    data: dict = Field(default_factory=dict, description="Structured JSON data from Paper Analyzer")
    parent_id: Optional[str] = Field(None, description="Unique ID of the parent paper")
    depth: int = Field(default=0, description="DFS depth level")
    created_at: Optional[str] = None


class PaperNodeResponse(BaseModel):
    """Response for a single paper node."""
    paper: PaperNode


class PaperListResponse(BaseModel):
    """Response for listing multiple paper nodes."""
    papers: list[PaperNode] = Field(default_factory=list)
    total: int = 0


class CitationTreeNode(BaseModel):
    """A node in the citation tree for visualization."""
    unique_id: str
    url: str
    title: str = ""
    depth: int = 0
    children: list["CitationTreeNode"] = Field(default_factory=list)


# Allow self-referencing model
CitationTreeNode.model_rebuild()



# ──────────────────────────────────────────────
# Knowledge Synthesis Agent Models
# ──────────────────────────────────────────────

class SynthesisRequest(BaseModel):
    """Request to update global knowledge using a specific paper node."""
    unique_id: str = Field(..., description="Unique ID of the paper to synthesize")

class ThematicCluster(BaseModel):
    theme: str
    nodes: list[str] = Field(description="List of unique_ids belonging to this theme")
    summary: str

class ContradictionSide(BaseModel):
    claim: str
    nodes: list[str] = Field(description="List of unique_ids supporting this claim")

class Contradiction(BaseModel):
    issue: str
    side_a: ContradictionSide
    side_b: ContradictionSide
    resolution_status: str

class TimelineEvent(BaseModel):
    year: int
    event: str
    impact: str

class GlobalKnowledgeState(BaseModel):
    topic_overview: str
    thematic_clusters: list[ThematicCluster]
    contradiction_matrix: list[Contradiction]
    timeline: list[TimelineEvent]
    research_gaps: list[str]
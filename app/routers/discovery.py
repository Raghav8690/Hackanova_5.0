from fastapi import APIRouter
from app.models import QueryRequest, QueryAnalysisResponse, PaperDiscoveryRequest, PaperDiscoveryResponse
from app.agents.query_agent import QueryUnderstandingAgent, QueryAnalysis
from app.agents.paper_discovery_agent import PaperDiscoveryAgent

router = APIRouter(prefix="/discovery", tags=["Query & Discovery"])

@router.post("/analyze-query", response_model=QueryAnalysisResponse)
async def analyze_query(request: QueryRequest):
    agent = QueryUnderstandingAgent()
    analysis = agent.analyze(request.query)
    return QueryAnalysisResponse(analysis=analysis.to_dict())

@router.post("/discover-papers", response_model=PaperDiscoveryResponse)
async def discover_papers(request: PaperDiscoveryRequest):
    agent = PaperDiscoveryAgent()
    analysis_obj = QueryAnalysis(**request.analysis)
    results = agent.discover(analysis_obj)
    return PaperDiscoveryResponse(result=results)
from fastapi import APIRouter, HTTPException, BackgroundTasks
from app.database import get_paper_by_id
from app.models import SynthesisRequest
from app.agents.synthesizer import update_global_knowledge_task, read_global_state

router = APIRouter(
    prefix="/synthesizer",
    tags=["Knowledge Synthesizer"]
)

@router.post("/update")
async def trigger_synthesis(request: SynthesisRequest, background_tasks: BackgroundTasks):
    """
    Triggers an update to the global knowledge file using an existing paper node.
    Does NOT modify the node in the database.
    """
    # 1. Go to the database and fetch the node
    node = get_paper_by_id(request.unique_id)
    
    if not node:
        raise HTTPException(status_code=404, detail=f"Paper node {request.unique_id} not found in DB.")

    # 2. Trigger the agent in the background so the API doesn't hang
    background_tasks.add_task(update_global_knowledge_task, node)
    
    return {
        "status": "Synthesis initiated",
        "message": f"Agent is updating global knowledge using {request.unique_id} in the background.",
        "unique_id": request.unique_id
    }

@router.get("/state")
async def get_global_state():
    """Returns the current living knowledge file."""
    return read_global_state()
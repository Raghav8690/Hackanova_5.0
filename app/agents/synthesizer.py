import os
import json
import logging
from google import genai
from google.genai import types
from app.models import GlobalKnowledgeState, PaperNode
from app.config import settings

logger = logging.getLogger(__name__)

# Ensure your GEMINI_API_KEY is in your .env file
client = genai.Client(api_key=settings.GEMINI_API_KEY_SYNTHESIS)

GLOBAL_STATE_FILE = "global_knowledge_state.json"

def read_global_state() -> dict:
    """Read the current global state, or initialize an empty one."""
    if not os.path.exists(GLOBAL_STATE_FILE):
        return GlobalKnowledgeState(
            topic_overview="Initial state. Awaiting paper nodes.",
            thematic_clusters=[],
            contradiction_matrix=[],
            timeline=[],
            research_gaps=[]
        ).model_dump()
    with open(GLOBAL_STATE_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)

def write_global_state(state: dict):
    """Overwrite the global state file."""
    with open(GLOBAL_STATE_FILE, 'w', encoding='utf-8') as f:
        json.dump(state, f, indent=2, ensure_ascii=False)

def update_global_knowledge_task(node: PaperNode):
    """
    Background task: Reads the current global state, extracts minimal info 
    from the new node, and asks Gemini to update the state file.
    """
    current_state = read_global_state()
    
    # TOKEN OPTIMIZATION: Extract only the core elements from the Paper Analyzer's output
    # Adjust these keys based on what your Paper Analyzer actually outputs in node.data!
    minimal_node_data = {
        "id": node.unique_id,
        "title": node.data.get("title", "Unknown Title"),
        "year": node.data.get("year", "Unknown"),
        "summary": node.data.get("summary", node.data.get("abstract", "")),
        "key_findings": node.data.get("key_findings", []),
        "future_work": node.data.get("future_work", node.data.get("limitations", ""))
    }

    prompt = f"""
    You are an expert academic synthesizer. Update the CURRENT GLOBAL KNOWLEDGE STATE 
    by integrating the findings from the NEW PAPER NODE. 
    
    Instructions:
    1. Merge the paper's findings into existing thematic clusters or create a new one. Add the paper's ID ({node.unique_id}) to the 'nodes' list.
    2. If this paper contradicts any existing claims in the topic_overview or clusters, log it in the contradiction_matrix.
    3. Add significant milestones to the timeline.
    4. Append new gaps to research_gaps.
    
    CURRENT GLOBAL STATE:
    {json.dumps(current_state, indent=2)}

    NEW PAPER NODE:
    {json.dumps(minimal_node_data, indent=2)}
    """

    prompt += "\nReturn the updated state as a valid JSON object matching the GlobalKnowledgeState schema."

    try:
        logger.info(f"Triggering Gemma Synthesis for {node.unique_id}...")
        response = client.models.generate_content(
            model='gemma-3-27b-it', # Bypassing Gemini 429s
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.1,
            ),
        )
        
        # Extract JSON from potential markdown code blocks
        text = response.text
        if "```json" in text:
            text = text.split("```json")[-1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[-1].split("```")[0].strip()
            
        updated_state_dict = json.loads(text)
        write_global_state(updated_state_dict)
        logger.info(f"Successfully updated global state file using {node.unique_id}")
        
    except Exception as e:
        logger.error(f"Failed to synthesize node {node.unique_id}: {e}")
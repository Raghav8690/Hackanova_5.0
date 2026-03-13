"""
Q&A Agent — Answers research questions based on the global knowledge state.

This agent:
1. Reads the global_knowledge_state.json file (synthesized research knowledge)
2. Takes user queries and identifies relevant papers/topics
3. Uses Gemini to synthesize answers based on the knowledge graph
4. Provides citations to the papers used in the answer
"""

import logging
import json
import time
from pathlib import Path
from typing import Optional
import re

import google.generativeai as genai
from app.config import settings, RATE_LIMIT_DELAY

logger = logging.getLogger(__name__)

# Configure Gemini API
genai.configure(api_key=settings.GEMINI_API_KEY_QUERY)


# ──────────────────────────────────────────────
# Knowledge State Management
# ──────────────────────────────────────────────


def _get_knowledge_state_path() -> Path:
    """Get the path to the global knowledge state JSON file."""
    # Try current directory first
    current_path = Path("global_knowledge_state.json")
    if current_path.exists():
        return current_path
    
    # Try parent directory
    parent_path = Path("..") / "global_knowledge_state.json"
    if parent_path.exists():
        return parent_path
    
    # Default
    return Path("global_knowledge_state.json")


def _load_knowledge_state() -> dict:
    """Load and parse the global knowledge state JSON file."""
    try:
        knowledge_path = _get_knowledge_state_path()
        logger.info("Loading knowledge state from: %s", knowledge_path)
        
        with open(knowledge_path, "r") as f:
            state = json.load(f)
        
        return state
    except FileNotFoundError:
        logger.warning("Knowledge state file not found: %s", _get_knowledge_state_path())
        return {}
    except json.JSONDecodeError:
        logger.error("Failed to parse knowledge state JSON")
        return {}
    except Exception as e:
        logger.error("Error loading knowledge state: %s", e)
        return {}


def _extract_relevant_context(knowledge_state: dict, query: str) -> str:
    """
    Extract relevant information from knowledge state based on query.
    Includes thematic clusters, timeline, and research gaps.
    """
    context_parts = []
    
    # Add overview
    if "topic_overview" in knowledge_state:
        context_parts.append(f"Overview: {knowledge_state['topic_overview']}")
    
    # Add thematic clusters
    if "thematic_clusters" in knowledge_state:
        clusters = knowledge_state["thematic_clusters"]
        context_parts.append("\nResearch Themes:")
        for cluster in clusters:
            theme = cluster.get("theme", "Unknown")
            summary = cluster.get("summary", "")
            nodes = cluster.get("nodes", [])
            context_parts.append(f"  - {theme}: {summary} (nodes: {', '.join(nodes[:3])})")
    
    # Add timeline
    if "timeline" in knowledge_state:
        timeline = knowledge_state["timeline"]
        context_parts.append("\nTimeline of Key Events:")
        for event in timeline[:5]:  # Limit to first 5 events
            year = event.get("year", "Unknown")
            event_text = event.get("event", "")
            impact = event.get("impact", "")
            context_parts.append(f"  - {year}: {event_text} ({impact})")
    
    # Add research gaps
    if "research_gaps" in knowledge_state:
        gaps = knowledge_state["research_gaps"]
        context_parts.append("\nResearch Gaps:")
        for gap in gaps[:5]:  # Limit to first 5 gaps
            context_parts.append(f"  - {gap}")
    
    # Add contradictions if any
    if "contradiction_matrix" in knowledge_state:
        contradictions = knowledge_state["contradiction_matrix"]
        if contradictions:
            context_parts.append("\nKey Contradictions in Literature:")
            for contra in contradictions[:3]:  # Limit to first 3
                context_parts.append(f"  - {contra}")
    
    return "\n".join(context_parts)


# ──────────────────────────────────────────────
# Q&A Engine
# ──────────────────────────────────────────────


QA_PROMPT_TEMPLATE = """You are an expert research assistant with access to a comprehensive knowledge graph of research papers on {topic}.

KNOWLEDGE BASE CONTEXT:
{knowledge_context}

RESEARCH QUESTION:
{query}

Based on the knowledge base provided above, answer the question comprehensively. Include:
1. Direct answer to the question
2. Key evidence or findings from the research  
3. Relevant papers/nodes that support your answer (list by their unique_ids if available)
4. Any contradictions or debates in the literature
5. Remaining research gaps related to this question

Format your response as JSON:
{{
  "answer": "Main answer text here",
  "evidence": ["Supporting finding 1", "Supporting finding 2"],
  "relevant_papers": ["paper_id_1", "paper_id_2"],
  "contradictions": ["Contradiction 1 about X and Y"],
  "research_gaps": ["Gap 1", "Gap 2"],
  "confidence": 0.8
}}

Return ONLY valid JSON, no other text."""


async def answer_query(query: str, topic: Optional[str] = None) -> dict:
    """
    Answer a research question using the global knowledge state.
    
    Args:
        query: Research question to answer
        topic: Optional topic name (for context)
        
    Returns:
        Dictionary with:
        - answer: Main answer text
        - evidence: Supporting findings
        - relevant_papers: Paper IDs cited
        - contradictions: Known contradictions
        - research_gaps: Identified gaps
        - confidence: Confidence score
        - error: Error message if any
    """
    logger.info("Q&A query: %s", query)
    
    try:
        # Load knowledge state
        knowledge_state = _load_knowledge_state()
        
        if not knowledge_state:
            logger.warning("Knowledge state is empty")
            return {
                "answer": "The knowledge base appears to be empty. Please run the crawling agent first to populate the research knowledge.",
                "evidence": [],
                "relevant_papers": [],
                "contradictions": [],
                "research_gaps": [],
                "confidence": 0.0,
                "error": "Empty knowledge state"
            }
        
        # Extract relevant context
        context = _extract_relevant_context(knowledge_state, query)
        
        # Prepare prompt
        prompt = QA_PROMPT_TEMPLATE.format(
            topic=topic or "research papers",
            knowledge_context=context,
            query=query
        )
        
        # Rate limiting
        time.sleep(RATE_LIMIT_DELAY)
        
        # Call Gemini
        model = genai.GenerativeModel("gemini-2.0-flash")
        logger.info("Calling Gemini for Q&A")
        response = model.generate_content(prompt)
        
        response_text = response.text.strip()
        
        # Extract JSON from response
        json_match = re.search(r'\{[\s\S]*\}', response_text)
        if not json_match:
            logger.warning("No JSON found in Gemini response")
            return {
                "answer": response_text,
                "evidence": [],
                "relevant_papers": [],
                "contradictions": [],
                "research_gaps": [],
                "confidence": 0.5
            }
        
        result = json.loads(json_match.group())
        
        # Ensure all required fields exist
        result.setdefault("answer", "Unable to answer the query")
        result.setdefault("evidence", [])
        result.setdefault("relevant_papers", [])
        result.setdefault("contradictions", [])
        result.setdefault("research_gaps", [])
        result.setdefault("confidence", 0.5)
        
        logger.info(
            "Successfully answered query with confidence: %s",
            result.get("confidence", 0)
        )
        
        return result
        
    except Exception as e:
        logger.error("Q&A error: %s", e)
        return {
            "answer": f"Error processing query: {str(e)}",
            "evidence": [],
            "relevant_papers": [],
            "contradictions": [],
            "research_gaps": [],
            "confidence": 0.0,
            "error": str(e)
        }

"""
Citation Agent — Uses LangChain + Gemini to find top relevant citations for a paper.

This agent:
1. Takes a paper URL and extracts its metadata
2. Searches Semantic Scholar API for references and citing papers
3. Falls back to CrossRef if Semantic Scholar fails
4. Uses Gemini to rank citations by relevance
5. Returns the top K most relevant citations
"""

import json
import logging
import re
import httpx
from typing import Optional

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

from app.config import settings, SEMANTIC_SCHOLAR_API_KEY
from app.models import CitationInfo
from app.utils import extract_arxiv_id, extract_doi_from_url, extract_pmcid, sanitize_url

logger = logging.getLogger(__name__)

# HTTP client for API calls
_http_client = httpx.Client(timeout=30.0, follow_redirects=True)

SEMANTIC_SCHOLAR_API = "https://api.semanticscholar.org/graph/v1"
CROSSREF_API = "https://api.crossref.org/works"

SEMANTIC_SCHOLAR_HEADERS = {"x-api-key": SEMANTIC_SCHOLAR_API_KEY} if SEMANTIC_SCHOLAR_API_KEY else {}

def _s2_get(endpoint: str, params: dict) -> httpx.Response:
    """Helper for Semantic Scholar GET requests with retries for 429."""
    for attempt in range(3):
        response = _http_client.get(endpoint, params=params, headers=SEMANTIC_SCHOLAR_HEADERS)
        if response.status_code == 429:
            time.sleep(2 ** attempt)
            continue
        return response
    return response

import time


# ──────────────────────────────────────────────
# Tool Functions (used by the LangChain agent)
# ──────────────────────────────────────────────


@tool
def search_semantic_scholar(query: str) -> str:
    """
    Search Semantic Scholar for papers matching a query.
    Returns a JSON string with paper details including title, authors, year, 
    citations, references, and URLs. Use this to find citations for a research paper.
    """
    try:
        # First try paper search
        response = _s2_get(
            f"{SEMANTIC_SCHOLAR_API}/paper/search",
            params={
                "query": query,
                "limit": 20,
                "fields": "title,authors,year,url,externalIds,citationCount,referenceCount,abstract",
            },
        )

        if response.status_code == 200:
            data = response.json()
            papers = data.get("data", [])
            results = []
            for p in papers:
                authors = [a.get("name", "") for a in p.get("authors", [])]
                external_ids = p.get("externalIds", {}) or {}
                doi = external_ids.get("DOI", "")
                paper_url = p.get("url", "")
                if doi and not paper_url:
                    paper_url = f"https://doi.org/{doi}"
                results.append({
                    "title": p.get("title", ""),
                    "authors": authors,
                    "year": p.get("year"),
                    "url": paper_url,
                    "doi": doi,
                    "citation_count": p.get("citationCount", 0),
                    "abstract": (p.get("abstract") or "")[:300],
                })
            return json.dumps(results, ensure_ascii=False)
        else:
            return json.dumps({"error": f"Semantic Scholar returned {response.status_code}"})

    except Exception as e:
        logger.error("Semantic Scholar search error: %s", e)
        return json.dumps({"error": str(e)})


@tool
def get_paper_references(paper_id: str) -> str:
    """
    Get references (papers cited BY a given paper) from Semantic Scholar.
    paper_id can be a Semantic Scholar paper ID, DOI, ArXiv ID (e.g. 'ArXiv:2301.00001'),
    or a URL. Returns JSON list of referenced papers.
    """
    try:
        response = _s2_get(
            f"{SEMANTIC_SCHOLAR_API}/paper/{paper_id}/references",
            params={
                "limit": 50,
                "fields": "title,authors,year,url,externalIds,citationCount,abstract",
            },
        )

        if response.status_code == 200:
            data = response.json()
            refs = data.get("data", [])
            results = []
            for ref in refs:
                cited = ref.get("citedPaper", {})
                if not cited or not cited.get("title"):
                    continue
                authors = [a.get("name", "") for a in cited.get("authors", [])]
                external_ids = cited.get("externalIds", {}) or {}
                doi = external_ids.get("DOI", "")
                paper_url = cited.get("url", "")
                if doi and not paper_url:
                    paper_url = f"https://doi.org/{doi}"
                results.append({
                    "title": cited.get("title", ""),
                    "authors": authors,
                    "year": cited.get("year"),
                    "url": paper_url,
                    "doi": doi,
                    "citation_count": cited.get("citationCount", 0),
                    "abstract": (cited.get("abstract") or "")[:300],
                })
            return json.dumps(results, ensure_ascii=False)
        else:
            return json.dumps({"error": f"API returned {response.status_code}"})

    except Exception as e:
        logger.error("Get references error: %s", e)
        return json.dumps({"error": str(e)})


@tool
def get_paper_citations(paper_id: str) -> str:
    """
    Get citations (papers that CITE a given paper) from Semantic Scholar.
    paper_id can be a Semantic Scholar paper ID, DOI, ArXiv ID, or URL.
    Returns JSON list of citing papers.
    """
    try:
        response = _s2_get(
            f"{SEMANTIC_SCHOLAR_API}/paper/{paper_id}/citations",
            params={
                "limit": 50,
                "fields": "title,authors,year,url,externalIds,citationCount,abstract",
            },
        )

        if response.status_code == 200:
            data = response.json()
            cits = data.get("data", [])
            results = []
            for cit in cits:
                citing = cit.get("citingPaper", {})
                if not citing or not citing.get("title"):
                    continue
                authors = [a.get("name", "") for a in citing.get("authors", [])]
                external_ids = citing.get("externalIds", {}) or {}
                doi = external_ids.get("DOI", "")
                paper_url = citing.get("url", "")
                if doi and not paper_url:
                    paper_url = f"https://doi.org/{doi}"
                results.append({
                    "title": citing.get("title", ""),
                    "authors": authors,
                    "year": citing.get("year"),
                    "url": paper_url,
                    "doi": doi,
                    "citation_count": citing.get("citationCount", 0),
                    "abstract": (citing.get("abstract") or "")[:300],
                })
            return json.dumps(results, ensure_ascii=False)
        else:
            return json.dumps({"error": f"API returned {response.status_code}"})

    except Exception as e:
        logger.error("Get citations error: %s", e)
        return json.dumps({"error": str(e)})


@tool
def search_crossref(query: str) -> str:
    """
    Search CrossRef API for papers matching a query. This is a fallback 
    when Semantic Scholar doesn't return enough results. Returns JSON list of papers.
    """
    try:
        response = _http_client.get(
            CROSSREF_API,
            params={
                "query": query,
                "rows": 20,
                "select": "DOI,title,author,published-print,URL,is-referenced-by-count",
            },
        )

        if response.status_code == 200:
            data = response.json()
            items = data.get("message", {}).get("items", [])
            results = []
            for item in items:
                title = item.get("title", [""])[0] if item.get("title") else ""
                authors = []
                for a in item.get("author", []):
                    name = f"{a.get('given', '')} {a.get('family', '')}".strip()
                    if name:
                        authors.append(name)
                doi = item.get("DOI", "")
                year = None
                pub_date = item.get("published-print", {}).get("date-parts", [[]])
                if pub_date and pub_date[0]:
                    year = pub_date[0][0]
                results.append({
                    "title": title,
                    "authors": authors,
                    "year": year,
                    "url": f"https://doi.org/{doi}" if doi else "",
                    "doi": doi,
                    "citation_count": item.get("is-referenced-by-count", 0),
                })
            return json.dumps(results, ensure_ascii=False)
        else:
            return json.dumps({"error": f"CrossRef returned {response.status_code}"})

    except Exception as e:
        logger.error("CrossRef search error: %s", e)
        return json.dumps({"error": str(e)})


@tool
def get_paper_metadata_by_url(url: str) -> str:
    """
    Resolve a paper URL to a title and DOI/ID.
    This helps identify a paper when direct ID lookup fails.
    """
    try:
        # Try searching by URL in Semantic Scholar
        response = _s2_get(
            f"{SEMANTIC_SCHOLAR_API}/paper/search",
            params={"query": url, "limit": 1, "fields": "title,authors,year,externalIds"},
        )
        if response.status_code == 200:
            data = response.json().get("data", [])
            if data:
                return json.dumps(data[0])
        
        # Try CrossRef
        response = _http_client.get(CROSSREF_API, params={"query": url, "rows": 1})
        if response.status_code == 200:
            items = response.json().get("message", {}).get("items", [])
            if items:
                item = items[0]
                return json.dumps({
                    "title": item.get("title", [""])[0],
                    "doi": item.get("DOI", ""),
                    "year": item.get("published-print", {}).get("date-parts", [[None]])[0][0]
                })
        
        return json.dumps({"error": "Metadata not found for URL"})
    except Exception as e:
        return json.dumps({"error": str(e)})


# Tools dictionary to map tool names to functions
TOOLS = {
    "search_semantic_scholar": search_semantic_scholar,
    "get_paper_references": get_paper_references,
    "get_paper_citations": get_paper_citations,
    "search_crossref": search_crossref,
    "get_paper_metadata_by_url": get_paper_metadata_by_url,
}


# ──────────────────────────────────────────────
# LangChain Agent Setup
# ──────────────────────────────────────────────

CITATION_SYSTEM_PROMPT = """You are a Citation Discovery Agent for academic research.

Your task is to find the most relevant citations for a given research paper. You have access to 
academic search tools (Semantic Scholar and CrossRef).

WORKFLOW:
1. First, try to identify the paper using its URL (extract ArXiv ID or DOI if possible)
2. If direct ID lookup (PMCID, DOI, ArXiv) fails with 404, use get_paper_metadata_by_url to resolve the URL.
3. Use the revealed title/DOI to search for the paper and its citations.
4. Use get_paper_references to find papers cited BY this paper
5. Use get_paper_citations to find papers that CITE this paper  
6. If needed, use search_semantic_scholar or search_crossref with the paper's title/keywords
5. Combine all results, remove duplicates, and rank by relevance

RANKING CRITERIA (in order of importance):
- Direct relevance to the original paper's topic
- Citation count (higher = more established)
- Recency (more recent = better for cutting-edge topics)
- Whether the paper is cited BY or CITES the original (both directions are valuable)

OUTPUT FORMAT:
Return a JSON array of the top citations. Each citation should have:
- title: paper title
- authors: list of author names  
- url: link to the paper
- year: publication year
- relevance_score: 0.0 to 1.0 score indicating relevance
- doi: DOI string if available

IMPORTANT:
- Always return valid JSON
- Remove any duplicates (same title or DOI)
- Include at least the number of citations requested (top_k)
- If you cannot find enough citations, return what you have with a note
"""


def _build_paper_identifier(paper_url: str) -> str:
    """Build a Semantic Scholar compatible paper identifier from a URL."""
    url = sanitize_url(paper_url)

    # Try ArXiv ID
    arxiv_id = extract_arxiv_id(url)
    if arxiv_id:
        return f"ArXiv:{arxiv_id}"

    # Try DOI
    doi = extract_doi_from_url(url)
    if doi:
        return f"DOI:{doi}"

    # Try PMC
    pmcid = extract_pmcid(url)
    if pmcid:
        return f"PMCID:{pmcid}"

    # Fall back to URL itself
    return f"URL:{url}"


# ──────────────────────────────────────────────
# Main Citation Discovery Function
# ──────────────────────────────────────────────


async def find_citations(paper_url: str, top_k: int = 12) -> list[CitationInfo]:
    """
    Find the top K most relevant citations for a given paper.
    
    Args:
        paper_url: URL of the research paper
        top_k: Number of citations to return (default: 12)
        
    Returns:
        List of CitationInfo objects
    """
    # Use Gemma-3 to avoid Gemini rate limits
    llm = ChatGoogleGenerativeAI(
        model="gemma-3-27b-it",
        google_api_key=settings.GEMINI_API_KEY_CITATION,
        temperature=0.1,
    )
    
    # Bind tools directly
    llm_with_tools = llm.bind_tools(list(TOOLS.values()))

    paper_id = _build_paper_identifier(paper_url)

    prompt_input = f"""Find the top {top_k} most relevant citations for this research paper:

Paper URL: {paper_url}
Paper Identifier: {paper_id}

Please search for both:
1. Papers that this paper REFERENCES (backward citations)
2. Papers that CITE this paper (forward citations)

Return the top {top_k} most relevant citations as a JSON array.
Each entry needs: title, authors (list), url, year, relevance_score (0-1), doi (or null).
Return ONLY the JSON array, no other text."""

    
    messages = [
        SystemMessage(content=CITATION_SYSTEM_PROMPT),
        HumanMessage(content=prompt_input),
    ]

    try:
        # Simple action loop (up to 5 iterations)
        for i in range(5):
            response = llm_with_tools.invoke(messages)
            messages.append(response)
            
            # If no tools called, we're done
            if not response.tool_calls:
                break
                
            # Execute tools
            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                
                logger.info(f"Agent calling tool: {tool_name} with args: {tool_args}")
                
                try:
                    tool_func = TOOLS[tool_name]
                    tool_result = tool_func.invoke(tool_args)
                except Exception as e:
                    tool_result = f"Error executing tool: {e}"
                    
                messages.append(ToolMessage(
                    content=str(tool_result),
                    tool_call_id=tool_call["id"]
                ))
            
        output_text = messages[-1].content


        # Parse the JSON from the agent's output
        citations = _parse_citations_from_output(output_text, top_k)
        logger.info(
            "Found %d citations for %s", len(citations), paper_url
        )
        return citations

    except Exception as e:
        logger.error("Citation agent error for %s: %s", paper_url, e)
        # Fallback: try direct API call without the agent
        return await _fallback_citation_search(paper_url, top_k)


def _parse_citations_from_output(output: str, top_k: int) -> list[CitationInfo]:
    """Parse citation list from the agent's text output."""
    # Clean up Markdown JSON blocks if present
    text = output
    if "```json" in text:
        text = text.split("```json")[-1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[-1].split("```")[0].strip()

    # Try to find JSON array in the output if standard removal didn't work
    json_match = re.search(r"\[[\s\S]*\]", text)
    if not json_match:
        logger.warning("No JSON array found in agent output")
        return []

    try:
        raw_citations = json.loads(json_match.group())
    except json.JSONDecodeError:
        logger.warning("Failed to parse JSON from agent output: %s", text[:100])
        return []

    citations = []
    for item in raw_citations[:top_k]:
        if not isinstance(item, dict):
            continue
        try:
            citation = CitationInfo(
                title=item.get("title", "Unknown"),
                authors=item.get("authors", []),
                url=item.get("url", ""),
                year=item.get("year"),
                relevance_score=min(max(float(item.get("relevance_score", 0.5)), 0.0), 1.0),
                doi=item.get("doi"),
            )
            citations.append(citation)
        except Exception as e:
            logger.warning("Skipping malformed citation: %s", e)
            continue

    return citations


async def _fallback_citation_search(paper_url: str, top_k: int) -> list[CitationInfo]:
    """Fallback: search directly via Semantic Scholar without the LLM agent."""
    paper_id = _build_paper_identifier(paper_url)
    citations = []

    async def _safe_get_refs(pid):
        try:
            res = get_paper_references.invoke(pid)
            if "error" in res:
                return None
            return json.loads(res) if isinstance(res, str) else res
        except Exception:
            return None

    try:
        # Try direct lookup
        refs = await _safe_get_refs(paper_id)
        
        # If direct lookup fails, try metadata resolution (S2 or CrossRef)
        if not refs:
            meta_json = get_paper_metadata_by_url.invoke(paper_url)
            meta = json.loads(meta_json)
            if "title" in meta:
                # Try S2 search first
                search_res = search_semantic_scholar.invoke(meta["title"])
                search_data = json.loads(search_res)
                if isinstance(search_data, list) and search_data:
                    best_match = search_data[0]
                    resolved_id = None
                    if best_match.get("doi"): resolved_id = f"DOI:{best_match['doi']}"
                    elif best_match.get("url"): resolved_id = f"URL:{best_match['url']}"
                    if resolved_id:
                        refs = await _safe_get_refs(resolved_id)
                
                # If still no refs, try CrossRef
                if not refs:
                    cr_res = search_crossref.invoke(meta["title"])
                    cr_data = json.loads(cr_res)
                    if isinstance(cr_data, list) and cr_data:
                        # CrossRef search gives us some papers, but usually not the references for the input paper itself
                        # However, we can use the top result as a "fallback" if we can't find anything else
                        for r in cr_data[:top_k]:
                            citations.append(CitationInfo(
                                title=r.get("title", ""),
                                authors=r.get("authors", []),
                                url=r.get("url", ""),
                                year=r.get("year"),
                                relevance_score=0.4,
                                doi=r.get("doi"),
                            ))

        if not citations and isinstance(refs, list):
            for r in refs[:top_k]:
                citations.append(CitationInfo(
                    title=r.get("title", ""),
                    authors=r.get("authors", []),
                    url=r.get("url", ""),
                    year=r.get("year"),
                    relevance_score=0.5,
                    doi=r.get("doi"),
                ))
    except Exception as e:
        logger.error("Fallback citation search error: %s", e)

    return citations[:top_k]

import asyncio
import logging
import app.agents.crawling_agent
from app.agents.crawling_agent import crawl_papers
from app.database import get_db, init_db
from app.models import CitationInfo

logging.basicConfig(level=logging.INFO)

async def mock_find_citations(paper_url, top_k):
    # Mock returning 2 citations for any paper
    return [
        CitationInfo(title=f"Mock Citation 1 of {paper_url[-5:]}", authors=["Alice Researcher"], url=f"{paper_url}/cit1", year=2024, relevance_score=0.9),
        CitationInfo(title=f"Mock Citation 2 of {paper_url[-5:]}", authors=["Bob Scientist"], url=f"{paper_url}/cit2", year=2023, relevance_score=0.8)
    ]

# Apply monkeypatch
app.agents.crawling_agent.find_citations = mock_find_citations

async def main():
    # Ensure DB is ready
    init_db()
    
    # Run the crawl with a very small depth/top_k for quick testing
    print("Starting test crawl on sample paper...")
    result = await crawl_papers(
        paper_links=["https://arxiv.org/abs/1706.03762"], # Attention is all you need
        max_depth=2,
        top_k_citations=2
    )
    
    print("\nCrawl Result:", result)
    
    print("\nChecking SQLite database...")
    with get_db() as conn:
        rows = conn.execute("SELECT unique_id, parent_id, depth, data FROM research_papers").fetchall()
        print(f"Total papers in DB: {len(rows)}")
        for row in rows:
            print(f"- ID: {row['unique_id']} | Depth: {row['depth']} | Parent: {row['parent_id']}")
            
if __name__ == "__main__":
    asyncio.run(main())

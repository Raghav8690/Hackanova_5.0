from __future__ import annotations
import logging
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any
from bs4 import BeautifulSoup
from app.config import ARXIV_API_URL, PUBMED_ESEARCH_URL, PUBMED_EFETCH_URL, SEMANTIC_SCHOLAR_API_URL, SEMANTIC_SCHOLAR_API_KEY, GOOGLE_SCHOLAR_URL, TOP_K_PAPERS, MAX_RESULTS_PER_QUERY
from app.utils import fetch_json, fetch_xml, create_session, get_rate_limiter
from app.agents.query_agent import QueryAnalysis

logger = logging.getLogger("PaperDiscoveryAgent")
CURRENT_YEAR = datetime.now().year

@dataclass
class Paper:
    title: str = ""
    authors: list[str] = field(default_factory=list)
    year: str = ""
    abstract: str = ""
    citations: int = 0
    url: str = ""
    source: str = ""
    _relevance_score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("_relevance_score", None)
        return d

def _keyword_overlap(text: str, keywords: list[str]) -> float:
    if not keywords: return 0.0
    text_lower = text.lower()
    hits = sum(1 for kw in keywords if kw.lower() in text_lower)
    return hits / len(keywords)

def _recency_score(year_str: str) -> float:
    try: year = int(year_str)
    except (ValueError, TypeError): return 0.0
    return max(0.0, 1.0 - max(0, CURRENT_YEAR - year) / 20.0)

def score_paper(paper: Paper, keywords: list[str]) -> float:
    return 0.45 * _keyword_overlap(f"{paper.title} {paper.abstract}", keywords) + 0.30 * _recency_score(paper.year) + 0.25 * (min(paper.citations / 500.0, 1.0) if paper.citations else 0.0)

def _rank_and_select(papers: list[Paper], keywords: list[str], top_k: int = TOP_K_PAPERS) -> list[Paper]:
    for p in papers: p._relevance_score = score_paper(p, keywords)
    papers.sort(key=lambda p: p._relevance_score, reverse=True)
    return papers[:top_k]

def _fetch_arxiv(queries: list[str], max_results: int = MAX_RESULTS_PER_QUERY) -> list[Paper]:
    papers: list[Paper] = []
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    for query in queries[:4]:
        xml_text = fetch_xml(ARXIV_API_URL, params={"search_query": f"all:{query}", "start": 0, "max_results": max_results, "sortBy": "relevance", "sortOrder": "descending"}, source="arxiv")
        if not xml_text: continue
        try: root = ET.fromstring(xml_text)
        except ET.ParseError: continue
        for entry in root.findall("atom:entry", ns):
            title = (entry.findtext("atom:title", "", ns) or "").strip().replace("\n", " ")
            abstract = (entry.findtext("atom:summary", "", ns) or "").strip().replace("\n", " ")
            year = (entry.findtext("atom:published", "", ns) or "")[:4]
            authors = [a.text.strip() for a in entry.findall("atom:author/atom:name", ns) if a.text]
            url = next((link.get("href", "") for link in entry.findall("atom:link", ns) if link.get("type") == "text/html"), entry.findtext("atom:id", "", ns) or "")
            papers.append(Paper(title=title, authors=authors, year=year, abstract=abstract, url=url, source="arXiv"))
    return list({p.title.lower(): p for p in papers}.values())

def _fetch_pubmed(queries: list[str], max_results: int = MAX_RESULTS_PER_QUERY) -> list[Paper]:
    all_ids: list[str] = []
    for query in queries[:4]:
        data = fetch_json(PUBMED_ESEARCH_URL, params={"db": "pubmed", "term": query, "retmax": max_results, "retmode": "json", "sort": "relevance"}, source="pubmed")
        if data and "esearchresult" in data: all_ids.extend(data["esearchresult"].get("idlist", []))
    all_ids = list(dict.fromkeys(all_ids))
    if not all_ids: return []
    papers: list[Paper] = []
    for i in range(0, len(all_ids), 50):
        xml_text = fetch_xml(PUBMED_EFETCH_URL, params={"db": "pubmed", "id": ",".join(all_ids[i:i+50]), "retmode": "xml"}, source="pubmed")
        if not xml_text: continue
        try: root = ET.fromstring(xml_text)
        except ET.ParseError: continue
        for article in root.findall(".//PubmedArticle"):
            med = article.find(".//MedlineCitation")
            if med is None: continue
            title_el = med.find(".//ArticleTitle")
            year_el = med.find(".//PubDate/Year")
            if year_el is None or not year_el.text:
                md = med.find(".//PubDate/MedlineDate")
                year = md.text[:4] if md is not None and md.text else ""
            else:
                year = year_el.text.strip()
            pmid = med.findtext("PMID", "")
            papers.append(Paper(title=(title_el.text or "").strip() if title_el is not None else "", authors=[f"{auth.findtext('ForeName', '')} {auth.findtext('LastName', '')}".strip() for auth in med.findall(".//Author") if auth.findtext('LastName', '')], year=year, abstract=(a.text or "").strip() if (a := med.find(".//AbstractText")) is not None else "", url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else "", source="PubMed"))
    return list({p.title.lower(): p for p in papers}.values())

def _fetch_semantic_scholar(queries: list[str], max_results: int = MAX_RESULTS_PER_QUERY) -> list[Paper]:
    papers: list[Paper] = []
    headers = {"x-api-key": SEMANTIC_SCHOLAR_API_KEY} if SEMANTIC_SCHOLAR_API_KEY else None
    for query in queries[:4]:
        data = fetch_json(f"{SEMANTIC_SCHOLAR_API_URL}/paper/search", params={"query": query, "limit": max_results, "fields": "title,authors,year,abstract,citationCount,url,externalIds"}, headers=headers, source="semantic_scholar")
        if not data or "data" not in data: continue
        for item in data["data"]:
            url = item.get("url", "") or (f"https://doi.org/{item.get('externalIds', {}).get('DOI')}" if item.get('externalIds', {}).get('DOI') else "")
            papers.append(Paper(title=item.get("title", ""), authors=[a.get("name", "") for a in item.get("authors", []) if a.get("name")], year=str(item.get("year", "")), abstract=item.get("abstract", "") or "", citations=item.get("citationCount", 0) or 0, url=url, source="Semantic Scholar"))
    return list({p.title.lower(): p for p in papers}.values())

def _fetch_google_scholar(queries: list[str], max_results: int = 10) -> list[Paper]:
    session = create_session()
    papers: list[Paper] = []
    for query in queries[:2]:
        get_rate_limiter("google_scholar").wait()
        try:
            resp = session.get(GOOGLE_SCHOLAR_URL, params={"q": query, "hl": "en", "num": max_results}, timeout=15)
            if resp.status_code == 429: continue
            resp.raise_for_status()
        except Exception: continue
        soup = BeautifulSoup(resp.text, "html.parser")
        for result in soup.select("div.gs_ri"):
            title_tag = result.select_one("h3.gs_rt a") or result.select_one("h3.gs_rt")
            info_text = (i.get_text(strip=True) if (i := result.select_one("div.gs_a")) else "")
            cite_tag = result.select_one("a:-soup-contains('Cited by')")
            if title_tag:
                papers.append(Paper(title=title_tag.get_text(strip=True), authors=[a.strip() for a in (info_text.split(" - ")[0] if " - " in info_text else "").split(",") if a.strip()], year=(m.group(0) if (m := re.search(r"\b(19|20)\d{2}\b", info_text)) else ""), abstract=(s.get_text(strip=True) if (s := result.select_one("div.gs_rs")) else ""), citations=(int(cm.group(1)) if cite_tag and (cm := re.search(r"Cited by (\d+)", cite_tag.get_text())) else 0), url=title_tag.get("href", "") if title_tag.name == "a" else "", source="Google Scholar"))
    return list({p.title.lower(): p for p in papers}.values())

class PaperDiscoveryAgent:
    _FETCHERS = [("arXiv", _fetch_arxiv), ("PubMed", _fetch_pubmed), ("Semantic Scholar", _fetch_semantic_scholar), ("Google Scholar", _fetch_google_scholar)]

    def discover(self, analysis: QueryAnalysis) -> dict[str, Any]:
        keywords = analysis.keywords or [analysis.original_query]
        results = {}
        for keyword in keywords:
            source_results = {}
            for source_name, fetcher in self._FETCHERS:
                papers = []
                try: papers = fetcher([keyword])
                except Exception: pass
                unique = list({p.url: p for p in papers if p.url}.values())
                top_papers = _rank_and_select(unique, analysis.keywords, top_k=TOP_K_PAPERS)
                source_results[source_name.lower().replace(" ", "_")] = [p.url for p in top_papers]
            results[keyword] = source_results
        return {"query": analysis.original_query, "keywords": results}
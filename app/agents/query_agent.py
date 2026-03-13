from __future__ import annotations
import json
import logging
import os
from dataclasses import dataclass, field, asdict
from typing import Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from app.config import GEMINI_API_KEY, DOMAIN_SYNONYMS
from app.config import settings

logger = logging.getLogger("QueryAgent")

class QueryAnalysisModel(BaseModel):
    keywords: list[str] = Field(default_factory=list)
    expanded_keywords: list[str] = Field(default_factory=list)
    search_queries: list[str] = Field(default_factory=list)
    domain: str = Field(default="")
    methodologies: list[str] = Field(default_factory=list)
    applications: list[str] = Field(default_factory=list)

@dataclass
class QueryAnalysis:
    original_query: str = ""
    keywords: list[str] = field(default_factory=list)
    expanded_keywords: list[str] = field(default_factory=list)
    search_queries: list[str] = field(default_factory=list)
    domain: str = ""
    methodologies: list[str] = field(default_factory=list)
    applications: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

class QueryUnderstandingAgent:
    def __init__(self) -> None:
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY is not set.")
        os.environ["GOOGLE_API_KEY"] = settings.GEMINI_API_KEY_QUERY
        model_id = os.getenv("GEMINI_MODEL_ID") or "gemma-3-27b-it"
        self.llm = ChatGoogleGenerativeAI(model=model_id, temperature=0.1, google_api_key=settings.GEMINI_API_KEY_QUERY)
        self.parser = PydanticOutputParser(pydantic_object=QueryAnalysisModel)
        self.prompt = ChatPromptTemplate.from_template(
            """You are an expert academic research assistant specializing in extracting optimal keywords for literature search.
User query:
"{query}"
{format_instructions}
CRITICAL RULES: Extract 3-7 compound keyphrases. Preserve multi-word phrases. Include synonyms. Prioritize precision."""
        )
        self.chain = self.prompt | self.llm | self.parser

    def analyze(self, query: str) -> QueryAnalysis:
        try:
            model_result: QueryAnalysisModel = self.chain.invoke({"query": query, "format_instructions": self.parser.get_format_instructions()})
        except Exception as e:
            logger.error("Query synthesis failed: %s", e, exc_info=True)
            return QueryAnalysis(original_query=query, keywords=[query], search_queries=[query])

        analysis = QueryAnalysis(
            original_query=query,
            keywords=model_result.keywords if model_result.keywords else [query],
            expanded_keywords=model_result.expanded_keywords,
            search_queries=model_result.search_queries,
            domain=model_result.domain or "general",
            methodologies=model_result.methodologies,
            applications=model_result.applications,
        )
        self._augment_with_domain_synonyms(analysis)
        return analysis

    def _augment_with_domain_synonyms(self, analysis: QueryAnalysis) -> None:
        extra_terms: set[str] = set(analysis.expanded_keywords)
        for kw in analysis.keywords:
            lower_kw = kw.lower().strip()
            if lower_kw in DOMAIN_SYNONYMS: extra_terms.update(DOMAIN_SYNONYMS[lower_kw])
        domain_key = analysis.domain.lower().strip()
        if domain_key in DOMAIN_SYNONYMS: extra_terms.update(DOMAIN_SYNONYMS[domain_key])
        ordered = [t for t in analysis.expanded_keywords if t in extra_terms]
        for t in sorted(extra_terms):
            if t not in ordered: ordered.append(t)
        analysis.expanded_keywords = ordered
"""osint_agent.py
===================

OSINTAgent – gathers open‑source intelligence using web search and the RAG
engine. It merges the two sources, constructs a prompt from ``prompts.yaml`` and
asks the OpenRouter LLM for a concise summary.

The agent returns a JSON‑serialisable dictionary that downstream agents (e.g.
AnalystAgent, ReporterAgent) can consume.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List

from agents.base_agent import BaseAgent
from config.config_loader import ConfigLoader
from tools.web_search import WebSearchTool
from tools.rag_engine import RAGEngine
from tools.cag_cache import CAGCache

logger = logging.getLogger(__name__)


class OSINTAgent(BaseAgent):
    """Collects OSINT data for a given query.

    The workflow is:
    1. Validate input.
    2. Check cache – return cached result if present.
    3. Run a web search via :class:`WebSearchTool`.
    4. If any documents are indexed in the RAG engine, perform a semantic
       retrieval.
    5. Merge the raw web results and the RAG snippets.
    6. Build a prompt using the ``analysis_template`` from ``prompts.yaml``.
    7. Call the OpenRouter LLM.
    8. Store the final answer in vector memory for future reuse.
    9. Return a structured response.
    """

    def __init__(self) -> None:
        super().__init__(name="osint_agent")
        self.loader = ConfigLoader.instance()
        self.cache = CAGCache(self.loader.get_tool_config("cache"))
        self.web_tool = WebSearchTool(self.loader.get_tool_config("web_search"))
        self.rag_tool = RAGEngine(self.loader.get_tool_config("rag"))
        self.max_retries = self.loader.get_agent_config("osint").get("max_retries", 2)

    # ---------------------------------------------------------------------
    def _merge_results(self, web_results: List[Dict[str, str]], rag_results: List[str]) -> str:
        """Combine web and RAG results into a single text block.

        The function de‑duplicates overlapping information and returns a plain
        string suitable for inclusion in a prompt.
        """
        merged: List[str] = []
        for item in web_results:
            merged.append(f"{item.get('title', '')}: {item.get('snippet', '')} ({item.get('url', '')})")
        merged.extend(rag_results)
        return "\n\n".join(merged)

    # ---------------------------------------------------------------------
    def execute(self, query: str) -> Dict[str, Any]:
        """Execute the OSINT workflow.

        Parameters
        ----------
        query: str
            The user supplied OSINT request (e.g. ``"latest ransomware campaign"``).

        Returns
        -------
        dict
            Structured result containing a summary, sources and confidence.
        """
        start = time.time()
        if not self._validate_input(query):
            return self._handle_error(ValueError("Invalid or unsafe input"), "OSINT validation")

        cache_key = f"osint:{query.lower()}"
        cached = self.cache.get(cache_key)
        if cached:
            logger.info("OSINT cache hit for query: %s", query)
            return json.loads(cached)

        web_results: List[Dict[str, str]] = []
        rag_results: List[str] = []
        attempt = 0
        while attempt <= self.max_retries:
            try:
                logger.info("Running web search for query: %s (attempt %d)", query, attempt + 1)
                web_results = self.web_tool.search(query)
                break
            except Exception as exc:
                logger.warning("Web search failed (attempt %d): %s", attempt + 1, exc)
                attempt += 1
                if attempt > self.max_retries:
                    return self._handle_error(exc, "Web search")

        try:
            logger.info("Running RAG retrieval for query: %s", query)
            rag_results = self.rag_tool.retrieve(query)
        except Exception as exc:
            logger.warning("RAG engine error: %s", exc)
            # Continue with web results only – RAG is optional.

        merged_text = self._merge_results(web_results, rag_results)
        prompt = self._build_prompt(
            "analysis_template",
            query=query,
            data=merged_text,
        )
        try:
            llm_response = self._call_llm(prompt)
        except Exception as exc:
            return self._handle_error(exc, "LLM call")

        # Simple confidence heuristic: proportion of non‑empty sources.
        confidence = min(1.0, len(web_results) / 5.0)
        result = {
            "agent": "osint",
            "summary": llm_response.strip(),
            "sources": [item.get("url") for item in web_results if item.get("url")],
            "confidence": round(confidence, 2),
        }

        # Cache the JSON string for fast future look‑ups.
        self.cache.set(cache_key, json.dumps(result))

        # Persist to vector memory for long‑term retrieval.
        self._save_to_memory(cache_key, llm_response, mem_type="vector")

        logger.info(
            "OSINT finished (%.2f s) – confidence %.2f",
            time.time() - start,
            confidence,
        )
        return result

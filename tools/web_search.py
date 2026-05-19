"""web_search.py
================

Web search tool for the CTI Multi-Agent System.

Supports multiple search providers:
* **DuckDuckGo** (default, no API key required)
* **Tavily** (requires ``TAVILY_API_KEY``)

Results are returned as a list of dicts with keys ``title``, ``snippet``
and ``url``.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class WebSearchTool:
    """Execute web searches and return structured results.

    Parameters
    ----------
    config : dict
        Tool configuration from ``settings.yaml``.  Recognised keys:

        * ``provider`` – ``"duckduckgo"`` (default) or ``"tavily"``.
        * ``max_results`` – number of results to return (default 5).
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        self.provider: str = self.config.get("provider", "duckduckgo").lower()
        self.fallback_provider: str = self.config.get("fallback_provider", "duckduckgo").lower()
        self.max_results: int = int(self.config.get("max_results", 5))
        self.provider_timeout_seconds: int = int(self.config.get("provider_timeout_seconds", 15))
        self.min_results_before_fallback: int = int(self.config.get("min_results_before_fallback", 1))
        self.last_meta: Dict[str, Any] = {}
        logger.info(
            "WebSearchTool initialised (provider=%s, fallback=%s, max_results=%d).",
            self.provider, self.fallback_provider, self.max_results
        )

    # =====================================================================
    # Public API
    # =====================================================================

    def search(self, query: str) -> List[Dict[str, str]]:
        """Search the web for *query* and return structured results.

        Returns
        -------
        list[dict]
            Each dict has keys ``title``, ``snippet``, ``url``.
        """
        if not query or not query.strip():
            logger.warning("WebSearchTool: empty query.")
            return []

        provider = self.provider
        results: List[Dict[str, str]] = []
        fallback_triggered = False

        if provider == "tavily":
            results = self._search_tavily(query)
            if len(results) < self.min_results_before_fallback and self.fallback_provider == "duckduckgo":
                fallback_triggered = True
                logger.info("Tavily returned %d results; falling back to DuckDuckGo.", len(results))
                ddg_results = self._search_duckduckgo(query)
                if ddg_results:
                    results = ddg_results
                    provider = "duckduckgo"
        else:
            results = self._search_duckduckgo(query)

        self.last_meta = {
            "search_provider_used": provider,
            "search_fallback_triggered": fallback_triggered,
            "search_results_count": len(results),
        }
        return results

    # =====================================================================
    # DuckDuckGo provider
    # =====================================================================

    def _search_duckduckgo(self, query: str) -> List[Dict[str, str]]:
        """Search using DuckDuckGo."""
        try:
            from ddgs import DDGS
        except ImportError:
            from duckduckgo_search import DDGS
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=self.max_results))
                return [
                    {
                        "title": r.get("title", ""),
                        "url": r.get("href", ""),
                        "snippet": r.get("body", ""),
                    }
                    for r in results
                ]
        except Exception as exc:
            logger.error("DuckDuckGo search error: %s", exc)
            return []

    # =====================================================================
    # Tavily provider
    # =====================================================================

    def _search_tavily(self, query: str) -> List[Dict[str, str]]:
        """Search using the Tavily API (requires TAVILY_API_KEY)."""
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            logger.warning("TAVILY_API_KEY not set; falling back to DuckDuckGo.")
            return self._search_duckduckgo(query)

        try:
            import requests
        except ImportError:
            logger.error("requests package not installed.")
            return []

        results: List[Dict[str, str]] = []
        try:
            resp = requests.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": api_key,
                    "query": query,
                    "search_depth": "advanced",
                    "max_results": self.max_results,
                    "include_answer": False,
                },
                timeout=self.provider_timeout_seconds,
            )
            resp.raise_for_status()
            data = resp.json()
            for item in data.get("results", []):
                results.append({
                    "title":   item.get("title", ""),
                    "snippet": item.get("content", ""),
                    "url":     item.get("url", ""),
                })
            logger.info("Tavily returned %d results for: %s", len(results), query)
        except Exception as exc:
            logger.exception("Tavily search failed: %s", exc)

        return results

    def __repr__(self) -> str:
        return f"WebSearchTool(provider={self.provider}, max_results={self.max_results})"

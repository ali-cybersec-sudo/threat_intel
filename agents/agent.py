"""
Base Agent – foundational class that every concrete agent inherits from.
It owns the shared tool‑providers so that sub‑agents can request exactly the
capabilities they need without pulling in unused dependencies.

Project layout (referenced by Structure.md):
    agents/
        agent.py          ← this file (BaseAgent)
        osint_agent.py
        analyst_agent.py
        reporter_agent.py
    tools/
        web_search.py
        rag_engine.py
        cag_cache.py
    memory/
        session_memory.py
        vector_memory.py
    config/
        settings.yaml
        prompts.yaml
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

# ---------------------------------------------------------------------------
# Import real tool classes; fall back to stubs if they are missing.
# ---------------------------------------------------------------------------
try:
    from tools.web_search import WebSearchTool as _WebSearchTool
except ImportError:
    class _WebSearchTool:
        def __init__(self, config: dict) -> None:
            raise NotImplementedError(
                "WebSearchTool not implemented yet – create tools/web_search.py"
            )
    # Re‑assign to keep the name stable for type hints
    _WebSearchTool = _WebSearchTool

try:
    from tools.rag_engine import RAGEngine as _RAGEngine
except ImportError:
    class _RAGEngine:
        def __init__(self, config: dict) -> None:
            raise NotImplementedError(
                "RAGEngine not implemented yet – create tools/rag_engine.py"
            )
    _RAGEngine = _RAGEngine

try:
    from tools.cag_cache import CAGCache as _CAGCache
except ImportError:
    class _CAGCache:
        def __init__(self, config: dict) -> None:
            raise NotImplementedError(
                "CAGCache not implemented yet – create tools/cag_cache.py"
            )
    _CAGCache = _CAGCache

try:
    from memory.session_memory import SessionMemory as _SessionMemory
except ImportError:
    class _SessionMemory:
        def __init__(self, config: dict) -> None:
            raise NotImplementedError(
                "SessionMemory not implemented yet – create memory/session_memory.py"
            )
    _SessionMemory = _SessionMemory

try:
    from memory.vector_memory import VectorMemory as _VectorMemory
except ImportError:
    class _VectorMemory:
        def __init__(self, config: dict) -> None:
            raise NotImplementedError(
                "VectorMemory not implemented yet – create memory/vector_memory.py"
            )
    _VectorMemory = _VectorMemory

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """
    Abstract base for all agents.

    Provides:
        * A uniform interface (``run``) that every concrete agent must implement.
        * Lazy‑initialised accessors for the shared tools (web_search, rag,
          cache, session memory, vector memory).  Sub‑classes obtain a tool by
          calling the corresponding ``provide_*`` method; the base guarantees
          that only one instance of each tool is created per agent process.
    """

    # ------------------------------------------------------------------
    # Constructor / lifecycle
    # ------------------------------------------------------------------
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Parameters
        ----------
        config:
            Dictionary with agent‑specific settings (e.g. model name, temperature,
            allowed tools).  If ``None`` an empty dict is used.
        """
        self._config = config or {}
        self._tools: Dict[str, Any] = {}
        logger.debug("BaseAgent created with config=%s", self._config)

    # ------------------------------------------------------------------
    # Abstract entry point
    # ------------------------------------------------------------------
    @abstractmethod
    def run(self, prompt: str, **kwargs) -> str:
        """Process a user request and return a textual answer."""
        ...

    # ------------------------------------------------------------------
    # Tool providers (lazy singletons)
    # ------------------------------------------------------------------
    def provide_web_search(self) -> _WebSearchTool:
        """Return a ready‑to‑use web‑search tool instance."""
        if "web_search" not in self._tools:
            self._tools["web_search"] = _WebSearchTool(
                self._config.get("web_search", {})
            )
        return self._tools["web_search"]

    def provide_rag_engine(self) -> _RAGEngine:
        """Return a RAG engine backed by the configured vector store."""
        if "rag_engine" not in self._tools:
            self._tools["rag_engine"] = _RAGEngine(
                self._config.get("rag", {})
            )
        return self._tools["rag_engine"]

    def provide_cache(self) -> _CAGCache:
        """Return a caching layer for previously generated answers."""
        if "cache" not in self._tools:
            self._tools["cache"] = _CAGCache(
                self._config.get("cache", {})
            )
        return self._tools["cache"]

    def provide_session_memory(self) -> _SessionMemory:
        """Return the short‑term conversation memory for this agent."""
        if "session_memory" not in self._tools:
            self._tools["session_memory"] = _SessionMemory(
                self._config.get("session", {})
            )
        return self._tools["session_memory"]

    def provide_vector_memory(self) -> _VectorMemory:
        """Return the long‑term vector‑store memory."""
        if "vector_memory" not in self._tools:
            self._tools["vector_memory"] = _VectorMemory(
                self._config.get("vector_memory", {})
            )
        return self._tools["vector_memory"]

    # ------------------------------------------------------------------
    # Helper utilities (may be overridden)
    # ------------------------------------------------------------------
    def log(self, msg: str, level: str = "info") -> None:
        """Centralised logging – respects the project's log format."""
        getattr(logger, level, logger.info)(msg)

    def _sanitize_input(self, text: str) -> str:
        """Strip potentially dangerous content before forwarding to tools."""
        # placeholder – will be replaced by the security module later
        return text.strip()

    def _sanitize_output(self, text: str) -> str:
        """Post‑process tool output to remove sensitive data."""
        # placeholder – will be replaced by the security module later
        return text
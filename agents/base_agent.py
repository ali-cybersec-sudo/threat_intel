"""BaseAgent - foundational class for all CTI agents.

This class provides:
* OpenRouter LLM integration (via ConfigLoader)
* Lazy-loaded tool helpers (web search, RAG, cache, memory)
* Prompt construction from ``config/prompts.yaml``
* Structured logging and error handling
* Helper methods for memory interaction and input validation
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from config.config_loader import ConfigLoader
from tools.llm_client import LLMClient

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """Abstract base class for all agents.

    Sub-classes must implement :meth:`execute` which receives a user query or
    structured data and returns a JSON-serialisable ``dict``.
    """

    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None) -> None:
        self.name = name
        self.cfg = config or {}
        self.loader = ConfigLoader.instance()
        self._llm_client = LLMClient(dict(self.loader.settings))
        self._last_llm_meta: Dict[str, Any] = {}
        self._last_memory_context: List[str] = []

    # ---------------------------------------------------------------------
    # LLM integration
    # ---------------------------------------------------------------------
    def _call_llm(
        self,
        prompt: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout: int | None = None,
    ) -> str:
        """Call the configured LLM and return the generated text.

        Parameters
        ----------
        prompt: str
            Full prompt sent to the model.
        temperature, max_tokens, timeout: optional overrides; fall back to config.
        """
        # Retrieve long-term facts
        vector_context = self._retrieve_from_memory(prompt, mem_type="vector", top_k=2)
        # Retrieve recent chat history
        session_context = self._retrieve_from_memory(prompt, mem_type="session", top_k=6)

        memory_context = vector_context + session_context
        self._last_memory_context = memory_context

        if memory_context:
            memory_block = "\n".join(f"- {item[:500]}" for item in memory_context if str(item).strip())
            prompt = (
                "Relevant memory and recent chat context:\n"
                f"{memory_block}\n\n"
                "Use this context to remember the conversation and prior facts. Do not invent missing facts.\n\n"
                f"{prompt}"
            )

        try:
            text, meta = self._llm_client.generate(
                prompt=prompt,
                provider=None,
                allow_fallback=True,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
            )
            self._last_llm_meta = dict(meta)
            return text
        except Exception as exc:
            logger.warning("LLM call failed; using fallback where possible: %s", exc)
            if memory_context:
                self._last_llm_meta = {
                    "provider": "memory_fallback",
                    "fallback_triggered": True,
                    "error": str(exc),
                }

                # Format snippets neatly (handling json session objects)
                formatted_snippets = []
                for item in memory_context:
                    if not item: continue
                    if isinstance(item, str) and item.startswith("{"):
                        try:
                            msg = __import__("json").loads(item)
                            formatted_snippets.append(f"{msg.get('role', 'System').title()}: {msg.get('content', '')[:300]}")
                        except:
                            formatted_snippets.append(f"- {item[:300]}")
                    else:
                        formatted_snippets.append(f"- {str(item)[:300]}")

                snippets_text = "\n".join(formatted_snippets)
                return (
                    "**Memory-enhanced response based on prior chat and CTI context:**\n\n"
                    f"{snippets_text}\n\n"
                    "> *Note: Live LLM generation was unavailable (API limit or error), so this answer is grounded in retrieved memory.*"
                )
            raise RuntimeError(f"LLM request failed: {exc}") from exc

    def get_last_llm_meta(self) -> Dict[str, Any]:
        """Return metadata for the most recent LLM call."""
        return dict(self._last_llm_meta)

    def get_last_memory_context(self) -> List[str]:
        """Return memory snippets injected into the most recent LLM call."""
        return list(self._last_memory_context)

    # ---------------------------------------------------------------------
    # Prompt handling
    # ---------------------------------------------------------------------
    def _build_prompt(self, template_key: str, **kwargs) -> str:
        """Render a prompt template from ``prompts.yaml``.

        ``template_key`` corresponds to a key defined under the agent's entry in
        the prompts file (e.g. ``"system_prompt"`` or ``"analysis_template"``).
        """
        template = self.loader.get_prompt_template(self.name, template_key)
        return template.format(**kwargs)

    # ---------------------------------------------------------------------
    # Memory helpers
    # ---------------------------------------------------------------------
    def _save_to_memory(self, key: str, value: str, mem_type: str = "vector") -> None:
        if mem_type == "vector":
            vector_mem = __import__("memory.vector_memory", fromlist=["VectorMemory"]).VectorMemory(self.loader.get_memory_config("vector_memory"))
            vector_mem.store(key, value)
        elif mem_type == "session":
            session_mem = __import__("memory.session_memory", fromlist=["SessionMemory"]).SessionMemory(self.loader.get_memory_config("session_memory"))
            session_mem.add(value)

    def _retrieve_from_memory(self, query: str, mem_type: str = "vector", top_k: int = 5) -> List[str]:
        if mem_type == "vector":
            vector_mem = __import__("memory.vector_memory", fromlist=["VectorMemory"]).VectorMemory(self.loader.get_memory_config("vector_memory"))
            return vector_mem.search(query, top_k=top_k)
        elif mem_type == "session":
            session_mem = __import__("memory.session_memory", fromlist=["SessionMemory"]).SessionMemory(self.loader.get_memory_config("session_memory"))
            return session_mem.get_recent()
        return []

    # ---------------------------------------------------------------------
    # Validation and error handling helpers
    # ---------------------------------------------------------------------
    def _validate_input(self, text: str) -> bool:
        guard_cfg = self.loader.get_tool_config("input_guard")
        if guard_cfg:
            from security.input_guard import InputGuard
            guard = InputGuard(guard_cfg)
            return guard.validate_input(text)
        return True

    def _handle_error(self, err: Exception, context: str) -> Dict[str, Any]:
        logger.error("%s: %s", context, err)
        return {"error": str(err), "context": context}

    # ---------------------------------------------------------------------
    # Abstract execution entry point
    # ---------------------------------------------------------------------
    @abstractmethod
    def execute(self, payload: Any) -> Dict[str, Any]:
        """Execute the agent's core logic.

        Returns a JSON-serialisable dictionary that downstream agents can consume.
        """
        raise NotImplementedError

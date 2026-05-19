"""BaseAgent – foundational class for all CTI agents.

This class provides:
* OpenRouter LLM integration (via ConfigLoader)
* Lazy‑loaded tool helpers (web search, RAG, cache, memory)
* Prompt construction from ``config/prompts.yaml``
* Structured logging and error handling
* Helper methods for memory interaction and input validation
"""

from __future__ import annotations

import json
import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import requests

from config.config_loader import ConfigLoader

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """Abstract base class for all agents.

    Sub‑classes must implement :meth:`execute` which receives a user query or
    structured data and returns a JSON‑serialisable ``dict``.
    """

    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None) -> None:
        self.name = name
        self.cfg = config or {}
        self.loader = ConfigLoader.instance()
        self._api_key = self.loader.get_api_key("openrouter")
        self._llm_cfg = self.loader.get_llm_config("openrouter")

    # ---------------------------------------------------------------------
    # LLM integration – OpenRouter
    # ---------------------------------------------------------------------
    def _call_llm(
        self,
        prompt: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout: int | None = None,
    ) -> str:
        """Call the OpenRouter LLM and return the generated text.

        Parameters
        ----------
        prompt: str
            Full prompt sent to the model.
        temperature, max_tokens, timeout: optional overrides; fall back to config.
        """
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        endpoint = self._llm_cfg.get("endpoint")
        model = self._llm_cfg.get("model")
        payload = {
            "model": model,
            "temperature": temperature if temperature is not None else self._llm_cfg.get("temperature", 0.2),
            "max_tokens": max_tokens if max_tokens is not None else self._llm_cfg.get("max_tokens", 1024),
        }

        # Detect chat-style endpoints and send messages accordingly
        if endpoint and "chat" in endpoint.lower():
            payload["messages"] = [{"role": "user", "content": prompt}]
        else:
            payload["prompt"] = prompt

        try:
            response = requests.post(
                endpoint,
                headers=headers,
                json=payload,
                timeout=timeout or self._llm_cfg.get("timeout", 30),
            )
            # If the provider returns a non-2xx status, include the body in our log/error
            try:
                response.raise_for_status()
            except requests.exceptions.HTTPError as http_err:
                logger.error(
                    "LLM HTTP error: status=%s response=%s",
                    getattr(response, "status_code", None),
                    (response.text[:2000] if response is not None else "<no response>"),
                )
                raise RuntimeError(f"LLM request failed: HTTP {response.status_code} - {response.text}") from http_err

            data = response.json()

            # OpenRouter / Chat APIs usually return a `choices` list. Support both
            # chat-style ('message' -> 'content') and legacy 'text' fields.
            if isinstance(data, dict):
                choices = data.get("choices") or []
                if choices:
                    first = choices[0]
                    if isinstance(first, dict):
                        # Chat-style response
                        msg = first.get("message")
                        if isinstance(msg, dict):
                            return msg.get("content", "")
                        # Legacy completion-style
                        txt = first.get("text")
                        if txt:
                            return txt
                # Fallbacks for other provider shapes
                out = data.get("output") or data.get("result")
                if isinstance(out, str):
                    return out
                if isinstance(out, list):
                    return "\n".join([o.get("content", str(o)) if isinstance(o, dict) else str(o) for o in out])

            return json.dumps(data)
        except Exception as exc:
            logger.exception("OpenRouter LLM call failed")
            raise RuntimeError(f"LLM request failed: {exc}") from exc

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

    def _retrieve_from_memory(self, query: str, mem_type: str = "vector") -> List[str]:
        if mem_type == "vector":
            vector_mem = __import__("memory.vector_memory", fromlist=["VectorMemory"]).VectorMemory(self.loader.get_memory_config("vector_memory"))
            return vector_mem.search(query)
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

        Returns a JSON‑serialisable dictionary that downstream agents can consume.
        """
        raise NotImplementedError

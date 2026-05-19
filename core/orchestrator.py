"""orchestrator.py
===================

Central orchestrator for the CTI Multi-Agent System.

Receives a raw user query, validates it through the security layer,
obtains a routing plan from the Router, executes agents in the
prescribed order while threading outputs between them, applies
output-guard sanitisation and persists the conversation in session
memory.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional

from config.config_loader import ConfigLoader
from core.router import Router
from agents.osint_agent import OSINTAgent
from agents.analyst_agent import AnalystAgent
from agents.reporter_agent import ReporterAgent
from memory.session_memory import SessionMemory
from security.input_guard import InputGuard
from security.output_guard import OutputGuard
from core.intent_classifier import IntentClassifier

logger = logging.getLogger(__name__)


class Orchestrator:
    """Coordinate agents, memory and security for end-to-end query processing.

    The orchestrator is the single entry-point that the UI layer calls.
    It lazily instantiates agents only when the routing plan requires them.

    Parameters
    ----------
    config : dict | None
        Full application settings.  Falls back to ``ConfigLoader.settings``.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.loader = ConfigLoader.instance()
        self.cfg = config or dict(self.loader.settings)

        # Security gates
        self.input_guard  = InputGuard(self.cfg.get("input_guard", {}))
        self.output_guard = OutputGuard(self.cfg.get("output_guard", {}))

        # Router
        self.router = Router(self.cfg)

        # Session memory (per-session conversation history)
        session_cfg = self.cfg.get("session", {})
        self.session = SessionMemory(session_cfg)

        # Agent registry (lazy - instantiated on first use)
        self._agents: Dict[str, Any] = {}

        # Execution history for the current session
        self._history: List[Dict[str, Any]] = []

        self.classifier = IntentClassifier()

        logger.info("Orchestrator initialised.")

    # =====================================================================
    # Agent factory
    # =====================================================================

    def _get_agent(self, name: str) -> Any:
        """Return a cached agent instance, creating it on first request."""
        if name not in self._agents:
            if name == "osint":
                self._agents[name] = OSINTAgent()
            elif name == "analyst":
                self._agents[name] = AnalystAgent()
            elif name == "reporter":
                self._agents[name] = ReporterAgent()
            else:
                raise ValueError(f"Unknown agent: {name}")
            logger.info("Instantiated agent: %s", name)
        return self._agents[name]

    # =====================================================================
    # Public entry point
    # =====================================================================

    def handle_query(self, query: str) -> Dict[str, Any]:
        """Process a user query end-to-end.

        Steps
        -----
        1. Sanitise and validate via InputGuard.
        2. Route via Router.
        3. Execute each agent in order, threading outputs.
        4. Sanitise the final result via OutputGuard.
        5. Persist to session memory.

        Returns
        -------
        dict
            Final result dictionary containing the report and metadata.
        """
        start_time = time.time()

        # ── 1. Input validation ──────────────────────────────────────────
        clean_query = self.input_guard.sanitise(query)
        if not self.input_guard.validate(clean_query):
            return self._error_response(
                "Your query was rejected by the security filter. "
                "Please rephrase and try again.",
                query=query,
            )

        # ── 2. Routing ───────────────────────────────────────────────────
        # Classify intent and short-circuit conversational queries before
        # invoking the Router and agents.
        classification = self.classifier.classify(clean_query)

        if classification.get("path") == "conversational":
            return self._handle_conversational(clean_query, classification.get("intent"))

        # Continue with normal routing for investigation intents
        plan = self.router.route(clean_query)

        if plan.get("error") or not plan.get("agents"):
            return self._error_response(
                plan.get("error", "Router could not determine an execution plan."),
                query=clean_query,
                plan=plan,
            )

        logger.info(
            "Executing plan: intent=%s  agents=%s",
            plan["intent"], plan["agents"],
        )

        # ── 3. Sequential agent execution ────────────────────────────────
        agent_results: Dict[str, Any] = {}
        current_error: Optional[str] = None

        for agent_name in plan["agents"]:
            step_start = time.time()
            try:
                agent = self._get_agent(agent_name)
                payload = self._prepare_payload(agent_name, plan, agent_results, clean_query)
                result = agent.execute(payload)

                # Check for agent-level errors
                if isinstance(result, dict) and "error" in result:
                    logger.warning("Agent '%s' returned error: %s", agent_name, result["error"])
                    agent_results[agent_name] = result
                    # Non-fatal for osint/analyst; fatal only if reporter fails
                    if agent_name == "reporter":
                        current_error = f"Reporter agent failed: {result['error']}"
                else:
                    agent_results[agent_name] = result

                logger.info(
                    "Agent '%s' completed in %.2f s",
                    agent_name, time.time() - step_start,
                )

            except Exception as exc:
                logger.exception("Agent '%s' raised an exception.", agent_name)
                agent_results[agent_name] = {"error": str(exc), "agent": agent_name}
                if agent_name == "reporter":
                    current_error = f"Reporter agent crashed: {exc}"

        # ── 4. Assemble final response ───────────────────────────────────
        elapsed = round(time.time() - start_time, 2)

        if current_error:
            response = self._error_response(current_error, query=clean_query, plan=plan)
        else:
            response = self._build_response(plan, agent_results, clean_query, elapsed)

        # ── 5. Output sanitisation ───────────────────────────────────────
        response = self.output_guard.sanitise_dict(response)

        # ── 6. Session memory ────────────────────────────────────────────
        self.session.add(json.dumps({"role": "user", "content": clean_query}))
        self.session.add(json.dumps({"role": "assistant", "content": response.get("summary", "")}))
        self._history.append({"query": clean_query, "response": response, "elapsed": elapsed})

        logger.info("Query processed in %.2f s.", elapsed)
        return response

    # =====================================================================
    # Payload preparation
    # =====================================================================

    def _prepare_payload(
        self,
        agent_name: str,
        plan: Dict[str, Any],
        prior_results: Dict[str, Any],
        query: str,
    ) -> Any:
        """Build the input payload for *agent_name*.

        * **osint** receives the raw query string.
        * **analyst** receives the primary indicator string extracted by
          the Router.
        * **reporter** receives a dict ``{"osint": ..., "analyst": ...}``
          assembled from prior agent outputs.
        """
        params = plan.get("params", {}).get(agent_name, {})

        if agent_name == "osint":
            return params.get("query", query)

        if agent_name == "analyst":
            return params.get("indicator", query)

        if agent_name == "reporter":
            osint_data = prior_results.get("osint", {
                "agent": "osint",
                "summary": "No OSINT data collected.",
                "sources": [],
                "confidence": 0.0,
            })
            analyst_data = prior_results.get("analyst", {
                "agent": "analyst",
                "indicator": params.get("query", query),
                "type": "unknown",
                "details": {},
                "severity": "Info",
                "score": 0.0,
                "mitre": {},
                "confidence": 0.0,
            })
            return {"osint": osint_data, "analyst": analyst_data}

        # Fallback for any future agent types
        return query

    # =====================================================================
    # Conversational handling
    # =====================================================================

    def _handle_conversational(self, query: str, intent: str) -> Dict[str, Any]:
        """Return a lightweight conversational response without running agents."""
        if intent == "greeting":
            answer = "Hello! I'm a CTI analysis system—ask me to investigate an IP, domain, hash, CVE, or request research."
        elif intent == "capabilities":
            answer = (
                "I can: \n"
                "- Look up IP addresses, domains and file hashes for indicators of compromise.\n"
                "- Research CVEs and MITRE techniques.\n"
                "- Run an OSINT -> Analyst -> Reporter pipeline to produce full CTI reports.\n"
                "- Provide concise guidance and context for general cybersecurity questions."
            )
        else:
            # Generic short answer for questions and other conversational intents
            answer = "I can help with that — ask me to investigate an indicator or request a threat report."

        return {
            "status": "success",
            "query": query,
            "intent": intent or "conversational",
            "confidence": None,
            "summary": answer,
            "answer": answer,
            "markdown": "",
            "report": None,
            "severity": None,
            "indicators": {},
            "agents_used": [],
            "agent_results": {},
            "elapsed_seconds": 0.0,
        }

    # =====================================================================
    # Response builders
    # =====================================================================

    def _build_response(
        self,
        plan: Dict[str, Any],
        agent_results: Dict[str, Any],
        query: str,
        elapsed: float,
    ) -> Dict[str, Any]:
        """Assemble the final response from all agent outputs."""
        reporter_result = agent_results.get("reporter", {})
        analyst_result  = agent_results.get("analyst", {})
        osint_result    = agent_results.get("osint", {})

        markdown = reporter_result.get("markdown", "")
        summary  = ""
        if markdown:
            # Extract first non‑title paragraph as a summary, skipping generic placeholders
            lines = [l.strip() for l in markdown.split("\n") if l.strip() and not l.startswith("#")]
            # Remove common placeholder lines
            placeholder_patterns = [
                "Here is the summary:",
                "Here is a sample executive summary:",
                "Here is a sample report:",
            ]
            meaningful = [ln for ln in lines if ln not in placeholder_patterns]
            summary = meaningful[0] if meaningful else (lines[0] if lines else "Report generated successfully.")
        else:
            summary = "Analysis completed but no report was generated."

        return {
            "status":     "success",
            "query":      query,
            "intent":     plan.get("intent", "unknown"),
            "confidence": plan.get("confidence", 0.0),
            "summary":    summary,
            "markdown":   markdown,
            "severity":   analyst_result.get("severity", "Info"),
            "indicators": plan.get("indicators", {}),
            "agents_used": plan.get("agents", []),
            "agent_results": {
                "osint":    osint_result,
                "analyst":  analyst_result,
                "reporter": reporter_result,
            },
            "elapsed_seconds": elapsed,
        }

    @staticmethod
    def _error_response(
        message: str,
        query: str = "",
        plan: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Return a standardised error response."""
        logger.warning("Error response: %s", message)
        return {
            "status":     "error",
            "query":      query,
            "intent":     plan.get("intent", "none") if plan else "none",
            "confidence": 0.0,
            "summary":    message,
            "markdown":   "",
            "severity":   "Info",
            "indicators": plan.get("indicators", {}) if plan else {},
            "agents_used": [],
            "agent_results": {},
            "elapsed_seconds": 0.0,
        }

    # =====================================================================
    # Session helpers
    # =====================================================================

    def get_history(self) -> List[Dict[str, Any]]:
        """Return the execution history for the current session."""
        return list(self._history)

    def clear_session(self) -> None:
        """Reset session memory and execution history."""
        self.session = SessionMemory(self.cfg.get("session", {}))
        self._history.clear()
        self._agents.clear()
        logger.info("Session cleared.")

    def __repr__(self) -> str:
        return f"Orchestrator(agents_loaded={list(self._agents.keys())})"

    def route_query(self, query: str) -> dict:
        """Route a user query and return a unified result dict.

        This method classifies the query, directs conversational intents to the
        lightweight handler, and for investigative intents delegates to the
        full ``handle_query`` pipeline to produce real results.
        """
        classification = self.classifier.classify(query)
        path = classification.get("path")
        intent = classification.get("intent")
        # Base result skeleton
        result: Dict[str, Any] = {
            "path": path,
            "intent": intent,
            "answer": None,
            "report": None,
            "severity": None,
            "confidence": classification.get("confidence"),
            "agents_used": [],
        }

        if path == "conversational":
            # Use the lightweight conversational handler
            conv = self._handle_conversational(query, intent)
            result["answer"] = conv.get("answer")
            result["report"] = conv.get("report")
            result["severity"] = conv.get("severity")
            result["confidence"] = conv.get("confidence")
            result["agents_used"] = conv.get("agents_used", [])
        elif path == "investigation":
            # Run the full pipeline via handle_query to get real agent output
            full = self.handle_query(query)
            # ``handle_query`` returns a dict with keys matching UI expectations
            result["answer"] = full.get("summary") or full.get("answer")
            result["report"] = full.get("markdown")
            result["severity"] = full.get("severity")
            result["confidence"] = full.get("confidence")
            result["agents_used"] = full.get("agents_used", [])
        else:
            result["answer"] = "Unsupported query type."

        return result

    def _run_pipeline(self, query):
        # Placeholder for the existing osint→analyst→reporter pipeline logic
        return {
            "answer": "Pipeline executed successfully.",
            "report": "Generated report content.",
            "severity": "high",
            "confidence": 0.95,
            "agents_used": ["osint_agent", "analyst_agent", "reporter_agent"]
        }

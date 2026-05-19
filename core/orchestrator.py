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
import re

from config.config_loader import ConfigLoader
from core.router import Router
from agents.osint_agent import OSINTAgent
from agents.analyst_agent import AnalystAgent
from agents.reporter_agent import ReporterAgent
from memory.session_memory import SessionMemory
from memory.cti_memory import CTIMemory
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
        self.cti_memory = CTIMemory(self.cfg.get("memory", {}).get("cti_memory", {}))

        # Agent registry (lazy - instantiated on first use)
        self._agents: Dict[str, Any] = {}

        # Execution history for the current session
        self._history: List[Dict[str, Any]] = []

        # Track whether a report exists in session (for context-aware intent)
        self._last_active_report: Optional[Dict[str, Any]] = None

        # Counter for unmatched intents (visible in session stats)
        self._unmatched_intent_count: int = 0

        self.classifier = IntentClassifier()

        # Validate LLM connection at startup
        self._llm_available = False
        try:
            from tools.llm_client import LLMClient, LLMConfigurationError
            client = LLMClient(dict(self.loader.settings))
            client.validate_connection()
            self._llm_available = True
            logger.info("LLM connection validated successfully.")
        except Exception as exc:
            logger.warning(
                "LLM validation failed at startup: %s. "
                "The system will run with memory-only fallback until an API key is configured.",
                exc,
            )

        logger.info("Orchestrator initialised (llm_available=%s).", self._llm_available)

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
        memory_response = self._try_memory_followup(clean_query, start_time)
        if memory_response:
            memory_response = self.output_guard.sanitise_dict(memory_response)
            self._record_turn(clean_query, memory_response, round(time.time() - start_time, 2))
            return memory_response

        classification = self.classifier.classify(clean_query)
        if classification.get("path") == "conversational" and self.classifier.has_cyber_context(clean_query):
            classification = {"path": "investigation", "intent": "research", "ioc_type": None, "ioc_value": None}

        if classification.get("path") == "conversational":
            response = self._handle_conversational(clean_query, classification.get("intent"))
            response = self.output_guard.sanitise_dict(response)
            self._record_turn(clean_query, response, round(time.time() - start_time, 2))
            return response

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
        self._remember_investigation(response)

        # ── 6. Session memory ────────────────────────────────────────────
        self._record_turn(clean_query, response, elapsed)

        logger.info("Query processed in %.2f s.", elapsed)
        return response

    def _record_turn(self, query: str, response: Dict[str, Any], elapsed: float) -> None:
        """Persist one user/assistant turn in short-term session history."""
        assistant_text = response.get("summary") or response.get("answer") or ""
        self.session.add(json.dumps({"role": "user", "content": query}, ensure_ascii=False))
        self.session.add(json.dumps({"role": "assistant", "content": assistant_text}, ensure_ascii=False))
        self._history.append({"query": query, "response": response, "elapsed": elapsed})

    def _try_memory_followup(self, query: str, start_time: float) -> Optional[Dict[str, Any]]:
        """Answer follow-up questions from persistent CTI memory before routing."""
        record = self.cti_memory.resolve_followup(query)
        if not record:
            if self.cti_memory.is_followup(query):
                answer = (
                    "I do not have a previous IOC investigation in memory for that follow-up. "
                    "Run an investigation first, for example: `search ip 185.164.81.156`."
                )
                return {
                    "status": "success",
                    "query": query,
                    "intent": "memory_followup_missing",
                    "confidence": None,
                    "summary": answer,
                    "answer": answer,
                    "markdown": "",
                    "report": None,
                    "severity": "Info",
                    "indicators": {},
                    "agents_used": ["memory"],
                    "agent_results": {},
                    "memory_used": False,
                    "elapsed_seconds": round(time.time() - start_time, 2),
                }
            return None
        answer = self.cti_memory.answer_followup(query, record)
        elapsed = round(time.time() - start_time, 2)
        return {
            "status": "success",
            "query": query,
            "intent": "memory_followup",
            "confidence": record.get("confidence"),
            "summary": answer,
            "answer": answer,
            "markdown": "",
            "report": None,
            "severity": record.get("severity", "Info"),
            "indicators": {
                f"{record.get('indicator_type', 'indicator')}s": [record.get("indicator")]
            },
            "agents_used": ["memory"],
            "agent_results": {},
            "memory_used": True,
            "memory_record": {
                "indicator": record.get("indicator"),
                "indicator_type": record.get("indicator_type"),
                "timestamp": record.get("timestamp"),
            },
            "elapsed_seconds": elapsed,
        }

    def _remember_investigation(self, response: Dict[str, Any]) -> None:
        """Store investigation responses in durable CTI memory."""
        record = self.cti_memory.store_response(response)
        if record:
            response["memory_saved"] = True
            response["memory_record"] = {
                "indicator": record.get("indicator"),
                "indicator_type": record.get("indicator_type"),
                "timestamp": record.get("timestamp"),
            }
            if response.get("markdown") or response.get("report"):
                self._last_active_report = response

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
        if self.classifier.has_cyber_context(query):
            return self.handle_query(query)

        recent = self.session.get_recent(3)
        recent_text = " ".join(recent).lower() if recent else ""
        in_security_context = any(k in recent_text for k in ["ip", "hash", "cve", "mitre", "threat", "domain"])

        if intent == "greeting":
            if "cti analysis system" in recent_text or "security analysis assistant" in recent_text:
                answer = "Still here. Send me an indicator or a research topic and I will investigate it from the available sources."
            elif in_security_context:
                answer = "Hi. I am still tracking this security analysis context, so you can continue with the next indicator or report request."
            else:
                answer = "Hi. I can help investigate security indicators, research current vulnerabilities, or build a CTI report."
            answer = "Hello! I'm a CTI analysis system—ask me to investigate an IP, domain, hash, CVE, or request research."
            answer = (
                "Still here. Send me an indicator or a research topic and I will investigate it from the available sources."
                if "cti analysis system" in recent_text or "security analysis assistant" in recent_text else
                "Hi. I am still tracking this security analysis context, so you can continue with the next indicator or report request."
                if in_security_context else
                "Hi. I can help investigate security indicators, research current vulnerabilities, or build a CTI report."
            )
        elif intent == "capabilities":
            if re.search(r"\bwho are you\b", query, re.IGNORECASE):
                answer = (
                    "I'm your security analysis assistant. I keep track of this investigation context, "
                    "run external intelligence checks when needed, and turn findings into evidence-based reports."
                    if in_security_context else
                    "I'm an analysis assistant that helps investigate security indicators and produce clear reports."
                )
            else:
                answer = (
                    "I can: \n"
                    "- Investigate IPs, domains, hashes, CVEs, and MITRE techniques.\n"
                    "- Pull external intelligence and separate raw findings from interpretation.\n"
                    "- Build structured security reports with risk assessment and evidence notes."
                )
        else:
            # Unmatched intent — log it so silent misrouting is visible
            self._unmatched_intent_count += 1
            logger.warning(
                "[UNMATCHED INTENT] input='%s' classified_as='%s' "
                "unmatched_count=%d",
                query[:120], intent, self._unmatched_intent_count,
            )
            answer = (
                "I didn't understand that command. Try:\n"
                "- Investigate an IP, domain, hash, CVE, or MITRE technique\n"
                "- Research a threat topic (e.g., 'research APT29')\n"
                "- Summarize or export the report\n"
                "- Ask a security question"
            )

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
                "EXECUTIVE SUMMARY:",
                "Executive Summary",
                "Summary of Object",
                "Findings from Analysis",
            ]
            meaningful = [ln for ln in lines if ln not in placeholder_patterns]
            summary = meaningful[0] if meaningful else (lines[0] if lines else "Report generated successfully.")
        else:
            summary = "Analysis completed but no report was generated."

        confidence = analyst_result.get("confidence")
        if confidence is None:
            confidence = osint_result.get("confidence", plan.get("confidence", 0.0))
        if osint_result.get("memory_context_used"):
            confidence = max(confidence, osint_result.get("confidence", 0.0), 0.86)
        severity = self._normalise_severity(analyst_result.get("severity", "Info"))

        return {
            "status":     "success",
            "query":      query,
            "intent":     plan.get("intent", "unknown"),
            "confidence": round(confidence, 2),
            "summary":    summary,
            "markdown":   markdown,
            "severity":   severity,
            "indicators": plan.get("indicators", {}),
            "agents_used": plan.get("agents", []),
            "agent_results": {
                "osint":    osint_result,
                "analyst":  analyst_result,
                "reporter": reporter_result,
            },
            "llm_provider_used": (
                reporter_result.get("llm_provider_used")
                or analyst_result.get("llm_provider_used")
                or osint_result.get("llm_provider_used")
                or self.router.last_llm_meta.get("provider")
            ),
            "llm_fallback_triggered": any([
                bool(osint_result.get("llm_fallback_triggered", False)),
                bool(analyst_result.get("llm_fallback_triggered", False)),
                bool(reporter_result.get("llm_fallback_triggered", False)),
                bool(self.router.last_llm_meta.get("fallback_triggered", False)),
            ]),
            "search_provider_used": osint_result.get("search_provider_used"),
            "search_fallback_triggered": bool(osint_result.get("search_fallback_triggered", False)),
            "vt_used": bool(analyst_result.get("vt_used", False)),
            "memory_used": False,
            "memory_saved": False,
            "elapsed_seconds": elapsed,
        }

    @staticmethod
    def _normalise_severity(severity: Any) -> str:
        allowed = {"Info", "Unknown", "Low", "Medium", "High", "Critical"}
        text = str(severity or "Info").strip().title()
        return text if text in allowed else "Info"

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
        self._last_active_report = None
        logger.info("Session cleared.")

    def __repr__(self) -> str:
        return f"Orchestrator(agents_loaded={list(self._agents.keys())})"

    def route_query(self, query: str) -> dict:
        """Route a user query and return a unified result dict.

        This method classifies the query, directs conversational intents to the
        lightweight handler, and for investigative intents delegates to the
        full ``handle_query`` pipeline to produce real results.
        """
        clean_query = self.input_guard.sanitise(query)
        memory_response = self._try_memory_followup(clean_query, time.time())
        if memory_response:
            memory_response = self.output_guard.sanitise_dict(memory_response)
            self._record_turn(clean_query, memory_response, memory_response.get("elapsed_seconds", 0.0))
            return {
                "status": "success",
                "path": "memory",
                "intent": "memory_followup",
                "answer": memory_response.get("answer"),
                "report": None,
                "severity": memory_response.get("severity"),
                "confidence": memory_response.get("confidence"),
                "agents_used": ["memory"],
                "memory_used": bool(memory_response.get("memory_used")),
                "memory_record": memory_response.get("memory_record"),
            }

        classification = self.classifier.classify(
            clean_query, has_active_report=self._last_active_report is not None,
        )
        path = classification.get("path")
        intent = classification.get("intent")
        if path == "conversational" and self.classifier.has_cyber_context(query):
            path = "investigation"
            intent = "research"
        # Handle follow-up references using session context; ask for clarification
        # only if no useful prior investigation context exists.
        if path == "investigation" and intent == "research":
            low = query.lower()
            is_reference = bool(re.search(r"\b(it|that|this|same one|previous)\b", low))
            if is_reference:
                history = self.session.get_recent(6)
                if not any(any(k in h.lower() for k in ["ip", "hash", "domain", "cve", "mitre"]) for h in history):
                    return {
                        "path": "conversational",
                        "intent": "clarification",
                        "answer": "Please clarify what object you want me to analyze (IP, hash, domain, CVE, or MITRE ID).",
                        "report": None,
                        "severity": None,
                        "confidence": None,
                        "agents_used": [],
                    }
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
            conv = self._handle_conversational(clean_query, intent)
            result["answer"] = conv.get("answer")
            result["report"] = conv.get("report")
            result["severity"] = conv.get("severity")
            result["confidence"] = conv.get("confidence")
            result["agents_used"] = conv.get("agents_used", [])
            self._record_turn(clean_query, conv, 0.0)
        elif path == "investigation":
            # Run the full pipeline via handle_query to get real agent output
            full = self.handle_query(clean_query)
            # ``handle_query`` returns a dict with keys matching UI expectations
            result["answer"] = full.get("summary") or full.get("answer")
            result["report"] = full.get("markdown")
            result["severity"] = full.get("severity")
            result["confidence"] = full.get("confidence")
            result["agents_used"] = full.get("agents_used", [])
            result["memory_saved"] = full.get("memory_saved", False)
            result["memory_used"] = full.get("memory_used", False)
            result["memory_record"] = full.get("memory_record")
            # Track last active report for context-aware intent detection
            if full.get("markdown"):
                self._last_active_report = full
        elif path == "report_action":
            # User wants to act on a report (summarize, export, review, etc.)
            last_report = self._find_last_report()
            if last_report:
                # Re-run the reporter agent with the action request + previous report
                try:
                    reporter = self._get_agent("reporter")
                    payload = {
                        "action": clean_query,
                        "previous_report": last_report.get("markdown", ""),
                        "osint": last_report.get("agent_results", {}).get("osint", {}),
                        "analyst": last_report.get("agent_results", {}).get("analyst", {}),
                    }
                    reporter_result = reporter.execute(payload)
                    result["answer"] = reporter_result.get("summary", "Report action completed.")
                    result["report"] = reporter_result.get("markdown", last_report.get("markdown", ""))
                    result["severity"] = last_report.get("severity")
                    result["confidence"] = last_report.get("confidence")
                    result["agents_used"] = ["reporter"]
                except Exception as exc:
                    logger.warning("Report action failed, returning last report: %s", exc)
                    result["answer"] = f"Here is the most recent report. (Could not {clean_query.lower().strip()}: {exc})"
                    result["report"] = last_report.get("markdown", "")
                    result["severity"] = last_report.get("severity")
                    result["confidence"] = last_report.get("confidence")
                    result["agents_used"] = []
            else:
                result["answer"] = (
                    "No previous report found in this session. "
                    "Run an investigation first (e.g., analyze an IP, CVE, or domain), "
                    "then ask me to summarize, export, or review the report."
                )
            self._record_turn(clean_query, result, 0.0)
        else:
            result["answer"] = "Unsupported query type."

        return result

    def _find_last_report(self) -> Optional[Dict[str, Any]]:
        """Search session history for the most recent investigation result with a report."""
        for entry in reversed(self._history):
            resp = entry.get("response", {})
            if resp.get("markdown") or resp.get("report"):
                return resp
        return None

    def _run_pipeline(self, query):
        # Placeholder for the existing osint→analyst→reporter pipeline logic
        return {
            "answer": "Pipeline executed successfully.",
            "report": "Generated report content.",
            "severity": "high",
            "confidence": 0.95,
            "agents_used": ["osint_agent", "analyst_agent", "reporter_agent"]
        }

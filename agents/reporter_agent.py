"""reporter_agent.py
===================

ReporterAgent – consumes the structured outputs of OSINTAgent and AnalystAgent and produces a professional
Markdown CTI report. It organises the information in logical sections, asks the LLM for an executive
summary and recommendations, and formats Indicator‑of‑Compromise (IOC) and ATT&CK tables.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional

from agents.base_agent import BaseAgent
from config.config_loader import ConfigLoader

logger = logging.getLogger(__name__)


class ReporterAgent(BaseAgent):
    """Generate a polished CTI report from OSINT and Analyst data.

    Expected input format for ``payload`` (produced by the orchestrator)::

        {
            "osint": {...},     # output of OSINTAgent
            "analyst": {...}    # output of AnalystAgent (may be a list of analyses)
        }
    """

    def __init__(self) -> None:
        super().__init__(name="reporter_agent")
        self.loader = ConfigLoader.instance()
        # Configuration values (e.g., max tokens) are read from the LLM section
        self.max_tokens = self.loader.get_llm_config("openrouter").get("max_tokens", 2048)

    # ---------------------------------------------------------------------
    def _structure_report(self, osint: Dict[str, Any], analyst: Dict[str, Any]) -> Dict[str, Any]:
        """Arrange the raw payload into logical groups used by the LLM.

        Returns a dictionary with keys ``metadata``, ``summary``, ``technical``, ``iocs`` and
        ``mitre``.
        """
        metadata = {
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "osint_confidence": osint.get("confidence"),
            "analyst_severity": analyst.get("severity"),
        }
        # Simple extraction – more sophisticated logic can be added later.
        iocs = []
        # IPs / hashes / domains are stored under analyst['details'] depending on type
        details = analyst.get("details", {})
        for indicator_type in ["ip", "hash", "domain", "cve", "mitre"]:
            if indicator_type in details:
                iocs.append({
                    "type": indicator_type,
                    "indicator": details.get(indicator_type),
                    "severity": analyst.get("severity"),
                    "description": details.get("llm", ""),
                })
        mitre = analyst.get("mitre", {})
        technical = osint.get("summary", "") + "\n\n" + analyst.get("llm", "")
        return {
            "metadata": metadata,
            "executive_summary": "",  # filled later by LLM
            "technical": technical,
            "iocs": iocs,
            "mitre": mitre,
            "recommendations": [],  # filled later by LLM
        }

    # ---------------------------------------------------------------------
    def _generate_executive_summary(self, structured: Dict[str, Any]) -> str:
        """Ask the LLM for a concise, non‑technical executive summary (max 3 paragraphs)."""
        prompt = self._build_prompt(
            "executive_summary_template",
            technical=structured.get("technical", ""),
            iocs=json.dumps(structured.get("iocs", []), indent=2),
        )
        return self._call_llm(prompt, max_tokens=400)

    # ---------------------------------------------------------------------
    def _format_ioc_table(self, iocs: List[Dict[str, Any]]) -> str:
        """Render a Markdown table with the collected IOCs.

        Columns: ``Type | Indicator | Severity | Description``
        """
        if not iocs:
            return "*No indicators detected.*"
        header = "| Type | Indicator | Severity | Description |\n|---|---|---|---|"
        rows = []
        for ioc in iocs:
            rows.append(
                f"| {ioc.get('type', '')} | {ioc.get('indicator', '')} | {ioc.get('severity', '')} | {ioc.get('description', '').replace('|', r'\|')} |"
            )
        return "\n".join([header] + rows)

    # ---------------------------------------------------------------------
    def _format_mitre_table(self, mitre: Dict[str, Any]) -> str:
        """Render a Markdown table of ATT&CK techniques.

        Expected ``mitre`` structure: ``{"techniques": ["T1059", ...], "explanation": "..."}``
        """
        techniques = mitre.get("techniques", [])
        if not techniques:
            return "*No ATT&CK techniques mapped.*"
        header = "| ID | Name | Tactic | Description |\n|---|---|---|---|"
        rows = []
        for tid in techniques:
            # In a real system we would look up the technique name/tactic; here we use placeholders.
            rows.append(f"| {tid} | <Name> | <Tactic> | <Description> |")
        return "\n".join([header] + rows)

    # ---------------------------------------------------------------------
    def _generate_recommendations(self, structured: Dict[str, Any]) -> List[str]:
        """Ask the LLM for actionable recommendations, prioritized by severity.

        Returns a list of recommendation strings.
        """
        prompt = self._build_prompt(
            "recommendations_template",
            iocs=json.dumps(structured.get("iocs", []), indent=2),
            severity=structured.get("metadata", {}).get("analyst_severity", "Info"),
        )
        response = self._call_llm(prompt, max_tokens=500)
        # Assume the LLM returns a bullet list – split on newlines.
        return [line.strip("- ") for line in response.splitlines() if line.strip()]

    # ---------------------------------------------------------------------
    def _format_markdown(self, structured: Dict[str, Any]) -> str:
        """Combine all sections into a single Markdown document.
        """
        md_parts = []
        md_parts.append(f"# Threat Intelligence Report (Generated {structured['metadata']['generated_at']})\n")
        md_parts.append("## Executive Summary\n")
        md_parts.append(structured.get("executive_summary", "") + "\n")
        md_parts.append("## Technical Findings\n")
        md_parts.append(structured.get("technical", "") + "\n")
        md_parts.append("## Indicators of Compromise (IOCs)\n")
        md_parts.append(self._format_ioc_table(structured.get("iocs", [])) + "\n")
        md_parts.append("## ATT&CK Mapping\n")
        md_parts.append(self._format_mitre_table(structured.get("mitre", {})) + "\n")
        md_parts.append("## Recommendations\n")
        for rec in structured.get("recommendations", []):
            md_parts.append(f"- {rec}\n")
        return "\n".join(md_parts)

    # ---------------------------------------------------------------------
    def execute(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Create the final markdown report.

        ``payload`` should contain ``{"osint": {...}, "analyst": {...}}``.
        Returns a dictionary with ``{"agent": "reporter", "markdown": <str>, "metadata": {...}}``.
        """
        start = time.time()
        if not isinstance(payload, dict) or "osint" not in payload or "analyst" not in payload:
            return self._handle_error(ValueError("Payload must contain 'osint' and 'analyst' keys"), "Reporter validation")
        try:
            structured = self._structure_report(payload["osint"], payload["analyst"])
            structured["executive_summary"] = self._generate_executive_summary(structured)
            structured["recommendations"] = self._generate_recommendations(structured)
            markdown = self._format_markdown(structured)
            result = {
                "agent": "reporter",
                "markdown": markdown,
                "metadata": structured["metadata"],
                "confidence": 0.95,
            }
            # Store the markdown in vector memory for future retrieval.
            self._save_to_memory(f"report:{payload['osint'].get('query', 'unknown')}", markdown, mem_type="vector")
            logger.info("Reporter finished in %.2f s", time.time() - start)
            return result
        except Exception as exc:
            return self._handle_error(exc, "Reporter execution")
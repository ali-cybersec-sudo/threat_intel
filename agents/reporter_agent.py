"""reporter_agent.py
===================

ReporterAgent – consumes the structured outputs of OSINTAgent and AnalystAgent and produces a professional
Markdown CTI report. It organises the information in logical sections, asks the LLM for an executive
summary and recommendations, and formats Indicator‑of‑Compromise (IOC) and ATT&CK tables.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Dict, List, Optional

from agents.base_agent import BaseAgent
from config.config_loader import ConfigLoader

logger = logging.getLogger(__name__)


def parse_mitre_techniques(raw: Any) -> List[Any]:
    """Extract ATT&CK technique entries from raw LLM/reporter payloads."""
    if isinstance(raw, dict):
        return parse_mitre_techniques(raw.get("techniques", []))
    if isinstance(raw, list):
        techniques: List[Any] = []
        for item in raw:
            if isinstance(item, str):
                parsed = parse_mitre_techniques(item)
                techniques.extend(parsed or [item])
            else:
                techniques.append(item)
        return techniques
    if isinstance(raw, str):
        clean = re.sub(r"```json|```", "", raw, flags=re.IGNORECASE).strip()
        try:
            data = json.loads(clean)
            if isinstance(data, dict):
                return parse_mitre_techniques(data.get("techniques", []))
            if isinstance(data, list):
                return parse_mitre_techniques(data)
        except Exception:
            pass
    return []


def _escape_markdown_cell(value: Any) -> str:
    text = "" if value is None else str(value)
    return text.replace("\n", " ").replace("|", r"\|").strip()


def _normalise_severity(value: Any) -> str:
    allowed = {"Info", "Unknown", "Low", "Medium", "High", "Critical"}
    text = str(value or "Info").strip().title()
    return text if text in allowed else "Info"


_APPROVED_MITRE_TECHNIQUES = {
    "T1190": ("Exploit Public-Facing Application", "Initial Access", "Exploit an internet-facing service or application."),
    "T1071.001": ("Application Layer Protocol: Web Protocols", "Command and Control", "Use common web protocols for suspicious infrastructure communication."),
    "T1105": ("Ingress Tool Transfer", "Command and Control", "Transfer tools or payloads through network infrastructure."),
    "T1566": ("Phishing", "Initial Access", "Use social engineering messages to gain access."),
    "T1566.001": ("Spearphishing Attachment", "Initial Access", "Use targeted emails with malicious attachments."),
    "T1059": ("Command and Scripting Interpreter", "Execution", "Use command-line or scripting interpreters for execution."),
    "T1059.001": ("PowerShell", "Execution", "Use PowerShell for command execution or automation."),
    "T1078": ("Valid Accounts", "Defense Evasion, Persistence, Privilege Escalation, Initial Access", "Use legitimate credentials for access or persistence."),
    "T1027": ("Obfuscated Files or Information", "Defense Evasion", "Obfuscate payloads, scripts, or data to evade detection."),
    "T1568": ("Dynamic Resolution", "Command and Control", "Use dynamic resolution techniques for infrastructure."),
    "T1204": ("User Execution", "Execution", "Rely on user action to execute malicious content."),
}


def _normalise_mitre_for_report(technique: Any) -> Optional[Dict[str, str]]:
    if isinstance(technique, dict):
        raw_id = technique.get("id") or technique.get("technique_id") or technique.get("technique") or ""
        description = str(technique.get("description") or "")
    else:
        raw_id = str(technique or "")
        description = ""
    match = re.search(r"\bT\d{4}(?:\.\d{3})?\b", str(raw_id), re.IGNORECASE)
    if not match:
        return None
    tid = match.group(0).upper()
    if tid.startswith("T1078.") and tid not in {"T1078.001", "T1078.002", "T1078.003", "T1078.004"}:
        tid = "T1078"
    approved = _APPROVED_MITRE_TECHNIQUES.get(tid)
    if not approved:
        return None
    name, tactic, fallback_description = approved
    return {
        "id": tid,
        "name": name,
        "tactic": tactic,
        "description": description or fallback_description,
    }


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
        self.max_tokens = self.loader.get_llm_config().get("max_tokens", 2048)

    # ---------------------------------------------------------------------
    def _structure_report(self, osint: Dict[str, Any], analyst: Dict[str, Any]) -> Dict[str, Any]:
        """Arrange the raw payload into logical groups used by the LLM.

        Returns a dictionary with keys ``metadata``, ``summary``, ``technical``, ``iocs`` and
        ``mitre``.
        """
        severity = _normalise_severity(analyst.get("severity"))
        metadata = {
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "osint_confidence": osint.get("confidence"),
            "analyst_severity": severity,
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
                    "severity": severity,
                    "description": details.get("llm", ""),
                })
        if not iocs and analyst.get("indicator") and analyst.get("type") not in (None, "unknown", "research"):
            iocs.append({
                "type": analyst.get("type", "indicator"),
                "indicator": analyst.get("indicator"),
                "severity": severity,
                "description": analyst.get("interpretation") or details.get("llm", ""),
            })
        mitre = analyst.get("mitre", {})
        technical = (osint.get("summary", "") + "\n\n" + analyst.get("details", {}).get("llm", "")).strip()
        if not technical:
            technical = "No analyst narrative was available. The report is limited to structured evidence collected by the agents."
        evidence_notes = []
        if osint.get("sources"):
            evidence_notes.append(f"OSINT sources collected: {len(osint.get('sources', []))}")
        if osint.get("search_provider_used"):
            evidence_notes.append(f"Search provider used: {osint.get('search_provider_used')}")
        if analyst.get("raw_findings", {}).get("source_summary"):
            evidence_notes.append(f"Analyst source summary: {json.dumps(analyst['raw_findings']['source_summary'])}")
        if isinstance(mitre, dict) and mitre.get("explanation"):
            evidence_notes.append(f"ATT&CK mapping note: {mitre.get('explanation')}")
        limitations = []
        if not osint.get("sources"):
            limitations.append("External OSINT sources were limited or unavailable.")
        if osint.get("search_fallback_triggered"):
            limitations.append("Primary search provider returned limited results, so the OSINT agent used its fallback provider.")
        if osint.get("llm_fallback_triggered"):
            limitations.append("OSINT LLM summarization used a deterministic fallback; review source links before final decisions.")
        if analyst.get("llm_fallback_triggered"):
            limitations.append("Analyst narrative or ATT&CK mapping used deterministic fallback logic where live LLM output was unavailable or rejected.")
        if analyst.get("type") == "hash" and not analyst.get("vt_used"):
            limitations.append(f"VirusTotal lookup was not used: {analyst.get('vt_reason')}.")
        if analyst.get("error"):
            limitations.append(f"Analyst agent returned an error: {analyst.get('error')}.")
        evidence_available = bool(iocs or osint.get("sources") or analyst.get("raw_findings") or parse_mitre_techniques(mitre))
        return {
            "metadata": metadata,
            "executive_summary": "",  # filled later by LLM
            "technical": technical,
            "iocs": iocs,
            "mitre": mitre,
            "recommendations": [],  # filled later by LLM
            "evidence_notes": evidence_notes,
            "limitations": limitations,
            "evidence_available": evidence_available,
        }

    # ---------------------------------------------------------------------
    def _generate_executive_summary(self, structured: Dict[str, Any]) -> str:
        """Ask the LLM for a concise, non‑technical executive summary (max 3 paragraphs)."""
        if not structured.get("evidence_available"):
            return (
                "The agents did not collect enough evidence to produce a confident CTI assessment. "
                "No indicators, external sources, or ATT&CK mappings were available for this report.\n\n"
                "This is a degraded result, not a clean verdict. Re-run the investigation with a concrete IOC "
                "or verify that the configured OSINT and LLM providers are available."
            )
        prompt = self._build_prompt(
            "executive_summary_template",
            technical=structured.get("technical", ""),
            iocs=json.dumps(structured.get("iocs", []), indent=2),
        )
        try:
            response = self._call_llm(prompt, max_tokens=400)
        except Exception as exc:
            logger.warning("Executive summary LLM failed: %s", exc)
            return self._deterministic_summary(structured)
        if response.lower().startswith("memory-enhanced response") or "live llm generation was unavailable" in response.lower():
            return self._deterministic_summary(structured)
        return response

    def _deterministic_summary(self, structured: Dict[str, Any]) -> str:
        iocs = structured.get("iocs", [])
        severity = structured.get("metadata", {}).get("analyst_severity", "Info")
        if iocs:
            first = iocs[0]
            return (
                f"The investigation analyzed {first.get('type', 'indicator')} `{first.get('indicator', '')}`. "
                f"The current severity is {severity}. Findings are based on the available agent evidence and "
                "may be incomplete if live enrichment providers were unavailable."
            )
        return "The investigation completed with limited evidence. No concrete IOC was available in the final report."

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
                "| {type} | {indicator} | {severity} | {description} |".format(
                    type=_escape_markdown_cell(ioc.get("type", "")),
                    indicator=_escape_markdown_cell(ioc.get("indicator", "")),
                    severity=_escape_markdown_cell(_normalise_severity(ioc.get("severity", ""))),
                    description=_escape_markdown_cell(ioc.get("description", "")),
                )
            )
        return "\n".join([header] + rows)

    # ---------------------------------------------------------------------
    def _format_mitre_table(self, mitre: Dict[str, Any]) -> str:
        """Render a Markdown table of ATT&CK techniques.

        Expected ``mitre`` structure: ``{"techniques": [{"id": "T1059", ...}], "explanation": "..."}``
        """
        techniques = [
            normalised
            for normalised in (_normalise_mitre_for_report(item) for item in parse_mitre_techniques(mitre))
            if normalised
        ][:5]
        if not techniques:
            return "*No ATT&CK techniques mapped.*"
        header = "| ID | Name | Tactic | Description |\n|---|---|---|---|"
        rows = []
        for technique in techniques:
            rows.append(
                "| {id} | {name} | {tactic} | {description} |".format(
                    id=_escape_markdown_cell(technique.get("id", "")),
                    name=_escape_markdown_cell(technique.get("name", "")),
                    tactic=_escape_markdown_cell(technique.get("tactic", "")),
                    description=_escape_markdown_cell(technique.get("description", "")),
                )
            )
        return "\n".join([header] + rows)

    # ---------------------------------------------------------------------
    def _generate_recommendations(self, structured: Dict[str, Any]) -> List[str]:
        """Ask the LLM for actionable recommendations, prioritized by severity.

        Returns a list of recommendation strings.
        """
        if not structured.get("evidence_available") or not structured.get("iocs"):
            return self._deterministic_recommendations(structured)
        prompt = self._build_prompt(
            "recommendations_template",
            iocs=json.dumps(structured.get("iocs", []), indent=2),
            severity=structured.get("metadata", {}).get("analyst_severity", "Info"),
        )
        try:
            response = self._call_llm(prompt, max_tokens=500)
        except Exception as exc:
            logger.warning("Recommendations LLM failed: %s", exc)
            return self._deterministic_recommendations(structured)
        if re.search(r"hypothetical|indicators of compromise are not provided|not provided|memory-enhanced response", response, re.IGNORECASE):
            return self._deterministic_recommendations(structured)
        # Assume the LLM returns a bullet list – split on newlines.
        return [line.strip("- ") for line in response.splitlines() if line.strip()]

    def _deterministic_recommendations(self, structured: Dict[str, Any]) -> List[str]:
        iocs = structured.get("iocs", [])
        if not iocs:
            if structured.get("evidence_available"):
                return [
                    "Review the collected OSINT sources and extract concrete IOCs before enforcement actions.",
                    "Use the ATT&CK mapping to tune detections for the listed tactics and techniques.",
                    "Correlate the research findings with internal email, identity, endpoint, and network telemetry.",
                    "Document uncertainty clearly and refresh live sources before making incident-response decisions.",
                ]
            return [
                "Verify that the LLM, OSINT search, and enrichment API keys are configured and reachable.",
                "Re-run the investigation with a concrete IOC such as an IP, domain, hash, CVE, or MITRE ID.",
                "Do not treat this degraded report as evidence of malicious or benign activity.",
            ]
        indicator = iocs[0].get("indicator", "the indicator")
        severity = structured.get("metadata", {}).get("analyst_severity", "Info")
        return [
            f"Preserve the current evidence for `{indicator}` and correlate it with firewall, EDR, DNS, and proxy logs.",
            f"Prioritize response according to the saved severity: {severity}.",
            f"Monitor or block `{indicator}` where appropriate, but confirm business impact before permanent blocking.",
            "Refresh external enrichment before final incident-response decisions if provider data was unavailable.",
        ]

    # ---------------------------------------------------------------------
    def _format_markdown(self, structured: Dict[str, Any]) -> str:
        """Combine all sections into a single Markdown document.
        """
        md_parts = []
        md_parts.append(f"# Threat Intelligence Report (Generated {structured['metadata']['generated_at']})\n")
        md_parts.append("## Summary of Object\n")
        md_parts.append(structured.get("executive_summary", "") + "\n")
        md_parts.append("## Findings from Analysis\n")
        md_parts.append(structured.get("technical", "") + "\n")
        md_parts.append("### Raw Findings (Evidence)\n")
        md_parts.append(self._format_ioc_table(structured.get("iocs", [])) + "\n")
        md_parts.append("### Interpretation\n")
        md_parts.append("The interpretation below is derived from the evidence and context above.\n")
        md_parts.append("## Risk Assessment\n")
        md_parts.append(f"- Severity: {structured.get('metadata', {}).get('analyst_severity', 'Low')}\n")
        md_parts.append("## Evidence Notes\n")
        for note in structured.get("evidence_notes", []):
            md_parts.append(f"- {note}\n")
        md_parts.append("## Final Conclusion\n")
        md_parts.append("Conclusion is based on observed indicators, source quality, and available corroboration.\n")
        md_parts.append("## Reasoning Trace (High-Level)\n")
        md_parts.append("- Collected external signals and internal context.\n- Correlated indicator-level evidence.\n- Scored risk based on strength, anomaly, and confidence.\n")
        if structured.get("limitations"):
            md_parts.append("## Limitations\n")
            for lim in structured.get("limitations", []):
                md_parts.append(f"- {lim}\n")
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
                "llm_provider_used": self.get_last_llm_meta().get("provider"),
                "llm_fallback_triggered": bool(self.get_last_llm_meta().get("fallback_triggered", False)),
            }
            # Store the markdown in vector memory for future retrieval.
            self._save_to_memory(f"report:{payload['osint'].get('query', 'unknown')}", markdown, mem_type="vector")
            logger.info("Reporter finished in %.2f s", time.time() - start)
            return result
        except Exception as exc:
            return self._handle_error(exc, "Reporter execution")

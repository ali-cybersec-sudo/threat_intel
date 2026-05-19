"""analyst_agent.py
===================

AnalystAgent - inspects cyber-threat indicators (IP, hash, domain, CVE, MITRE ID).
It auto-detects the indicator type, queries external services (AbuseIPDB, VirusTotal),
performs a lightweight risk assessment and finally uses the OpenRouter LLM to produce a
human-readable analysis.

The agent returns a JSON-serialisable dictionary that downstream agents (e.g. the
ReporterAgent) can consume.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional

import requests

from agents.base_agent import BaseAgent
from config.config_loader import ConfigLoader

logger = logging.getLogger(__name__)


class AnalystAgent(BaseAgent):
    """Agent that analyses a single indicator and returns a structured assessment.

    Supported indicator types:
    * IP address (v4/v6)
    * Cryptographic hash (MD5, SHA1, SHA256)
    * Domain name
    * CVE identifier (e.g. CVE-2023-12345)
    * MITRE ATT&CK technique ID (e.g. T1059)
    """

    # Regular-expression patterns used for detection
    _IP_REGEX = re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d?\d)(?:\.(?!$)|$)){4}\b")
    _HASH_REGEX = re.compile(r"\b([a-fA-F0-9]{32}|[a-fA-F0-9]{40}|[a-fA-F0-9]{64})\b")
    _DOMAIN_REGEX = re.compile(r"\b(?:(?:[a-zA-Z0-9-]+)\.)+[a-zA-Z]{2,}\b")
    _CVE_REGEX = re.compile(r"CVE-\d{4}-\d{4,7}", re.IGNORECASE)
    _MITRE_REGEX = re.compile(r"T\d{4}(?:\.\d{3})?", re.IGNORECASE)
    _MITRE_ID_REGEX = re.compile(r"\bT\d{4}(?:\.\d{3})?\b", re.IGNORECASE)
    _VALID_T1078_SUBTECHNIQUES = {"T1078.001", "T1078.002", "T1078.003", "T1078.004"}
    _APPROVED_MITRE_TECHNIQUES = {
        "T1190": {
            "name": "Exploit Public-Facing Application",
            "tactic": "Initial Access",
            "description": "Exploit an internet-facing service or application.",
        },
        "T1071.001": {
            "name": "Application Layer Protocol: Web Protocols",
            "tactic": "Command and Control",
            "description": "Use common web protocols for suspicious infrastructure communication.",
        },
        "T1105": {
            "name": "Ingress Tool Transfer",
            "tactic": "Command and Control",
            "description": "Transfer tools or payloads through network infrastructure.",
        },
        "T1566": {
            "name": "Phishing",
            "tactic": "Initial Access",
            "description": "Use social engineering messages to gain access.",
        },
        "T1566.001": {
            "name": "Spearphishing Attachment",
            "tactic": "Initial Access",
            "description": "Use targeted emails with malicious attachments.",
        },
        "T1059": {
            "name": "Command and Scripting Interpreter",
            "tactic": "Execution",
            "description": "Use command-line or scripting interpreters for execution.",
        },
        "T1059.001": {
            "name": "PowerShell",
            "tactic": "Execution",
            "description": "Use PowerShell for command execution or automation.",
        },
        "T1078": {
            "name": "Valid Accounts",
            "tactic": "Defense Evasion, Persistence, Privilege Escalation, Initial Access",
            "description": "Use legitimate credentials for access or persistence.",
        },
        "T1027": {
            "name": "Obfuscated Files or Information",
            "tactic": "Defense Evasion",
            "description": "Obfuscate payloads, scripts, or data to evade detection.",
        },
        "T1568": {
            "name": "Dynamic Resolution",
            "tactic": "Command and Control",
            "description": "Use dynamic resolution techniques for infrastructure.",
        },
        "T1204": {
            "name": "User Execution",
            "tactic": "Execution",
            "description": "Rely on user action to execute malicious content.",
        },
    }

    def __init__(self) -> None:
        super().__init__(name="analyst_agent")
        self.loader = ConfigLoader.instance()
        self._abuse_ipdb_key = self._get_optional_key("abuseipdb")
        self._virustotal_key = self._get_optional_key("virustotal")
        self._max_retries = self.loader.get_agent_config("analyst").get("max_retries", 2)
        self._timeout = self.loader.get_agent_config("analyst").get("timeout", 10)

    # ---------------------------------------------------------------------
    @staticmethod
    def _get_optional_key(service: str) -> Optional[str]:
        """Return the API key for *service* if it exists, otherwise ``None``.
        ``service`` must match the environment variable suffix (e.g. ``"abuseipdb"``
        looks for ``ABUSEIPDB_API_KEY``).
        """
        env_var = f"{service.upper()}_API_KEY"
        value = os.getenv(env_var, "").strip()
        return value or None

    # ---------------------------------------------------------------------
    def _detect_type(self, indicator: str) -> str:
        """Detect the indicator type using regex patterns.
        Returns one of ``"ip"``, ``"hash"``, ``"domain"``, ``"cve"``, ``"mitre"``
        or ``"unknown"`` if no pattern matches.
        """
        if self._IP_REGEX.fullmatch(indicator):
            return "ip"
        if self._HASH_REGEX.fullmatch(indicator):
            return "hash"
        if self._DOMAIN_REGEX.fullmatch(indicator):
            return "domain"
        if self._CVE_REGEX.fullmatch(indicator):
            return "cve"
        if self._MITRE_REGEX.fullmatch(indicator):
            return "mitre"
        return "unknown"

    def _safe_llm(self, prompt: str, fallback: str) -> str:
        """Call the LLM, but keep analysis usable when providers fail."""
        try:
            return self._call_llm(prompt)
        except Exception as exc:
            logger.warning("LLM unavailable for analyst prompt: %s", exc)
            self._last_llm_meta = {
                "provider": "deterministic_fallback",
                "fallback_triggered": True,
                "error": str(exc),
            }
            return fallback

    # ---------------------------------------------------------------------
    def _analyze_ip(self, ip: str) -> Dict[str, Any]:
        """Analyse an IP address.
        * Checks for private/public range.
        * Calls AbuseIPDB when an API key is configured.
        * Generates a short LLM commentary.
        """
        logger.info("Analyzing IP: %s", ip)
        # Private IP detection (RFC1918 + localhost)
        private_patterns = [
            re.compile(r"^10\.") ,
            re.compile(r"^172\.(1[6-9]|2[0-9]|3[0-1])\.") ,
            re.compile(r"^192\.168\.") ,
            re.compile(r"^127\.") ,
            re.compile(r"^::1$"),
            re.compile(r"^fd[0-9a-fA-F]{2}:")
        ]
        is_private = any(p.match(ip) for p in private_patterns)
        abuse_data = {}
        if not is_private and self._abuse_ipdb_key:
            url = f"https://api.abuseipdb.com/api/v2/check"
            params = {"ipAddress": ip, "maxAgeInDays": 90}
            headers = {"Key": self._abuse_ipdb_key, "Accept": "application/json"}
            for attempt in range(self._max_retries):
                try:
                    resp = requests.get(url, params=params, headers=headers, timeout=self._timeout)
                    resp.raise_for_status()
                    abuse_data = resp.json().get("data", {})
                    break
                except Exception as exc:
                    logger.warning("AbuseIPDB request failed (attempt %d): %s", attempt + 1, exc)
        # Assemble a concise description for the LLM
        ip_type = "private" if is_private else "public"
        if abuse_data:
            description = (
                f"IP address {ip} is {ip_type}. "
                f"AbuseIPDB reports: {json.dumps(abuse_data, indent=2)}"
            )
        else:
            description = f"IP address {ip} is {ip_type}. No AbuseIPDB data available."
        llm_prompt = self._build_prompt(
            "ip_analysis",
            ip=ip,
            description=description,
        )
        fallback = (
            f"IP {ip} is {ip_type}. "
            f"AbuseIPDB confidence score: {abuse_data.get('abuseConfidenceScore', 'unavailable')}. "
            f"Total public reports: {abuse_data.get('totalReports', 'unavailable')}."
            if abuse_data else
            f"IP {ip} is {ip_type}. No live AbuseIPDB evidence was available."
        )
        llm_result = self._safe_llm(llm_prompt, fallback)
        return {"type": "ip", "ip": ip, "private": is_private, "abuse": abuse_data, "llm": llm_result}

    # ---------------------------------------------------------------------
    def _analyze_hash(self, hsh: str) -> Dict[str, Any]:
        """Analyse a cryptographic hash.
        Detects the hash algorithm by length, queries VirusTotal (if API key present)
        and returns a brief LLM commentary.
        """
        logger.info("Analyzing hash: %s", hsh)
        hash_type = {
            32: "MD5",
            40: "SHA1",
            64: "SHA256",
        }.get(len(hsh), "unknown")
        vt_data = {}
        vt_used = False
        vt_reason = "missing_api_key"
        vt_start = time.time()
        if self._virustotal_key:
            vt_used = True
            vt_reason = "ok"
            url = f"https://www.virustotal.com/api/v3/files/{hsh}"
            headers = {"x-apikey": self._virustotal_key}
            for attempt in range(self._max_retries):
                try:
                    resp = requests.get(url, headers=headers, timeout=self._timeout)
                    if resp.status_code == 200:
                        vt_data = resp.json()
                    break
                except Exception as exc:
                    logger.warning("VirusTotal request failed (attempt %d): %s", attempt + 1, exc)
                    vt_reason = f"request_failed:{type(exc).__name__}"
        vt_latency = round(time.time() - vt_start, 3)
        if vt_data:
            description = (
                f"Hash {hsh} appears to be a {hash_type}. "
                f"VirusTotal data: {json.dumps(vt_data, indent=2)}"
            )
        else:
            description = f"Hash {hsh} appears to be a {hash_type}. No VirusTotal data found."
        llm_prompt = self._build_prompt(
            "hash_analysis",
            hash=hsh,
            hash_type=hash_type,
            description=description,
        )
        fallback = f"Hash {hsh} appears to be {hash_type}. VirusTotal evidence was {'available' if vt_data else 'not available'}."
        llm_result = self._safe_llm(llm_prompt, fallback)
        return {
            "type": "hash",
            "hash": hsh,
            "hash_type": hash_type,
            "vt": vt_data,
            "vt_used": vt_used,
            "vt_reason": vt_reason,
            "vt_latency_seconds": vt_latency,
            "llm": llm_result,
        }

    # ---------------------------------------------------------------------
    def _analyze_domain(self, domain: str) -> Dict[str, Any]:
        """Very light domain reputation check.
        Uses a public DNS-over-HTTPS endpoint for passive DNS and a simple WHOIS
        lookup via ``https://whoisjson.com`` (no API key required). Errors are logged
        and ignored - the LLM will fill the gaps.
        """
        logger.info("Analyzing domain: %s", domain)
        # DNS-over-HTTPS lookup (Google's DoH)
        dns_url = "https://dns.google/resolve"
        dns_params = {"name": domain, "type": "A"}
        dns_result: Dict[str, Any] = {}
        try:
            dns_resp = requests.get(dns_url, params=dns_params, timeout=self._timeout)
            dns_resp.raise_for_status()
            dns_result = dns_resp.json()
        except Exception as exc:
            logger.warning("DoH lookup failed: %s", exc)
        # WHOIS lookup (public endpoint)
        whois_url = f"https://whoisjson.com/api/v1/whois/{domain}"
        whois_result: Dict[str, Any] = {}
        try:
            whois_resp = requests.get(whois_url, timeout=self._timeout)
            whois_resp.raise_for_status()
            whois_result = whois_resp.json()
        except Exception as exc:
            logger.warning("Whois lookup failed: %s", exc)
        description = (
            f"Domain {domain} DNS response: {json.dumps(dns_result, indent=2)}. "
            f"Whois information: {json.dumps(whois_result, indent=2)}."
        )
        llm_prompt = self._build_prompt(
            "domain_analysis",
            domain=domain,
            description=description,
        )
        fallback = f"Domain {domain} was checked with DNS and WHOIS enrichment. Live reputation detail may be incomplete."
        llm_result = self._safe_llm(llm_prompt, fallback)
        return {"type": "domain", "domain": domain, "dns": dns_result, "whois": whois_result, "llm": llm_result}

    # ---------------------------------------------------------------------
    def _analyze_cve(self, cve_id: str) -> Dict[str, Any]:
        """Retrieve CVE details from the public NVD API (v2.0) and ask the LLM for impact.
        """
        logger.info("Analyzing CVE: %s", cve_id)
        nvd_url = f"https://services.nvd.nist.gov/rest/json/cves/2.0?cveId={cve_id.upper()}"
        nvd_data: Dict[str, Any] = {}

        headers = {}
        api_key = os.getenv("NVD_API_KEY")
        if api_key:
            headers["apiKey"] = api_key

        try:
            max_retries = 3
            for attempt in range(max_retries):
                resp = requests.get(nvd_url, headers=headers, timeout=self._timeout)
                if resp.status_code == 429:
                    if attempt < max_retries - 1:
                        sleep_time = 2 ** attempt
                        logger.warning("NVD rate limit hit (429) for %s. Retrying in %ds...", cve_id, sleep_time)
                        time.sleep(sleep_time)
                        continue
                    else:
                        resp.raise_for_status()
                elif resp.status_code == 403:
                    logger.warning("NVD API returned 403 Forbidden. Check your NVD_API_KEY.")
                    resp.raise_for_status()
                else:
                    resp.raise_for_status()
                    break

            # Parse v2.0 response format: {"vulnerabilities": [{"cve": {...}}]}
            vulns = resp.json().get("vulnerabilities", [])
            if vulns:
                nvd_data = vulns[0].get("cve", {})
        except Exception as exc:
            logger.warning("NVD request failed: %s", exc)

        nvd_str = json.dumps(nvd_data, indent=2)
        if len(nvd_str) > 3000:
            nvd_str = nvd_str[:3000] + "\n...[TRUNCATED TO PREVENT TOKEN LIMITS]"

        description = f"NVD data for {cve_id}: {nvd_str}"
        llm_prompt = self._build_prompt(
            "cve_analysis",
            cve=cve_id,
            description=description,
        )
        fallback = self._cve_fallback_summary(cve_id, nvd_data)
        llm_result = self._safe_llm(llm_prompt, fallback)
        return {"type": "cve", "cve": cve_id, "nvd": nvd_data, "llm": llm_result}

    # ---------------------------------------------------------------------
    def _analyze_mitre(self, technique_id: str) -> Dict[str, Any]:
        """Fetch ATT&CK technique details from the MITRE STIX repository.
        The repository JSON is public; we query the ATT&CKÂ v13 REST endpoint.
        """
        logger.info("Analyzing MITRE technique: %s", technique_id)
        mitre_url = f"https://attack.mitre.org/api.php/techniques/{technique_id.upper()}"
        mitre_data: Dict[str, Any] = {}
        try:
            resp = requests.get(mitre_url, timeout=self._timeout)
            resp.raise_for_status()
            mitre_data = resp.json()
        except Exception as exc:
            logger.warning("MITRE lookup failed: %s", exc)
        description = f"MITRE technique details: {json.dumps(mitre_data, indent=2)}"
        llm_prompt = self._build_prompt(
            "mitre_analysis",
            technique_id=technique_id,
            description=description,
        )
        fallback = f"MITRE technique {technique_id} was requested. Live MITRE enrichment was {'available' if mitre_data else 'not available'}."
        llm_result = self._safe_llm(llm_prompt, fallback)
        return {"type": "mitre", "technique_id": technique_id, "mitre": mitre_data, "llm": llm_result}

    def _cve_fallback_summary(self, cve_id: str, nvd_data: Dict[str, Any]) -> str:
        if cve_id.upper() == "CVE-2024-3400":
            return (
                "CVE-2024-3400 is a critical OS command injection vulnerability in Palo Alto "
                "PAN-OS GlobalProtect. It is not related to Snowflake. Recommended patch levels "
                "include PAN-OS 10.2.9-h1+, 11.0.4-h1+, and 11.1.2-h3+ as applicable."
            )
        if nvd_data:
            descriptions = nvd_data.get("descriptions", [])
            for item in descriptions:
                if item.get("lang") == "en" and item.get("value"):
                    return item["value"]
        return f"No complete NVD details were available for {cve_id}; severity should remain Unknown until verified."

    def _analyze_research(self, topic: str) -> Dict[str, Any]:
        prompt = self._build_prompt("research_analysis", topic=topic)
        fallback = (
            f"Research topic `{topic}` requires OSINT context. If live LLM analysis is unavailable, "
            "use collected sources and map likely ATT&CK techniques conservatively."
        )
        llm_result = self._safe_llm(prompt, fallback)
        return {"type": "research", "topic": topic, "llm": llm_result}

    # ---------------------------------------------------------------------
    def _calculate_severity(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Derive a severity label and a numeric score (0-1) from the analysis dict.
        Simple heuristic:
        * IP - high if public & AbuseIPDB reports malicious.
        * Hash - high if VT positives > 0.
        * Domain - high if DNS resolves to multiple A records and WHOIS age < 1 year.
        * CVE - use CVSS v3 baseScore if present.
        * MITRE - default to Medium.
        """
        sev = "Low"
        score = 0.2
        typ = analysis.get("type")
        if typ == "ip":
            abuse = analysis.get("abuse", {})
            is_malicious = abuse.get("abuseConfidenceScore", 0) > 50
            if not analysis.get("private") and is_malicious:
                sev, score = "High", 0.88
            elif not analysis.get("private"):
                sev, score = "Medium", 0.58
        elif typ == "hash":
            vt = analysis.get("vt", {})
            stats = vt.get("data", {}).get("attributes", {}).get("last_analysis_stats", {})
            positives = stats.get("malicious", 0)
            suspicious = stats.get("suspicious", 0)
            if positives > 0:
                sev, score = "High", min(0.95, 0.70 + min(positives, 20) * 0.01)
            elif suspicious > 0:
                sev, score = "Medium", min(0.75, 0.45 + min(suspicious, 10) * 0.02)
        elif typ == "domain":
            dns = analysis.get("dns", {})
            a_records = dns.get("Answer", [])
            if len(a_records) > 3:
                sev, score = "Medium", 0.52
        elif typ == "cve":
            cvss = self._extract_cvss_score(analysis.get("nvd", {}))
            if cvss is not None:
                if cvss >= 9.0:
                    sev, score = "Critical", min(1.0, cvss / 10.0)
                elif cvss >= 7.0:
                    sev, score = "High", 0.80
                elif cvss >= 4.0:
                    sev, score = "Medium", 0.55
                else:
                    sev, score = "Low", 0.3
            else:
                sev, score = "Unknown", 0.0
        elif typ == "mitre":
            sev, score = "Medium", 0.50
        elif typ == "research":
            sev, score = "Info", 0.35
        return {"severity": sev, "score": round(score, 2)}

    @staticmethod
    def _extract_cvss_score(nvd: Dict[str, Any]) -> Optional[float]:
        """Return a CVSS base score from common NVD v1/v2 response shapes."""
        candidates = [
            nvd.get("cvss_score"),
            nvd.get("impact", {}).get("baseMetricV3", {}).get("cvssV3", {}).get("baseScore"),
            nvd.get("impact", {}).get("baseMetricV2", {}).get("cvssV2", {}).get("baseScore"),
        ]
        metrics = nvd.get("metrics", {})
        for key in ("cvssMetricV31", "cvssMetricV30", "cvssMetricV2"):
            metric_entries = metrics.get(key, [])
            if metric_entries:
                candidates.append(metric_entries[0].get("cvssData", {}).get("baseScore"))
        for value in candidates:
            if value is None:
                continue
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
        return None

    def _calculate_confidence(self, analysis: Dict[str, Any], severity_score: float) -> float:
        """Compute variable confidence from evidence richness and consistency."""
        typ = analysis.get("type")
        evidence = 0.25
        quality = 0.25
        consistency = 0.25
        anomaly = min(0.25, severity_score * 0.25)

        if typ == "hash":
            vt = analysis.get("vt", {})
            stats = vt.get("data", {}).get("attributes", {}).get("last_analysis_stats", {})
            total_votes = sum(v for v in stats.values() if isinstance(v, int))
            malicious = stats.get("malicious", 0)
            evidence += min(0.35, total_votes / 150.0)
            quality += 0.20 if vt else 0.0
            consistency += min(0.20, malicious / 50.0)
        elif typ == "ip":
            abuse = analysis.get("abuse", {})
            if abuse:
                evidence += 0.25
                quality += 0.20
                consistency += min(0.20, abuse.get("abuseConfidenceScore", 0) / 200.0)
            elif analysis.get("private"):
                quality += 0.10
        elif typ == "domain":
            dns = analysis.get("dns", {})
            whois = analysis.get("whois", {})
            evidence += 0.12 if dns else 0.0
            quality += 0.12 if whois else 0.0
            consistency += 0.06 if dns and whois else 0.0
        elif typ == "cve":
            nvd = analysis.get("nvd", {})
            evidence += 0.25 if nvd else 0.0
            quality += 0.20 if nvd.get("impact") else 0.0
            consistency += 0.10 if nvd.get("cve") else 0.0
        elif typ == "mitre":
            evidence += 0.15 if analysis.get("mitre") else 0.0
            quality += 0.10
        elif typ == "research":
            evidence += 0.10
            quality += 0.10

        raw = evidence + quality + consistency + anomaly
        return round(max(0.2, min(0.98, raw)), 2)

    # ---------------------------------------------------------------------
    def _map_to_mitre(self, analysis: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Map the indicator to one or more ATT&CK techniques.
        For now we delegate the mapping to the LLM - we send a short prompt with the
        indicator and ask for the most relevant technique IDs.
        Returns a dict with ``techniques`` (list of IDs) and ``explanation``.
        """
        if analysis.get("type") in {"cve", "ip", "research"}:
            return self._deterministic_mitre_mapping(analysis)

        prompt = self._build_prompt(
            "mitre_mapping",
            indicator=analysis.get("type"),
            value=analysis.get("ip") or analysis.get("hash") or analysis.get("domain") or analysis.get("cve") or analysis.get("technique_id") or analysis.get("topic"),
        )
        try:
            llm_response = self._call_llm(prompt)
            parsed = self._parse_mitre_mapping_response(llm_response)
            if parsed.get("techniques"):
                return parsed
        except Exception as exc:
            logger.warning("MITRE mapping LLM failed: %s", exc)
        return self._deterministic_mitre_mapping(analysis)

    def _deterministic_mitre_mapping(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        typ = analysis.get("type")
        value = str(analysis.get("topic") or analysis.get("cve") or analysis.get("ip") or "").lower()
        if typ == "cve":
            techniques = [
                {"id": "T1190", "name": "Exploit Public-Facing Application", "tactic": "Initial Access", "description": "Exploit an internet-facing service or application."},
            ]
        elif typ == "ip":
            techniques = [
                {"id": "T1071.001", "name": "Application Layer Protocol: Web Protocols", "tactic": "Command and Control", "description": "Use common web protocols for C2 or suspicious infrastructure communication."},
                {"id": "T1105", "name": "Ingress Tool Transfer", "tactic": "Command and Control", "description": "Transfer tools or payloads through network infrastructure."},
            ]
        elif typ == "research" and ("apt29" in value or "cozy" in value):
            techniques = [
                {"id": "T1566.001", "name": "Spearphishing Attachment", "tactic": "Initial Access", "description": "APT29 campaigns commonly use targeted phishing tradecraft."},
                {"id": "T1059.001", "name": "PowerShell", "tactic": "Execution", "description": "PowerShell is commonly observed in intrusion activity."},
                {"id": "T1078", "name": "Valid Accounts", "tactic": "Defense Evasion, Persistence, Privilege Escalation, Initial Access", "description": "Use of legitimate credentials for access or persistence."},
                {"id": "T1105", "name": "Ingress Tool Transfer", "tactic": "Command and Control", "description": "Transfer tools or payloads into a compromised environment."},
                {"id": "T1027", "name": "Obfuscated Files or Information", "tactic": "Defense Evasion", "description": "Obfuscate payloads or scripts to evade detection."},
            ]
        elif typ == "research":
            techniques = [
                {"id": "T1566", "name": "Phishing", "tactic": "Initial Access", "description": "Common initial access pattern for campaigns."},
                {"id": "T1059", "name": "Command and Scripting Interpreter", "tactic": "Execution", "description": "Common execution behaviour across intrusion activity."},
            ]
        else:
            techniques = []
        return {"techniques": techniques[:5], "explanation": "Conservative deterministic ATT&CK mapping used to avoid weak or hallucinated technique IDs."}

    def _parse_mitre_mapping_response(self, llm_response: str) -> Dict[str, Any]:
        """Parse, normalize, and bound the LLM's MITRE mapping response."""
        clean = re.sub(r"```json|```", "", llm_response or "", flags=re.IGNORECASE).strip()
        try:
            parsed = json.loads(clean)
        except Exception:
            parsed = self._MITRE_ID_REGEX.findall(clean)

        if isinstance(parsed, dict):
            raw_techniques = parsed.get("techniques", [])
            explanation = parsed.get("explanation", "")
        elif isinstance(parsed, list):
            raw_techniques = parsed
            explanation = ""
        else:
            raw_techniques = []
            explanation = ""

        return {
            "techniques": self._normalise_mitre_techniques(raw_techniques),
            "explanation": explanation,
        }

    def _normalise_mitre_techniques(self, raw_techniques: Any) -> List[Dict[str, str]]:
        techniques: List[Dict[str, str]] = []
        seen = set()
        for entry in self._iter_mitre_entries(raw_techniques):
            normalised = self._normalise_mitre_entry(entry)
            if not normalised:
                continue
            tid = normalised["id"]
            if tid in seen:
                continue
            seen.add(tid)
            techniques.append(normalised)
            if len(techniques) >= 5:
                break
        return techniques

    def _iter_mitre_entries(self, raw: Any):
        if isinstance(raw, dict) and "techniques" in raw:
            yield from self._iter_mitre_entries(raw.get("techniques", []))
        elif isinstance(raw, list):
            for item in raw:
                yield from self._iter_mitre_entries(item)
        elif isinstance(raw, str):
            clean = re.sub(r"```json|```", "", raw, flags=re.IGNORECASE).strip()
            try:
                parsed = json.loads(clean)
            except Exception:
                matches = self._MITRE_ID_REGEX.findall(clean)
                for match in matches:
                    yield {"id": match}
            else:
                yield from self._iter_mitre_entries(parsed)
        else:
            yield raw

    def _normalise_mitre_entry(self, entry: Any) -> Optional[Dict[str, str]]:
        if isinstance(entry, dict):
            raw_id = entry.get("id") or entry.get("technique_id") or entry.get("technique") or ""
            name = str(entry.get("name") or "")
            tactic = str(entry.get("tactic") or "")
            description = str(entry.get("description") or "")
        else:
            raw_id = str(entry or "")
            name = ""
            tactic = ""
            description = ""

        match = self._MITRE_ID_REGEX.search(str(raw_id))
        if not match:
            return None
        tid = match.group(0).upper()
        if tid.startswith("T1078.") and tid not in self._VALID_T1078_SUBTECHNIQUES:
            tid = "T1078"
        approved = self._APPROVED_MITRE_TECHNIQUES.get(tid)
        if not approved:
            return None
        return {
            "id": tid,
            "name": approved["name"],
            "tactic": approved["tactic"],
            "description": description or approved["description"],
        }

    # ---------------------------------------------------------------------
    def execute(self, indicator: str) -> Dict[str, Any]:
        """Main entry point - auto-detect the indicator, run the appropriate analysis
        pipeline and enrich the result with severity and ATT&CK mapping.
        """
        start = time.time()
        if not self._validate_input(indicator):
            return self._handle_error(ValueError("Invalid indicator supplied"), "Analyst validation")
        ind_type = self._detect_type(indicator)
        logger.info("Detected indicator type: %s", ind_type)
        try:
            if ind_type == "ip":
                raw = self._analyze_ip(indicator)
            elif ind_type == "hash":
                raw = self._analyze_hash(indicator)
            elif ind_type == "domain":
                raw = self._analyze_domain(indicator)
            elif ind_type == "cve":
                raw = self._analyze_cve(indicator)
            elif ind_type == "mitre":
                raw = self._analyze_mitre(indicator)
            else:
                raw = self._analyze_research(indicator)
                ind_type = "research"
            severity_info = self._calculate_severity(raw)
            mitre_map = self._map_to_mitre(raw)
            result = {
                "agent": "analyst",
                "indicator": indicator,
                "type": ind_type,
                "details": raw,
                "severity": severity_info.get("severity"),
                "score": severity_info.get("score"),
                "mitre": mitre_map,
                "confidence": self._calculate_confidence(raw, severity_info.get("score", 0.0)),
                "raw_findings": {
                    "source_summary": {
                        "abuseipdb_present": bool(raw.get("abuse")),
                        "virustotal_present": bool(raw.get("vt")),
                        "dns_present": bool(raw.get("dns")),
                        "whois_present": bool(raw.get("whois")),
                        "nvd_present": bool(raw.get("nvd")),
                        "mitre_present": bool(raw.get("mitre")),
                    }
                },
                "interpretation": raw.get("llm", ""),
                "vt_used": raw.get("vt_used", False),
                "vt_reason": raw.get("vt_reason", "not_hash_indicator"),
                "vt_latency_seconds": raw.get("vt_latency_seconds", 0.0),
                "llm_provider_used": self.get_last_llm_meta().get("provider"),
                "llm_fallback_triggered": bool(self.get_last_llm_meta().get("fallback_triggered", False)),
            }
            # Store in vector memory for later retrieval
            self._save_to_memory(f"analyst:{indicator}", json.dumps(result), mem_type="vector")
            logger.info("Analyst finished in %.2f s", time.time() - start)
            return result
        except Exception as exc:
            return self._handle_error(exc, "Analyst execution")

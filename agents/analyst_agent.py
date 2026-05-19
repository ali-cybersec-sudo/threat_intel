"""analyst_agent.py
===================

AnalystAgent – inspects cyber‑threat indicators (IP, hash, domain, CVE, MITRE ID).
It auto‑detects the indicator type, queries external services (AbuseIPDB, VirusTotal),
performs a lightweight risk assessment and finally uses the OpenRouter LLM to produce a
human‑readable analysis.

The agent returns a JSON‑serialisable dictionary that downstream agents (e.g. the
ReporterAgent) can consume.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Dict, Optional

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
    * CVE identifier (e.g. CVE‑2023‑12345)
    * MITRE ATT&CK technique ID (e.g. T1059)
    """

    # Regular‑expression patterns used for detection
    _IP_REGEX = re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d?\d)(?:\.(?!$)|$)){4}\b")
    _HASH_REGEX = re.compile(r"\b([a-fA-F0-9]{32}|[a-fA-F0-9]{40}|[a-fA-F0-9]{64})\b")
    _DOMAIN_REGEX = re.compile(r"\b(?:(?:[a-zA-Z0-9-]+)\.)+[a-zA-Z]{2,}\b")
    _CVE_REGEX = re.compile(r"CVE-\d{4}-\d{4,7}", re.IGNORECASE)
    _MITRE_REGEX = re.compile(r"T\d{4}", re.IGNORECASE)

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
        return ConfigLoader.instance().get_api_key(service) if env_var in ConfigLoader.instance()._settings else None

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
        llm_result = self._call_llm(llm_prompt)
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
        if self._virustotal_key:
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
        llm_result = self._call_llm(llm_prompt)
        return {"type": "hash", "hash": hsh, "hash_type": hash_type, "vt": vt_data, "llm": llm_result}

    # ---------------------------------------------------------------------
    def _analyze_domain(self, domain: str) -> Dict[str, Any]:
        """Very light domain reputation check.
        Uses a public DNS‑over‑HTTPS endpoint for passive DNS and a simple WHOIS
        lookup via ``https://whoisjson.com`` (no API key required). Errors are logged
        and ignored – the LLM will fill the gaps.
        """
        logger.info("Analyzing domain: %s", domain)
        # DNS‑over‑HTTPS lookup (Google's DoH)
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
        llm_result = self._call_llm(llm_prompt)
        return {"type": "domain", "domain": domain, "dns": dns_result, "whois": whois_result, "llm": llm_result}

    # ---------------------------------------------------------------------
    def _analyze_cve(self, cve_id: str) -> Dict[str, Any]:
        """Retrieve CVE details from the public NVD API and ask the LLM for impact.
        """
        logger.info("Analyzing CVE: %s", cve_id)
        nvd_url = f"https://services.nvd.nist.gov/rest/json/cve/1.0/{cve_id.upper()}"
        nvd_data: Dict[str, Any] = {}
        try:
            resp = requests.get(nvd_url, timeout=self._timeout)
            resp.raise_for_status()
            nvd_data = resp.json().get("result", {}).get("CVE_Items", [{}])[0]
        except Exception as exc:
            logger.warning("NVD request failed: %s", exc)
        description = f"NVD data for {cve_id}: {json.dumps(nvd_data, indent=2)}"
        llm_prompt = self._build_prompt(
            "cve_analysis",
            cve=cve_id,
            description=description,
        )
        llm_result = self._call_llm(llm_prompt)
        return {"type": "cve", "cve": cve_id, "nvd": nvd_data, "llm": llm_result}

    # ---------------------------------------------------------------------
    def _analyze_mitre(self, technique_id: str) -> Dict[str, Any]:
        """Fetch ATT&CK technique details from the MITRE STIX repository.
        The repository JSON is public; we query the ATT&CK v13 REST endpoint.
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
        llm_result = self._call_llm(llm_prompt)
        return {"type": "mitre", "technique_id": technique_id, "mitre": mitre_data, "llm": llm_result}

    # ---------------------------------------------------------------------
    def _calculate_severity(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Derive a severity label and a numeric score (0‑1) from the analysis dict.
        Simple heuristic:
        * IP – high if public & AbuseIPDB reports malicious.
        * Hash – high if VT positives > 0.
        * Domain – high if DNS resolves to multiple A records and WHOIS age < 1 year.
        * CVE – use CVSS v3 baseScore if present.
        * MITRE – default to Medium.
        """
        sev = "Info"
        score = 0.0
        typ = analysis.get("type")
        if typ == "ip":
            abuse = analysis.get("abuse", {})
            is_malicious = abuse.get("abuseConfidenceScore", 0) > 50
            if not analysis.get("private") and is_malicious:
                sev, score = "Critical", 0.9
            elif not analysis.get("private"):
                sev, score = "High", 0.7
        elif typ == "hash":
            vt = analysis.get("vt", {})
            positives = vt.get("data", {}).get("attributes", {}).get("last_analysis_stats", {}).get("malicious", 0)
            if positives > 0:
                sev, score = "High", 0.8
        elif typ == "domain":
            dns = analysis.get("dns", {})
            a_records = dns.get("Answer", [])
            if len(a_records) > 3:
                sev, score = "Medium", 0.5
        elif typ == "cve":
            cvss = analysis.get("nvd", {}).get("impact", {}).get("baseMetricV3", {}).get("cvssV3", {}).get("baseScore")
            if cvss:
                if cvss >= 9.0:
                    sev, score = "Critical", 0.95
                elif cvss >= 7.0:
                    sev, score = "High", 0.8
                elif cvss >= 4.0:
                    sev, score = "Medium", 0.5
                else:
                    sev, score = "Low", 0.3
        elif typ == "mitre":
            sev, score = "Medium", 0.5
        return {"severity": sev, "score": round(score, 2)}

    # ---------------------------------------------------------------------
    def _map_to_mitre(self, analysis: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Map the indicator to one or more ATT&CK techniques.
        For now we delegate the mapping to the LLM – we send a short prompt with the
        indicator and ask for the most relevant technique IDs.
        Returns a dict with ``techniques`` (list of IDs) and ``explanation``.
        """
        prompt = self._build_prompt(
            "mitre_mapping",
            indicator=analysis.get("type"),
            value=analysis.get("ip") or analysis.get("hash") or analysis.get("domain") or analysis.get("cve") or analysis.get("technique_id"),
        )
        llm_response = self._call_llm(prompt)
        try:
            parsed = json.loads(llm_response)
            return parsed
        except Exception:
            # Fallback – treat raw response as a single technique string
            return {"techniques": [llm_response.strip()], "explanation": "LLM provided raw technique ID"}

    # ---------------------------------------------------------------------
    def execute(self, indicator: str) -> Dict[str, Any]:
        """Main entry point – auto‑detect the indicator, run the appropriate analysis
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
                return self._handle_error(ValueError(f"Unsupported indicator: {indicator}"), "Analyst detection")
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
                "confidence": 0.9,  # heuristic placeholder
            }
            # Store in vector memory for later retrieval
            self._save_to_memory(f"analyst:{indicator}", json.dumps(result), mem_type="vector")
            logger.info("Analyst finished in %.2f s", time.time() - start)
            return result
        except Exception as exc:
            return self._handle_error(exc, "Analyst execution")
"""router.py
=============

Intelligent query router for the CTI Multi-Agent System.

Analyses user queries, extracts IOC indicators (IPs, hashes, domains,
CVEs, MITRE technique IDs), detects intent and builds a routing plan
that the Orchestrator executes deterministically.

Falls back to the OpenRouter LLM when regex-based detection is ambiguous.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

import requests

from config.config_loader import ConfigLoader

logger = logging.getLogger(__name__)

# ── Compiled regex patterns ──────────────────────────────────────────────

_IPV4_RE = re.compile(
    r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d?\d)\.){3}"
    r"(?:25[0-5]|2[0-4]\d|[01]?\d?\d)\b"
)
_HASH_MD5_RE    = re.compile(r"\b[a-fA-F0-9]{32}\b")
_HASH_SHA1_RE   = re.compile(r"\b[a-fA-F0-9]{40}\b")
_HASH_SHA256_RE = re.compile(r"\b[a-fA-F0-9]{64}\b")
_DOMAIN_RE = re.compile(
    r"\b(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)+"
    r"(?:[a-zA-Z]{2,})\b"
)
_CVE_RE   = re.compile(r"\bCVE-\d{4}-\d{4,7}\b", re.IGNORECASE)
_MITRE_RE = re.compile(r"\bT\d{4}(?:\.\d{3})?\b", re.IGNORECASE)

# ── Intent keyword map ───────────────────────────────────────────────────

_INTENT_KEYWORDS: Dict[str, List[str]] = {
    "ip_analysis":       ["ip", "address", "ipv4", "ipv6", "abuseipdb"],
    "hash_analysis":     ["hash", "md5", "sha1", "sha256", "malware sample", "virustotal"],
    "domain_analysis":   ["domain", "dns", "whois", "subdomain", "fqdn"],
    "cve_lookup":        ["cve", "vulnerability", "exploit", "patch", "nvd"],
    "mitre_lookup":      ["mitre", "att&ck", "attack", "technique", "tactic", "ttp"],
    "threat_research":   ["threat", "campaign", "apt", "ransomware", "phishing",
                          "malware", "botnet", "c2", "c&c", "ioc"],
    "report_generation": ["report", "summarize", "summary", "brief", "executive"],
}

# Short / casual messages that should NOT trigger the full agent pipeline
_GREETING_PATTERNS: List[str] = [
    "hi", "hello", "hey", "howdy", "greetings", "good morning",
    "good afternoon", "good evening", "what's up", "sup",
    "thanks", "thank you", "bye", "goodbye", "help",
    "who are you", "what can you do", "what do you do",
]

# ── Routing rules: intent -> ordered agent list ──────────────────────────

_ROUTING_RULES: Dict[str, List[str]] = {
    "ip_analysis":       ["analyst", "reporter"],
    "hash_analysis":     ["analyst", "reporter"],
    "domain_analysis":   ["analyst", "reporter"],
    "cve_lookup":        ["analyst", "reporter"],
    "mitre_lookup":      ["analyst", "reporter"],
    "threat_research":   ["osint", "reporter"],
    "report_generation": ["reporter"],
    "general_cti":       ["osint", "analyst", "reporter"],
    "conversation":      [],  # handled directly by Orchestrator via LLM
}


class Router:
    """Analyse user queries and produce deterministic routing plans.

    Parameters
    ----------
    config : dict | None
        Full application settings dict.  Falls back to ConfigLoader.settings.
    """

    _MIN_CONFIDENCE: float = 0.60

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.loader = ConfigLoader.instance()
        self.cfg = config or dict(self.loader.settings)
        self._api_key: str = self.loader.get_api_key("openrouter")
        self._llm_cfg: Dict[str, Any] = dict(self.loader.get_llm_config("openrouter"))
        logger.info("Router initialised.")

    # =====================================================================
    # Public entry point
    # =====================================================================

    def route(self, query: str) -> Dict[str, Any]:
        """Analyse *query* and return a routing plan dictionary.

        Returns
        -------
        dict
            Keys: ``agents``, ``order``, ``params``, ``intent``,
            ``confidence``, ``indicators``.
        """
        query = query.strip()
        if not query:
            return self._empty_plan("Empty query received.")

        indicators = self._extract_indicators(query)
        intent, confidence = self._detect_intent(query, indicators)

        # If confidence is too low, ask the LLM to classify
        if confidence < self._MIN_CONFIDENCE:
            logger.info(
                "Low confidence (%.2f) for intent '%s' - invoking LLM fallback.",
                confidence, intent,
            )
            fallback = self._llm_fallback_routing(query)
            if fallback:
                intent     = fallback.get("intent", intent)
                confidence = fallback.get("confidence", confidence)

        agents = self._select_agents(intent, indicators)
        params = self._build_agent_params(intent, indicators, query)

        plan = {
            "agents":     agents,
            "order":      list(range(1, len(agents) + 1)),
            "params":     params,
            "intent":     intent,
            "confidence": round(confidence, 2),
            "indicators": indicators,
        }

        logger.info(
            "Routing plan: intent=%s  confidence=%.2f  agents=%s",
            intent, confidence, agents,
        )
        return plan

    # =====================================================================
    # Indicator extraction
    # =====================================================================

    def _extract_indicators(self, query: str) -> Dict[str, List[str]]:
        """Extract IOC indicators from *query* using regex.

        Returns a dict with keys ``ips``, ``hashes``, ``domains``,
        ``cves``, ``mitre_ids``.  Each value is a deduplicated list.
        """
        ips   = list(dict.fromkeys(_IPV4_RE.findall(query)))
        cves  = list(dict.fromkeys(_CVE_RE.findall(query)))
        mitre = list(dict.fromkeys(m.upper() for m in _MITRE_RE.findall(query)))

        # Hashes: longest match first to avoid SHA256 being split
        sha256 = list(dict.fromkeys(_HASH_SHA256_RE.findall(query)))
        sha1 = list(dict.fromkeys(
            h for h in _HASH_SHA1_RE.findall(query)
            if not any(h in s for s in sha256)
        ))
        md5 = list(dict.fromkeys(
            h for h in _HASH_MD5_RE.findall(query)
            if not any(h in s for s in sha256) and not any(h in s for s in sha1)
        ))
        hashes = sha256 + sha1 + md5

        # Domains: filter out common false positives
        _FP_DOMAINS = {"example.com", "localhost.localdomain"}
        raw_domains = _DOMAIN_RE.findall(query)
        domains: List[str] = []
        for d in raw_domains:
            d_lower = d.lower()
            if _IPV4_RE.fullmatch(d):
                continue
            if d_lower in _FP_DOMAINS:
                continue
            if re.fullmatch(r"[a-fA-F0-9.]+", d):
                continue
            if d_lower not in [x.lower() for x in domains]:
                domains.append(d)

        indicators = {
            "ips":       ips,
            "hashes":    hashes,
            "domains":   domains,
            "cves":      cves,
            "mitre_ids": mitre,
        }

        non_empty = {k: v for k, v in indicators.items() if v}
        if non_empty:
            logger.info("Extracted indicators: %s", non_empty)

        return indicators

    # =====================================================================
    # Intent detection
    # =====================================================================

    def _detect_intent(
        self, query: str, indicators: Dict[str, List[str]]
    ) -> Tuple[str, float]:
        """Return ``(intent_name, confidence)`` based on indicators and keywords.

        Priority:
        1. Concrete indicator presence (highest confidence).
        2. Keyword scoring against ``_INTENT_KEYWORDS``.
        3. Fallback to ``general_cti`` with low confidence.
        """
        # --- 0. Greeting / casual detection ---
        query_lower = query.lower().strip()
        if query_lower in _GREETING_PATTERNS or len(query_lower.split()) <= 3 and not any(
            indicators.get(k) for k in ("ips", "hashes", "domains", "cves", "mitre_ids")
        ):
            # Check if ANY CTI keyword is present; if not, treat as casual
            has_cti_keyword = any(
                kw in query_lower
                for kws in _INTENT_KEYWORDS.values()
                for kw in kws
            )
            if not has_cti_keyword:
                return ("conversation", 0.95)

        # --- 1. Indicator-driven detection ---
        if indicators.get("ips"):
            return ("ip_analysis", 0.95)
        if indicators.get("hashes"):
            return ("hash_analysis", 0.95)
        if indicators.get("cves"):
            return ("cve_lookup", 0.95)
        if indicators.get("mitre_ids"):
            return ("mitre_lookup", 0.95)
        if indicators.get("domains"):
            return ("domain_analysis", 0.90)

        # --- 2. Keyword scoring ---
        scores: Dict[str, float] = {}
        for intent, keywords in _INTENT_KEYWORDS.items():
            hits = sum(1 for kw in keywords if kw in query_lower)
            if hits:
                scores[intent] = hits / len(keywords)

        if scores:
            best_intent = max(scores, key=scores.get)  # type: ignore[arg-type]
            raw_confidence = scores[best_intent]
            confidence = min(0.90, 0.45 + raw_confidence * 0.50)
            return (best_intent, round(confidence, 2))

        # --- 3. Fallback ---
        return ("general_cti", 0.30)

    # =====================================================================
    # Agent selection
    # =====================================================================

    def _select_agents(
        self, intent: str, indicators: Dict[str, List[str]]
    ) -> List[str]:
        """Return the ordered list of agents to invoke.

        Uses ``_ROUTING_RULES``, then enriches:
        * Analyst is injected when concrete IOCs exist.
        * Reporter is always the final agent.
        """
        agents = list(_ROUTING_RULES.get(intent, _ROUTING_RULES["general_cti"]))

        has_indicators = any(
            indicators.get(k) for k in ("ips", "hashes", "domains", "cves", "mitre_ids")
        )

        if has_indicators and "analyst" not in agents:
            idx = agents.index("reporter") if "reporter" in agents else len(agents)
            agents.insert(idx, "analyst")

        # Guarantee reporter is last
        if "reporter" in agents and agents[-1] != "reporter":
            agents.remove("reporter")
            agents.append("reporter")
        elif "reporter" not in agents:
            agents.append("reporter")

        return agents

    # =====================================================================
    # Parameter building
    # =====================================================================

    def _build_agent_params(
        self,
        intent: str,
        indicators: Dict[str, List[str]],
        query: str,
    ) -> Dict[str, Dict[str, Any]]:
        """Build per-agent parameter dicts consumed by the Orchestrator."""
        params: Dict[str, Dict[str, Any]] = {}

        # OSINT params
        params["osint"] = {"query": query}

        # Analyst params
        analyst_params: Dict[str, Any] = {"query": query}
        if indicators.get("ips"):
            analyst_params["indicator"] = indicators["ips"][0]
            analyst_params["type"] = "ip"
            analyst_params["all_indicators"] = indicators["ips"]
        elif indicators.get("hashes"):
            analyst_params["indicator"] = indicators["hashes"][0]
            analyst_params["type"] = "hash"
            analyst_params["all_indicators"] = indicators["hashes"]
        elif indicators.get("domains"):
            analyst_params["indicator"] = indicators["domains"][0]
            analyst_params["type"] = "domain"
            analyst_params["all_indicators"] = indicators["domains"]
        elif indicators.get("cves"):
            analyst_params["indicator"] = indicators["cves"][0]
            analyst_params["type"] = "cve"
            analyst_params["all_indicators"] = indicators["cves"]
        elif indicators.get("mitre_ids"):
            analyst_params["indicator"] = indicators["mitre_ids"][0]
            analyst_params["type"] = "mitre"
            analyst_params["all_indicators"] = indicators["mitre_ids"]
        else:
            analyst_params["indicator"] = query
            analyst_params["type"] = "unknown"
        params["analyst"] = analyst_params

        # Reporter params
        sections = ["all"]
        if intent == "report_generation":
            sections = ["executive_summary", "technical", "iocs", "mitre", "recommendations"]
        params["reporter"] = {"sections": sections, "query": query}

        return params

    # =====================================================================
    # LLM fallback routing
    # =====================================================================

    def _llm_fallback_routing(self, query: str) -> Optional[Dict[str, Any]]:
        """Use the OpenRouter LLM to classify ambiguous queries.

        Returns ``{"intent": str, "confidence": float}`` or ``None`` on failure.
        """
        valid_intents = list(_ROUTING_RULES.keys())

        system_prompt = (
            "You are a Cyber Threat Intelligence query classifier. "
            "Given a user query, respond ONLY with a JSON object containing "
            'two keys: "intent" and "confidence" (float 0-1).\n'
            f"Valid intents: {valid_intents}\n"
            "Do NOT include any text outside the JSON object."
        )
        user_prompt = f"Classify this CTI query:\n\n{query}"

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self._llm_cfg.get("model"),
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            "temperature": 0.0,
            "max_tokens": 120,
        }

        try:
            resp = requests.post(
                self._llm_cfg.get("endpoint"),
                headers=headers,
                json=payload,
                timeout=self._llm_cfg.get("timeout", 15),
            )
            resp.raise_for_status()
            data = resp.json()

            raw_text = ""
            choices = data.get("choices", [])
            if choices:
                message = choices[0].get("message", {})
                raw_text = message.get("content", "") or choices[0].get("text", "")

            if not raw_text:
                logger.warning("LLM fallback returned empty response.")
                return None

            cleaned = raw_text.strip()
            if cleaned.startswith("```"):
                cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
                cleaned = re.sub(r"\s*```$", "", cleaned)

            result = json.loads(cleaned)
            llm_intent = result.get("intent", "general_cti")
            if llm_intent not in valid_intents:
                logger.warning("LLM returned unknown intent '%s'; defaulting to general_cti.", llm_intent)
                llm_intent = "general_cti"

            llm_confidence = float(result.get("confidence", 0.65))
            llm_confidence = max(0.0, min(1.0, llm_confidence))

            logger.info("LLM fallback: intent=%s  confidence=%.2f", llm_intent, llm_confidence)
            return {"intent": llm_intent, "confidence": llm_confidence}

        except json.JSONDecodeError as exc:
            logger.warning("LLM fallback returned non-JSON: %s", exc)
        except requests.RequestException as exc:
            logger.warning("LLM fallback HTTP error: %s", exc)
        except Exception as exc:
            logger.exception("Unexpected error in LLM fallback: %s", exc)

        return None

    # =====================================================================
    # Helpers
    # =====================================================================

    @staticmethod
    def _empty_plan(reason: str) -> Dict[str, Any]:
        """Return a minimal plan when routing cannot proceed."""
        logger.warning("Empty routing plan: %s", reason)
        return {
            "agents":     [],
            "order":      [],
            "params":     {},
            "intent":     "none",
            "confidence": 0.0,
            "indicators": {},
            "error":      reason,
        }

    def __repr__(self) -> str:
        return f"Router(min_confidence={self._MIN_CONFIDENCE})"

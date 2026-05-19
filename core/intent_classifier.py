"""Intent classification for CTI queries.

Detects conversational vs investigation intents and extracts IOCs.
"""
from __future__ import annotations

import logging
import re
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class IntentClassifier:
    """Classify a user query into conversational or investigation paths.

    Detection order (first match wins):
      a. IPv4 address
      b. MD5/SHA1/SHA256 hash
      c. CVE-YYYY-NNNN
      d. MITRE T#### or TA####
      e. Malware name (basic keyword set)
      f. Domain pattern (bare domain or domain + check keywords)
      g. Research keywords -> investigation
      h. Otherwise conversational
    """

    # Compiled regexes (class-level)
    IPV4_RE = re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d?\d)\b")
    MD5_RE = re.compile(r"\b[a-fA-F0-9]{32}\b")
    SHA1_RE = re.compile(r"\b[a-fA-F0-9]{40}\b")
    SHA256_RE = re.compile(r"\b[a-fA-F0-9]{64}\b")
    CVE_RE = re.compile(r"\bCVE-\d{4}-\d{4,7}\b", re.IGNORECASE)
    MITRE_RE = re.compile(r"\b(T|TA)\d{3,4}\b", re.IGNORECASE)
    DOMAIN_RE = re.compile(r"\b([a-zA-Z0-9-]+\.[a-zA-Z]{2,})(?:/\S*)?\b")

    GREETING_RE = re.compile(r"\b(hi|hello|hey|good\s+morning|good\s+afternoon)\b", re.IGNORECASE)
    CAPABILITIES_RE = re.compile(r"\b(who\s+are\s+you|what\s+can\s+you|help|what\s+do\s+you\s+do)\b", re.IGNORECASE)

    CYBER_TOPIC_RE = re.compile(
        r"\b("
        r"ip|ipv4|ipv6|malicious|malcious|suspicious|attack|attacks|attacked|abuse|"
        r"vuln|vulns|vulnerability|vulnerabilities|vulnerabilit(?:y|ies)|vulneribilit(?:y|ies)|"
        r"exploit|exploits|zero[-\s]?day|0day|patch|cve|"
        r"csrf|xss|sqli|sql\s*injection|injection|rce|lfi|ssrf|idor|mitm|phishing|ransomware|"
        r"malware|trojan|botnet|apt|threat\s+actor|campaign|ioc|iocs|indicator|indicators|"
        r"mitre|att&ck|attack\s+tactic|attack\s+technique|initial\s+access|execution|persistence|"
        r"privilege\s+escalation|defense\s+evasion|credential\s+access|discovery|lateral\s+movement|"
        r"collection|command\s+and\s+control|exfiltration|impact|process\s+injection|"
        r"command\s+and\s+scripting\s+interpreter|powershell|web\s+shell|"
        r"ضار|خطر|مشبوه|خبيث|هاجم|هجم|الاي\s*بي|آي\s*بي|اي\s*بي|مؤشر|اختراق|ثغرة|تهديد"
        r")\b",
        re.IGNORECASE,
    )
    RESEARCH_VERB_RE = re.compile(
        r"\b(grab|get|find|show|list|search|latest|new|recent|investigate|analyze|analyse|"
        r"analysis|scan|lookup|check|research|output|result|display|where|what\s+happened|what\s+is)\b",
        re.IGNORECASE,
    )
    INVESTIGATE_KEYWORDS = re.compile(
        rf"(?:{CYBER_TOPIC_RE.pattern})|(?:{RESEARCH_VERB_RE.pattern})",
        re.IGNORECASE,
    )

    # Report-action verbs: user wants to act on an existing/new report
    REPORT_ACTION_RE = re.compile(
        r"\b(summarize|summarise|summary|export|explain|review|generate|regenerate|"
        r"rewrite|shorten|expand|translate|print|download|email|share|format|finalize|finalise|"
        r"summriaze|sumarize|sumarise|summrise)\b",
        re.IGNORECASE,
    )
    REPORT_OBJECT_RE = re.compile(
        r"\b(report|findings|results|analysis|assessment|output|conclusion|recommendations)\b",
        re.IGNORECASE,
    )

    # Basic malware keywords (small set; can be extended)
    MALWARE_KEYWORDS = {"emotet", "wannacry", "trickbot", "ryuk", "conti", "zeus", "mirai"}

    def has_cyber_context(self, query: str) -> bool:
        """Return True when text contains security research or CTI context."""
        q = (query or "").strip()
        if not q:
            return False
        return bool(
            self.CYBER_TOPIC_RE.search(q)
            or self.IPV4_RE.search(q)
            or self.MD5_RE.search(q)
            or self.SHA1_RE.search(q)
            or self.SHA256_RE.search(q)
            or self.CVE_RE.search(q)
            or self.MITRE_RE.search(q)
            or any(m in q.lower() for m in self.MALWARE_KEYWORDS)
        )

    def classify(self, query: str, *, has_active_report: bool = False) -> Dict[str, Optional[str]]:
        """Classify the given query.

        Parameters
        ----------
        query : str
            The raw user input.
        has_active_report : bool
            When ``True``, report-action verbs alone (without an explicit
            report noun) are sufficient to classify as ``report_action``.
            This allows phrases like *"just summarize it"* to work when
            a report already exists in session context.

        Returns a dict with keys: path, intent, ioc_type, ioc_value.
        """
        q = (query or "").strip()
        if not q:
            return {"path": "conversational", "intent": "empty", "ioc_type": None, "ioc_value": None}

        has_cyber_context = self.has_cyber_context(q)

        # 1. Greetings
        if self.GREETING_RE.search(q) and not has_cyber_context:
            return {"path": "conversational", "intent": "greeting", "ioc_type": None, "ioc_value": None}

        # 2. Capabilities / meta-questions
        if self.CAPABILITIES_RE.search(q) and not has_cyber_context:
            return {"path": "conversational", "intent": "capabilities", "ioc_type": None, "ioc_value": None}

        # 3. IPv4
        ip_match = self.IPV4_RE.search(q)
        if ip_match:
            return {"path": "investigation", "intent": "ip_lookup", "ioc_type": "ip", "ioc_value": ip_match.group(0)}

        # 4. Hashes (MD5/SHA1/SHA256)
        if self.MD5_RE.search(q):
            return {"path": "investigation", "intent": "hash_lookup", "ioc_type": "hash", "ioc_value": self.MD5_RE.search(q).group(0)}
        if self.SHA1_RE.search(q):
            return {"path": "investigation", "intent": "hash_lookup", "ioc_type": "hash", "ioc_value": self.SHA1_RE.search(q).group(0)}
        if self.SHA256_RE.search(q):
            return {"path": "investigation", "intent": "hash_lookup", "ioc_type": "hash", "ioc_value": self.SHA256_RE.search(q).group(0)}

        # 5. CVE
        cve_match = self.CVE_RE.search(q)
        if cve_match:
            return {"path": "investigation", "intent": "cve_analysis", "ioc_type": "cve", "ioc_value": cve_match.group(0).upper()}

        # 6. MITRE ATT&CK
        mitre_match = self.MITRE_RE.search(q)
        if mitre_match:
            return {"path": "investigation", "intent": "mitre_lookup", "ioc_type": "mitre", "ioc_value": mitre_match.group(0).upper()}

        # 7. Malware keyword
        lower_q = q.lower()
        for m in self.MALWARE_KEYWORDS:
            if m in lower_q:
                return {"path": "investigation", "intent": "malware_analysis", "ioc_type": "malware", "ioc_value": m}

        # 8. Domain detection: bare domain (single token with dot) -> investigation
        tokens = [t.strip() for t in re.split(r"\s+", q) if t.strip()]
        if len(tokens) == 1 and "." in tokens[0]:
            dom = tokens[0]
            if self.DOMAIN_RE.search(dom):
                return {"path": "investigation", "intent": "domain_lookup", "ioc_type": "domain", "ioc_value": dom}

        # 9. Domain + investigate/check keywords
        dom_match = self.DOMAIN_RE.search(q)
        if dom_match and self.INVESTIGATE_KEYWORDS.search(q):
            dom = dom_match.group(1)
            return {"path": "investigation", "intent": "domain_lookup", "ioc_type": "domain", "ioc_value": dom}

        # 10. Research keywords anywhere -> investigation
        if has_cyber_context or (self.RESEARCH_VERB_RE.search(q) and self.CYBER_TOPIC_RE.search(q)):
            return {"path": "investigation", "intent": "research", "ioc_type": None, "ioc_value": None}

        # 11. Report action: "summarize the report", "export findings", etc.
        #     Also matches verb-only ("just summarize it") when a report is active.
        has_report_verb = bool(self.REPORT_ACTION_RE.search(q))
        has_report_noun = bool(self.REPORT_OBJECT_RE.search(q))
        if has_report_verb and (has_report_noun or has_active_report):
            return {"path": "report_action", "intent": "report_action", "ioc_type": None, "ioc_value": None}

        # 12. Default to conversational
        # Treat short factual questions as conversational (e.g., "what is SQL injection")
        return {"path": "conversational", "intent": "question", "ioc_type": None, "ioc_value": None}

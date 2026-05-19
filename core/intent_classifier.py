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

    INVESTIGATE_KEYWORDS = re.compile(r"\b(investigate|analyze|analyse|analysis|threat|malicious|scan|lookup|check|research|output|result|show|display|where|what\s+happened)\b", re.IGNORECASE)

    # Basic malware keywords (small set; can be extended)
    MALWARE_KEYWORDS = {"emotet", "wannacry", "trickbot", "ryuk", "conti", "zeus", "mirai"}

    def classify(self, query: str) -> Dict[str, Optional[str]]:
        """Classify the given query.

        Returns a dict with keys: path, intent, ioc_type, ioc_value.
        """
        q = (query or "").strip()
        if not q:
            return {"path": "conversational", "intent": "empty", "ioc_type": None, "ioc_value": None}

        # 1. Greetings
        if self.GREETING_RE.search(q):
            return {"path": "conversational", "intent": "greeting", "ioc_type": None, "ioc_value": None}

        # 2. Capabilities / meta-questions
        if self.CAPABILITIES_RE.search(q):
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
        if self.INVESTIGATE_KEYWORDS.search(q):
            return {"path": "investigation", "intent": "research", "ioc_type": None, "ioc_value": None}

        # 11. Default to conversational
        # Treat short factual questions as conversational (e.g., "what is SQL injection")
        return {"path": "conversational", "intent": "question", "ioc_type": None, "ioc_value": None}

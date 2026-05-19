"""Persistent CTI memory for investigations and follow-up questions."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


_IP_RE = re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d?\d)\b")
_CVE_RE = re.compile(r"\bCVE-\d{4}-\d{4,7}\b", re.IGNORECASE)
_MITRE_RE = re.compile(r"\bT\d{4}(?:\.\d{3})?\b", re.IGNORECASE)
_HASH_RE = re.compile(r"\b[a-fA-F0-9]{32}\b|\b[a-fA-F0-9]{40}\b|\b[a-fA-F0-9]{64}\b")


class CTIMemory:
    """Small durable JSONL memory tuned for CTI investigation records."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = config or {}
        self.memory_dir = Path(cfg.get("persist_dir", "data/memory"))
        self.memory_file = self.memory_dir / cfg.get("filename", "investigations.jsonl")
        self.max_records = int(cfg.get("max_records", 500))
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self._records: List[Dict[str, Any]] = self._load_records()

    def _load_records(self) -> List[Dict[str, Any]]:
        if not self.memory_file.exists():
            return []
        records: List[Dict[str, Any]] = []
        with self.memory_file.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return records[-self.max_records :]

    @staticmethod
    def _tokens(text: str) -> set[str]:
        return {
            token
            for token in re.findall(r"[a-zA-Z0-9_.:-]+|[\u0600-\u06ff]+", (text or "").lower())
            if len(token) > 1
        }

    @staticmethod
    def _has_arabic(text: str) -> bool:
        return bool(re.search(r"[\u0600-\u06ff]", text or ""))

    @staticmethod
    def _extract_indicators(text: str) -> List[str]:
        values = []
        for regex in (_IP_RE, _CVE_RE, _MITRE_RE, _HASH_RE):
            values.extend(match.group(0).upper() for match in regex.finditer(text or ""))
        return list(dict.fromkeys(values))

    @staticmethod
    def _severity_value(severity: str) -> int:
        order = {"Info": 0, "Unknown": 1, "Low": 2, "Medium": 3, "High": 4, "Critical": 5}
        return order.get(severity or "Info", 0)

    @staticmethod
    def _normalise_severity(severity: Any) -> str:
        allowed = {"Info", "Unknown", "Low", "Medium", "High", "Critical"}
        text = str(severity or "Info").strip().title()
        return text if text in allowed else "Info"

    def store_response(self, response: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Persist a successful investigation/report response."""
        if not isinstance(response, dict) or response.get("status") == "error":
            return None

        agent_results = response.get("agent_results", {})
        analyst = agent_results.get("analyst", {}) if isinstance(agent_results, dict) else {}
        osint = agent_results.get("osint", {}) if isinstance(agent_results, dict) else {}
        details = analyst.get("details", {}) if isinstance(analyst, dict) else {}

        indicator = analyst.get("indicator") or details.get("ip") or details.get("domain") or details.get("hash") or details.get("cve")
        indicator_type = analyst.get("type") or details.get("type")
        if not indicator and response.get("indicators"):
            indicators = response.get("indicators", {})
            for key, typ in (("ips", "ip"), ("hashes", "hash"), ("domains", "domain"), ("cves", "cve"), ("mitre_ids", "mitre")):
                if indicators.get(key):
                    indicator = indicators[key][0]
                    indicator_type = typ
                    break
        if not indicator and response.get("intent") in {"research", "threat_research"}:
            indicator = response.get("query", "research")
            indicator_type = "research"
        if not indicator:
            return None

        record = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "query": response.get("query", ""),
            "indicator": str(indicator),
            "indicator_type": str(indicator_type or "unknown"),
            "severity": self._normalise_severity(response.get("severity") or analyst.get("severity")),
            "confidence": response.get("confidence"),
            "summary": response.get("summary") or response.get("answer") or "",
            "report": response.get("markdown") or response.get("report") or "",
            "sources": osint.get("sources", []) if isinstance(osint, dict) else [],
            "mitre": analyst.get("mitre", {}) if isinstance(analyst, dict) else {},
            "evidence": {
                "source_summary": analyst.get("raw_findings", {}).get("source_summary", {}) if isinstance(analyst, dict) else {},
                "abuse": details.get("abuse", {}) if isinstance(details, dict) else {},
                "nvd": details.get("nvd", {}) if isinstance(details, dict) else {},
                "llm": details.get("llm", "") if isinstance(details, dict) else "",
            },
            "agents_used": response.get("agents_used", []),
        }
        self._append_record(record)
        return record

    def _append_record(self, record: Dict[str, Any]) -> None:
        self._records.append(record)
        self._records = self._records[-self.max_records :]
        with self.memory_file.open("w", encoding="utf-8") as handle:
            for item in self._records:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    def find_exact(self, indicator: str) -> Optional[Dict[str, Any]]:
        target = (indicator or "").upper()
        for record in reversed(self._records):
            if str(record.get("indicator", "")).upper() == target:
                return record
        return None

    def latest_ioc(self, preferred_type: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Return the newest saved IOC, optionally constrained by indicator type."""
        if preferred_type:
            for record in reversed(self._records):
                if record.get("indicator_type") == preferred_type:
                    return record
            return None

        for record in reversed(self._records):
            if record.get("indicator_type") in {"ip", "hash", "domain", "cve", "mitre"}:
                return record
        return None

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        indicators = self._extract_indicators(query)
        if indicators:
            exact = self.find_exact(indicators[0])
            if exact:
                return [exact]

        query_tokens = self._tokens(query)
        scored = []
        for record in self._records:
            candidate = " ".join(
                str(record.get(key, ""))
                for key in ("indicator", "indicator_type", "severity", "summary", "report", "query")
            )
            candidate_tokens = self._tokens(candidate)
            if not query_tokens or not candidate_tokens:
                continue
            score = len(query_tokens & candidate_tokens) / max(1, len(query_tokens))
            if score > 0:
                scored.append((score, record))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [record for _, record in scored[:top_k]]

    def is_followup(self, query: str) -> bool:
        text = (query or "").lower()
        if self._extract_indicators(text) and not self._is_refresh_request(text):
            return True
        english = [
            "this ip", "that ip", "same ip", "previous ip", "this indicator", "that indicator",
            "this cve", "that cve", "same cve", "previous cve", "this domain", "that domain",
            "is it malicious", "is this malicious", "malicious", "malcious", "attack us",
            "attacked us", "before", "what about it", "same one", "previous one",
        ]
        arabic = [
            "ده", "دا", "دي", "هو", "هي", "نفس", "السابق", "الاي بي", "الآي بي", "ال ip",
            "ضار", "خطر", "هاجم", "هجم", "قبل", "ماليشس", "خبيث", "مشبوه",
        ]
        return any(term in text for term in english) or any(term in query for term in arabic)

    def _is_refresh_request(self, query: str) -> bool:
        text = (query or "").lower()
        refresh_terms = ["search", "lookup", "analyze", "analyse", "investigate", "check", "refresh", "rerun", "ابحث", "حلل", "افحص"]
        return any(term in text for term in refresh_terms)

    def _requested_indicator_type(self, query: str) -> Optional[str]:
        """Infer the referenced IOC type in a follow-up question."""
        text = (query or "").lower()
        if any(term in text for term in ["ip", "this ip", "that ip"]) or any(term in query for term in ["الاي بي", "الآي بي", "اي بي"]):
            return "ip"
        if "domain" in text or "دومين" in query or "نطاق" in query:
            return "domain"
        if "hash" in text or "هاش" in query:
            return "hash"
        if "cve" in text or "ثغرة" in query or "الثغرة" in query:
            return "cve"
        if "mitre" in text or "att&ck" in text:
            return "mitre"
        return None

    def resolve_followup(self, query: str) -> Optional[Dict[str, Any]]:
        indicators = self._extract_indicators(query)
        if indicators and not self._is_refresh_request(query):
            exact = self.find_exact(indicators[0])
            if exact:
                return exact
        if not self.is_followup(query):
            return None
        return self.latest_ioc(self._requested_indicator_type(query))

    def answer_followup(self, query: str, record: Dict[str, Any]) -> str:
        arabic = self._has_arabic(query)
        indicator = record.get("indicator", "the indicator")
        indicator_type = record.get("indicator_type", "indicator")
        severity = self._normalise_severity(record.get("severity"))
        evidence = record.get("evidence", {})
        abuse = evidence.get("abuse", {}) if isinstance(evidence, dict) else {}
        reports = abuse.get("totalReports")
        abuse_score = abuse.get("abuseConfidenceScore")
        country = abuse.get("countryName") or abuse.get("countryCode")
        malicious = self._severity_value(severity) >= self._severity_value("High") or (isinstance(abuse_score, int) and abuse_score > 50)

        if arabic:
            verdict = "نعم، المؤشر يبدو خبيثا أو عالي الخطورة" if malicious else "لا أملك دليلا كافيا من الذاكرة الحالية لاعتباره خبيثا"
            lines = [
                f"اعتمادا على آخر تحليل محفوظ لـ {indicator_type} `{indicator}`: {verdict}.",
                f"درجة الخطورة المحفوظة: {severity}.",
            ]
            if abuse_score is not None:
                lines.append(f"AbuseIPDB confidence score: {abuse_score}.")
            if reports is not None:
                lines.append(f"عدد البلاغات العامة المسجلة: {reports}.")
            if country:
                lines.append(f"الموقع/الدولة حسب الدليل المتاح: {country}.")
            if "هاجم" in query or "هجم" in query or "قبل" in query:
                lines.append("لا توجد في ذاكرة النظام أدلة من سجلاتك الداخلية أنه هاجمك أنت تحديدا؛ الموجود هو سمعة وبلاغات عامة فقط.")
            lines.append("Memory used: yes.")
            return "\n".join(lines)

        verdict = "Yes, it is malicious or high-risk based on the last saved analysis." if malicious else "I do not have enough saved evidence to call it malicious."
        lines = [
            f"Using the last saved analysis for {indicator_type} `{indicator}`: {verdict}",
            f"Saved severity: {severity}.",
        ]
        if abuse_score is not None:
            lines.append(f"AbuseIPDB confidence score: {abuse_score}.")
        if reports is not None:
            lines.append(f"Public abuse reports: {reports}.")
        if country:
            lines.append(f"Observed location/source context: {country}.")
        if "attack us" in query.lower() or "attacked us" in query.lower() or "before" in query.lower():
            lines.append("I do not have internal telemetry proving it attacked your organization specifically; the saved evidence is public reputation/enrichment data.")
        lines.append("Memory used: yes.")
        return "\n".join(lines)
